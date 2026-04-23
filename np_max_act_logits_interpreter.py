import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import torch
from tqdm import tqdm

from neuron_explainer.activations.activation_records import (
    calculate_max_activation,
    format_activation_records,
    non_zero_activation_proportion,
)
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.prompt_builder import PromptBuilder, PromptFormat, Role

DEFAULT_TOKENIZER_PATH = "/mnt/wuyuzhang/models/gemma-2-2b"


def _torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_activations(path: str) -> dict[str, list[dict[str, Any]]]:
    with open(path, "rb") as f:
        head = f.read(1)
    if head in (b"{", b"["):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = _torch_load(path)
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported activation payload type: {type(data)}")
    return data


def load_logits(path: str) -> dict[str, Any]:
    data = _torch_load(path)
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported logits payload type: {type(data)}")
    return data


def build_token_decoder(tokenizer_path: Optional[str]) -> tuple[Callable[[int], str], Optional[str]]:
    if not tokenizer_path:
        return (lambda token_id: f"<id:{token_id}>"), None

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        def decode_fn(token_id: int) -> str:
            return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)

        return decode_fn, None
    except Exception as exc:
        warn = f"Failed to load tokenizer '{tokenizer_path}', fallback to token ids: {exc}"
        return (lambda token_id: f"<id:{token_id}>"), warn


def _safe_feature_dict_get(data: dict[Any, Any], feature_id: int) -> Any:
    if feature_id in data:
        return data[feature_id]
    key = str(feature_id)
    if key in data:
        return data[key]
    return None


def parse_feature_ids(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None or raw.strip() == "":
        return None
    result: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return sorted(set(result))


def to_activation_records(
    feature_payload: list[dict[str, Any]],
    decode_token: Callable[[int], str],
    max_records: Optional[int] = None,
) -> list[ActivationRecord]:
    records: list[ActivationRecord] = []
    limit = len(feature_payload) if max_records is None else min(max_records, len(feature_payload))
    for item in feature_payload[:limit]:
        token_ids = item.get("token_ids", [])
        tokens = item.get("tokens", [])
        feat_acts = item.get("feat_acts", [])
        if not isinstance(feat_acts, list):
            continue

        if isinstance(tokens, list) and len(tokens) == len(feat_acts):
            resolved_tokens = [str(tok) for tok in tokens]
        else:
            if not isinstance(token_ids, list) or len(token_ids) != len(feat_acts):
                continue
            resolved_tokens = [decode_token(int(tid)) for tid in token_ids]

        records.append(
            ActivationRecord(
                tokens=resolved_tokens,
                activations=[float(x) for x in feat_acts],
            )
        )
    return records


def _join_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    if all(t.startswith("<id:") and t.endswith(">") for t in tokens):
        return " ".join(tokens)
    return "".join(tokens)


def _argmax(values: list[float]) -> int:
    best_idx = 0
    best_val = float("-inf")
    for i, v in enumerate(values):
        if v > best_val:
            best_idx = i
            best_val = v
    return best_idx


def _norm_token(token: str) -> str:
    t = token.strip().lower().replace("\n", " ")
    t = t.replace("\u2581", " ")
    return " ".join(t.split())


def extract_top_positive_logits(feature_logits_payload: Optional[dict[str, Any]], top_k: int) -> list[str]:
    if not isinstance(feature_logits_payload, dict):
        return []
    positive = feature_logits_payload.get("positive", [])
    if not isinstance(positive, list):
        return []
    return [str(item.get("token", "")) for item in positive[:top_k] if isinstance(item, dict)]


@dataclass
class FeatureSignals:
    max_activating_tokens: list[str]
    tokens_after_max: list[str]
    top_activating_texts: list[str]
    max_activation: float


def extract_feature_signals(
    activation_records: list[ActivationRecord],
    tokens_around_max_activating_token: int,
) -> FeatureSignals:
    max_tokens: list[str] = []
    tokens_after: list[str] = []
    top_texts: list[str] = []
    max_activation = 0.0

    for record in activation_records:
        if not record.activations:
            continue
        idx = _argmax(record.activations)
        max_activation = max(max_activation, float(record.activations[idx]))
        max_tokens.append(record.tokens[idx])

        if idx + 1 < len(record.tokens):
            tokens_after.append(record.tokens[idx + 1])
        else:
            tokens_after.append("")

        start = max(0, idx - tokens_around_max_activating_token)
        end = min(len(record.tokens), idx + tokens_around_max_activating_token + 1)
        top_texts.append(_join_tokens(record.tokens[start:end]).replace("\n", " "))

    return FeatureSignals(
        max_activating_tokens=max_tokens,
        tokens_after_max=tokens_after,
        top_activating_texts=top_texts,
        max_activation=max_activation,
    )


def postprocess_explanation(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    # Prefer explicitly formatted final answer blocks.
    final_patterns = [
        r"(?im)^\s*FINAL\s*[:?]\s*(.+?)\s*$",
        r"(?im)^\s*ANSWER\s*[:?]\s*(.+?)\s*$",
        r"(?im)^\s*EXPLANATION\s*[:?]\s*(.+?)\s*$",
        r"(?is)<\s*final\s*>\s*(.+?)\s*<\s*/\s*final\s*>",
    ]
    for pat in final_patterns:
        matches = re.findall(pat, cleaned)
        if matches:
            cleaned = str(matches[-1]).strip()
            break

    # Fallback: pick a likely short final line, not a reasoning line.
    if "\n" in cleaned and not re.search(r"(?im)^\s*(FINAL|ANSWER|EXPLANATION)\s*[:?]", cleaned):
        lines = [ln.strip().lstrip("-*0123456789. ").strip() for ln in cleaned.splitlines() if ln.strip()]
        reject_prefixes = (
            "we need", "let's", "analysis", "reasoning", "step", "i think", "i should", "first,", "then,",
        )
        for ln in reversed(lines):
            low = ln.lower()
            if any(low.startswith(p) for p in reject_prefixes):
                continue
            cleaned = ln
            break

    if cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    lower = cleaned.lower()
    prefixes = [
        "final:",
        "answer:",
        "explanation:",
        "this neuron is looking for",
        "this neuron activates for",
        "this feature activates for",
        "the main thing this neuron does is find",
    ]
    for p in prefixes:
        if lower.startswith(p):
            cleaned = cleaned[len(p) :].strip()
            break
    cleaned = cleaned.strip().strip("`\"' ")
    # Keep concise phrase for downstream UI consistency.
    words = cleaned.split()
    if len(words) > 12:
        cleaned = " ".join(words[:12])
    return cleaned


def _heuristic_common_token(tokens: list[str]) -> Optional[str]:
    cleaned = [t for t in (_norm_token(x) for x in tokens) if t]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    token, freq = counts.most_common(1)[0]
    if freq >= max(3, int(0.6 * len(cleaned))):
        return token
    return None


def _is_informative_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return False
    return any(ch.isalnum() for ch in stripped)


def _heuristic_prefix_suffix(tokens: list[str]) -> Optional[str]:
    cleaned = [t for t in (_norm_token(x) for x in tokens) if len(t) >= 3]
    if len(cleaned) < 3:
        return None

    prefix_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    for t in cleaned:
        prefix_counts[t[:3]] += 1
        suffix_counts[t[-3:]] += 1

    pref, pref_count = prefix_counts.most_common(1)[0]
    if pref_count >= max(3, len(cleaned) // 2):
        return f"tokens starting '{pref}'"

    suf, suf_count = suffix_counts.most_common(1)[0]
    if suf_count >= max(3, len(cleaned) // 2):
        return f"tokens ending '{suf}'"

    return None


def heuristic_np_max_act_logits(signals: FeatureSignals, top_positive_logits: list[str]) -> str:
    common_max = _heuristic_common_token(signals.max_activating_tokens)
    if common_max:
        return common_max[:80]

    after_tokens = [t for t in (_norm_token(x) for x in signals.tokens_after_max) if t]
    if after_tokens:
        digit_like = [t for t in after_tokens if any(ch.isdigit() for ch in t)]
        if len(digit_like) >= int(0.8 * len(after_tokens)):
            return "say number-like token"

    pref_suf = _heuristic_prefix_suffix(top_positive_logits)
    if pref_suf:
        return pref_suf

    if top_positive_logits:
        return _norm_token(top_positive_logits[0])[:80] or top_positive_logits[0][:80]
    if signals.max_activating_tokens:
        return _norm_token(signals.max_activating_tokens[0])[:80]
    return "high-activation token pattern"


def heuristic_token_activation_pair(signals: FeatureSignals, activation_records: list[ActivationRecord]) -> str:
    common_max = _heuristic_common_token(signals.max_activating_tokens)
    if common_max and _is_informative_token(common_max):
        return common_max[:80]

    if not activation_records:
        return "activation-context pattern"

    token_weight: Counter[str] = Counter()
    for rec in activation_records:
        for tok, act in zip(rec.tokens, rec.activations):
            nt = _norm_token(tok)
            if nt and _is_informative_token(nt):
                token_weight[nt] += int(max(act, 0.0) * 100)

    if token_weight:
        for top_token, _ in token_weight.most_common(20):
            if _is_informative_token(top_token):
                return top_token[:80]

    return "activation-context pattern"


def heuristic_token_space(tokens: list[str]) -> str:
    common_tok = _heuristic_common_token(tokens)
    if common_tok:
        return common_tok[:80]

    pref_suf = _heuristic_prefix_suffix(tokens)
    if pref_suf:
        return pref_suf

    if tokens:
        return _norm_token(tokens[0])[:80] or tokens[0][:80]
    return "token-list pattern"


class NPMaxActLogitsMethod:
    name = "np_max-act-logits"

    @staticmethod
    def build_messages(
        feature_id: int,
        signals: FeatureSignals,
        top_positive_logits: list[str],
        _: argparse.Namespace,
    ) -> list[dict[str, str]]:
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            "You are explaining the behavior of an SAE feature in a neural network. "
            "Give a concise explanation in 1-6 words.\n\n"
            "Priority order:\n"
            "1) If MAX_ACTIVATING_TOKENS are nearly identical, return that token/pattern.\n"
            "2) If TOKENS_AFTER_MAX_ACTIVATING_TOKEN and TOP_POSITIVE_LOGITS show a clear pattern, "
            "return 'say [pattern]'.\n"
            "3) Otherwise summarize commonality in TOP_POSITIVE_LOGITS.\n"
            "4) If still unclear, infer broad context from TOP_ACTIVATING_TEXTS.\n\n"
            "Rules:\n"
            "- Keep answer concise, mostly 1-3 words.\n"
            "- Do not say 'tokens' or 'patterns'.\n"
            "- Avoid vague phrases like 'words related to'.\n"
            "- If uncertain, make the best specific guess.\n\n"
            "Output format (strict):\n"
            "FINAL: <1-6 words>\n"
            "Do NOT output analysis, steps, rationale, or any extra lines.",
        )

        user_message = f"""
Feature {feature_id}

[START MAX_ACTIVATING_TOKENS]
{chr(10).join(signals.max_activating_tokens)}
[END MAX_ACTIVATING_TOKENS]

[START TOP_ACTIVATING_TEXTS]
{chr(10).join(signals.top_activating_texts)}
[END TOP_ACTIVATING_TEXTS]

[START TOKENS_AFTER_MAX_ACTIVATING_TOKEN]
{chr(10).join(signals.tokens_after_max)}
[END TOKENS_AFTER_MAX_ACTIVATING_TOKEN]

[START TOP_POSITIVE_LOGITS]
{chr(10).join(top_positive_logits)}
[END TOP_POSITIVE_LOGITS]

Return exactly one line:\nFINAL: <1-6 words>"""
        prompt_builder.add_message(Role.USER, user_message)
        messages = prompt_builder.build(PromptFormat.HARMONY_V4)
        assert isinstance(messages, list)
        return messages

    @staticmethod
    def heuristic(signals: FeatureSignals, activation_records: list[ActivationRecord], top_positive_logits: list[str]) -> str:
        return heuristic_np_max_act_logits(signals, top_positive_logits)


class TokenActivationPairMethod:
    name = "token-activation-pair"
    explanation_prefix = "the main thing this neuron does is find"

    @classmethod
    def build_messages(
        cls,
        feature_id: int,
        _: FeatureSignals,
        __: list[str],
        args: argparse.Namespace,
        activation_records: list[ActivationRecord],
    ) -> list[dict[str, str]]:
        max_activation = calculate_max_activation(activation_records)
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            "We're studying neurons in a neural network. Each neuron looks for some particular "
            "thing in a short document. Look at the parts of the document the neuron activates for "
            "and summarize what the neuron is looking for in a short phrase. Don't list "
            "examples of words.\n\nThe activation format is token<tab>activation. Activation "
            "values range from 0 to 10. A neuron finding what it's looking for is represented by a "
            "non-zero activation value. The higher the activation value, the stronger the match.\n\n"
            "Output format (strict):\n"
            "FINAL: <2-8 words>\n"
            "Do NOT output analysis, steps, rationale, or any extra lines.",
        )

        activation_block = format_activation_records(activation_records, max_activation, omit_zeros=False)
        user_message = f"""
Neuron {feature_id}
Activations:{activation_block}"""

        if non_zero_activation_proportion(activation_records, max_activation) < 0.2:
            user_message += (
                "\nSame activations, but with all zeros filtered out:"
                + format_activation_records(activation_records, max_activation, omit_zeros=True)
            )

        user_message += "\nReturn exactly one line:\nFINAL: <2-8 words>"
        prompt_builder.add_message(Role.USER, user_message)
        messages = prompt_builder.build(PromptFormat.HARMONY_V4)
        assert isinstance(messages, list)
        return messages

    @staticmethod
    def heuristic(signals: FeatureSignals, activation_records: list[ActivationRecord], top_positive_logits: list[str]) -> str:
        _ = top_positive_logits
        return heuristic_token_activation_pair(signals, activation_records)


class TokenSpaceRepresentationMethod:
    name = "token-space-representation"

    @staticmethod
    def build_messages(
        feature_id: int,
        _: FeatureSignals,
        token_list: list[str],
        __: argparse.Namespace,
    ) -> list[dict[str, str]]:
        stringified_tokens = ", ".join([f"'{t}'" for t in token_list])
        prompt_builder = PromptBuilder()
        prompt_builder.add_message(
            Role.SYSTEM,
            "We're studying neurons in a neural network. Each neuron looks for some particular "
            "kind of token (which can be a word, or part of a word). Look at the tokens the neuron "
            "activates for (listed below) and summarize what the neuron is looking for in a short phrase. "
            "Don't list examples of words.\n\n"
            "Output format (strict):\n"
            "FINAL: <2-8 words>\n"
            "Do NOT output analysis, steps, rationale, or any extra lines.",
        )
        user_message = (
            f"\n\n\n\nFeature {feature_id}\nTokens:\n{stringified_tokens}\n\n"
            "Return exactly one line:\nFINAL: <2-8 words>"
        )
        prompt_builder.add_message(Role.USER, user_message)
        messages = prompt_builder.build(PromptFormat.HARMONY_V4)
        assert isinstance(messages, list)
        return messages

    @staticmethod
    def heuristic(signals: FeatureSignals, activation_records: list[ActivationRecord], token_list: list[str]) -> str:
        _ = signals, activation_records
        return heuristic_token_space(token_list)


def choose_token_space_tokens(
    signals: FeatureSignals,
    top_positive_logits: list[str],
    top_k: int,
) -> list[str]:
    cleaned_logits = [x for x in top_positive_logits if x]
    if cleaned_logits:
        return cleaned_logits[:top_k]

    # Fallback when logits are unavailable: use frequent max-activating tokens.
    max_tokens = [t for t in signals.max_activating_tokens if t]
    counts = Counter(max_tokens)
    return [tok for tok, _ in counts.most_common(top_k)]


def call_chat_completion(
    *,
    api_base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    url = api_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpret SAE features using multiple methods from neuron-explainer architecture."
    )
    parser.add_argument("--activations-path", default="graduate/gemma2_robust_layer19_activations.pt")
    parser.add_argument("--logits-path", default="graduate/gemma2_robust_layer19_logits.pt")
    parser.add_argument("--output-path", default="graduate/interp_explanations.jsonl")
    parser.add_argument("--summary-path", default="graduate/interp_summary.json")
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)

    parser.add_argument(
        "--method",
        choices=["np_max-act-logits", "token-activation-pair", "token-space-representation", "all"],
        default="np_max-act-logits",
        help="Which explanation method to run. 'all' runs every supported method.",
    )
    parser.add_argument("--mode", choices=["auto", "api", "heuristic"], default="auto")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b-it:free")
    parser.add_argument("--api-base-url", default=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120.0)

    parser.add_argument("--tokens-around", type=int, default=24)
    parser.add_argument("--top-logits-k", type=int, default=10)
    parser.add_argument("--token-space-k", type=int, default=50)
    parser.add_argument("--max-records-per-feature", type=int, default=25)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--feature-offset", type=int, default=0)
    parser.add_argument("--feature-ids", type=str, default=None)
    parser.add_argument("--save-prompts", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    os.makedirs(Path(args.summary_path).parent, exist_ok=True)

    activations = load_activations(args.activations_path)
    feat_to_logits: dict[Any, Any] = {}
    if args.logits_path and os.path.exists(args.logits_path):
        logits_payload = load_logits(args.logits_path)
        maybe_feat_to_logits = logits_payload.get("feat_to_logits", {})
        if isinstance(maybe_feat_to_logits, dict):
            feat_to_logits = maybe_feat_to_logits
    else:
        print(f"[WARN] logits file not found: {args.logits_path}. Some methods will fallback.")

    decode_token, tokenizer_warn = build_token_decoder(args.tokenizer_path)
    if tokenizer_warn:
        print(f"[WARN] {tokenizer_warn}")

    requested_feature_ids = parse_feature_ids(args.feature_ids)
    all_feature_ids = sorted(int(k) for k in activations.keys())
    if requested_feature_ids is None:
        feature_ids = all_feature_ids
    else:
        feature_ids = [f for f in requested_feature_ids if str(f) in activations or f in activations]

    if args.feature_offset > 0:
        feature_ids = feature_ids[args.feature_offset :]
    if args.max_features is not None:
        feature_ids = feature_ids[: args.max_features]
    if not feature_ids:
        raise ValueError("No features selected after filtering.")

    method_registry = {
        "np_max-act-logits": NPMaxActLogitsMethod,
        "token-activation-pair": TokenActivationPairMethod,
        "token-space-representation": TokenSpaceRepresentationMethod,
    }
    methods_to_run = list(method_registry.keys()) if args.method == "all" else [args.method]

    use_api = args.mode == "api" or (args.mode == "auto" and bool(args.api_key))
    if args.mode == "api" and not args.api_key:
        raise ValueError("--mode api requires --api-key or OPENAI_API_KEY")

    started_at = datetime.now(timezone.utc).isoformat()
    failures: list[dict[str, Any]] = []
    stats_by_method: dict[str, dict[str, int]] = defaultdict(
        lambda: {"written": 0, "api_success": 0, "heuristic_success": 0}
    )

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for feature_id in tqdm(feature_ids, desc="Interpreting features"):
            feature_act_payload = _safe_feature_dict_get(activations, feature_id)
            if not isinstance(feature_act_payload, list):
                for method_name in methods_to_run:
                    failures.append({"feature_id": feature_id, "method": method_name, "error": "missing_activation_payload"})
                continue

            activation_records = to_activation_records(
                feature_payload=feature_act_payload,
                decode_token=decode_token,
                max_records=args.max_records_per_feature,
            )
            if not activation_records:
                for method_name in methods_to_run:
                    failures.append({"feature_id": feature_id, "method": method_name, "error": "empty_activation_records"})
                continue

            signals = extract_feature_signals(activation_records, args.tokens_around)
            feature_logits_payload = _safe_feature_dict_get(feat_to_logits, feature_id)
            top_positive_logits = extract_top_positive_logits(feature_logits_payload, args.top_logits_k)
            token_space_tokens = choose_token_space_tokens(signals, top_positive_logits, args.token_space_k)

            for method_name in methods_to_run:
                method_cls = method_registry[method_name]
                try:
                    if method_name == "token-activation-pair":
                        messages = method_cls.build_messages(
                            feature_id=feature_id,
                            _=signals,
                            __=top_positive_logits,
                            args=args,
                            activation_records=activation_records,
                        )
                        heuristic_text = method_cls.heuristic(signals, activation_records, top_positive_logits)
                    elif method_name == "token-space-representation":
                        messages = method_cls.build_messages(feature_id, signals, token_space_tokens, args)
                        heuristic_text = method_cls.heuristic(signals, activation_records, token_space_tokens)
                    else:
                        messages = method_cls.build_messages(feature_id, signals, top_positive_logits, args)
                        heuristic_text = method_cls.heuristic(signals, activation_records, top_positive_logits)

                    if use_api:
                        try:
                            raw = call_chat_completion(
                                api_base_url=args.api_base_url,
                                api_key=str(args.api_key),
                                model=args.model,
                                messages=messages,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                timeout=args.timeout,
                            )
                            explanation_text = postprocess_explanation(raw)
                            mode_used = "api"
                            stats_by_method[method_name]["api_success"] += 1
                        except Exception as exc:
                            if args.mode == "api":
                                raise
                            explanation_text = heuristic_text
                            mode_used = f"heuristic_fallback({type(exc).__name__})"
                            stats_by_method[method_name]["heuristic_success"] += 1
                    else:
                        explanation_text = heuristic_text
                        mode_used = "heuristic"
                        stats_by_method[method_name]["heuristic_success"] += 1

                    record = {
                        "feature_id": feature_id,
                        "method": method_name,
                        "mode_used": mode_used,
                        "explanation": explanation_text,
                        "max_activation": signals.max_activation,
                        "max_activating_tokens": signals.max_activating_tokens,
                        "tokens_after_max_activating_token": signals.tokens_after_max,
                        "top_positive_logits": top_positive_logits,
                    }
                    if method_name == "token-space-representation":
                        record["token_space_tokens"] = token_space_tokens
                    if args.save_prompts:
                        record["messages"] = messages

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats_by_method[method_name]["written"] += 1
                except Exception as exc:
                    failures.append({"feature_id": feature_id, "method": method_name, "error": repr(exc)})

    finished_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "activations_path": args.activations_path,
        "logits_path": args.logits_path,
        "output_path": args.output_path,
        "summary_path": args.summary_path,
        "mode": args.mode,
        "model": args.model,
        "method_arg": args.method,
        "methods_run": methods_to_run,
        "total_selected_features": len(feature_ids),
        "stats_by_method": stats_by_method,
        "num_failures": len(failures),
        "failures": failures[:300],
    }

    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
