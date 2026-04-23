from __future__ import annotations

import gc
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import torch

from np_max_act_logits_interpreter import (
    NPMaxActLogitsMethod,
    TokenActivationPairMethod,
    TokenSpaceRepresentationMethod,
    extract_feature_signals,
    extract_top_positive_logits,
    load_activations,
    load_logits,
    postprocess_explanation,
    to_activation_records,
)

DEFAULT_TOKENIZER_PATH = "/mnt/wuyuzhang/models/gemma-2-2b"


@dataclass
class FeatureDataBundle:
    activations: dict[str, list[dict[str, Any]]]
    logits_payload: dict[str, Any]
    feat_to_logits: dict[Any, Any]
    token_to_feat: Optional[dict[str, Any]]
    activations_path: Optional[str]
    logits_path: Optional[str]
    source: str
    metadata: Optional[dict[str, Any]] = None


# -------------------------
# Loading utilities
# -------------------------


def load_feature_data(
    activations_path: str,
    logits_path: Optional[str],
) -> FeatureDataBundle:
    activations = load_activations(activations_path)

    logits_payload: dict[str, Any] = {}
    feat_to_logits: dict[Any, Any] = {}
    token_to_feat: Optional[dict[str, Any]] = None

    if logits_path and os.path.exists(logits_path):
        logits_payload = load_logits(logits_path)
        maybe_feat_to_logits = logits_payload.get("feat_to_logits", {})
        if isinstance(maybe_feat_to_logits, dict):
            feat_to_logits = maybe_feat_to_logits
        maybe_token_to_feat = logits_payload.get("token_to_feat")
        if isinstance(maybe_token_to_feat, dict):
            token_to_feat = maybe_token_to_feat

    inferred_top_k = None
    if feat_to_logits:
        try:
            first_key = next(iter(feat_to_logits.keys()))
            first_row = feat_to_logits[first_key]
            if isinstance(first_row, dict) and isinstance(first_row.get("positive"), list):
                inferred_top_k = len(first_row.get("positive", []))
        except Exception:
            inferred_top_k = None

    return FeatureDataBundle(
        activations=activations,
        logits_payload=logits_payload,
        feat_to_logits=feat_to_logits,
        token_to_feat=token_to_feat,
        activations_path=activations_path,
        logits_path=logits_path,
        source="load_existing",
        metadata={
            "inferred_top_logits_k": inferred_top_k,
            "activations_path": activations_path,
            "logits_path": logits_path,
        },
    )


def _safe_feature_dict_get(data: dict[Any, Any], feature_id: int) -> Any:
    if feature_id in data:
        return data[feature_id]
    key = str(feature_id)
    if key in data:
        return data[key]
    return None


def get_feature_ids(bundle: FeatureDataBundle) -> list[int]:
    act_ids = {int(k) for k in bundle.activations.keys()}
    if bundle.feat_to_logits:
        logit_ids = {int(k) for k in bundle.feat_to_logits.keys()}
        ids = sorted(act_ids & logit_ids)
        if ids:
            return ids
    return sorted(act_ids)


def get_feature_payloads(
    bundle: FeatureDataBundle,
    feature_id: int,
) -> tuple[Optional[list[dict[str, Any]]], Optional[dict[str, Any]]]:
    act_payload = _safe_feature_dict_get(bundle.activations, feature_id)
    logits_payload = _safe_feature_dict_get(bundle.feat_to_logits, feature_id)
    if not isinstance(act_payload, list):
        act_payload = None
    if not isinstance(logits_payload, dict):
        logits_payload = None
    return act_payload, logits_payload


# -------------------------
# Top logits view helpers
# -------------------------


def get_top_logits_for_feature(
    bundle: FeatureDataBundle,
    feature_id: int,
    top_k: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, logits_payload = get_feature_payloads(bundle, feature_id)
    if not logits_payload:
        return [], []
    pos = logits_payload.get("positive", [])
    neg = logits_payload.get("negative", [])
    pos = pos if isinstance(pos, list) else []
    neg = neg if isinstance(neg, list) else []
    if top_k is not None:
        pos = pos[:top_k]
        neg = neg[:top_k]
    return pos, neg


# -------------------------
# Sequence token-activation view helpers
# -------------------------


def _activation_color(value: float, max_value: float) -> str:
    if max_value <= 0:
        return "rgba(200,200,200,0.15)"
    # Orange palette similar to SAE dashboard heat marks.
    alpha = max(0.0, min(1.0, value / max_value))
    alpha = 0.08 + 0.75 * alpha
    return f"rgba(255, 140, 0, {alpha:.3f})"


def build_sequence_activation_html(
    records: list[dict[str, Any]],
    max_records: int = 8,
) -> str:
    if not records:
        return "<p>No sequence records available.</p>"

    selected = records[:max_records]
    all_acts: list[float] = []
    for r in selected:
        acts_raw = r.get("feat_acts", [])
        if isinstance(acts_raw, torch.Tensor):
            acts = acts_raw.detach().cpu().tolist()
        elif isinstance(acts_raw, tuple):
            acts = list(acts_raw)
        elif isinstance(acts_raw, list):
            acts = acts_raw
        else:
            acts = []
        all_acts.extend(float(x) for x in acts)
    max_act = max([0.0] + all_acts)

    blocks: list[str] = []
    for i, rec in enumerate(selected, start=1):
        tokens_raw = rec.get("tokens", [])
        acts_raw = rec.get("feat_acts", [])
        if isinstance(tokens_raw, torch.Tensor):
            tokens = [str(x) for x in tokens_raw.detach().cpu().tolist()]
        elif isinstance(tokens_raw, tuple):
            tokens = [str(x) for x in tokens_raw]
        elif isinstance(tokens_raw, list):
            tokens = [str(x) for x in tokens_raw]
        else:
            tokens = []

        if isinstance(acts_raw, torch.Tensor):
            acts = [float(x) for x in acts_raw.detach().cpu().tolist()]
        elif isinstance(acts_raw, tuple):
            acts = [float(x) for x in acts_raw]
        elif isinstance(acts_raw, list):
            acts = [float(x) for x in acts_raw]
        else:
            acts = []

        if not tokens or not acts:
            continue
        if len(tokens) != len(acts):
            n = min(len(tokens), len(acts))
            if n <= 0:
                continue
            tokens = tokens[:n]
            acts = acts[:n]

        spans = []
        local_max = max([0.0] + acts)
        for tok, act in zip(tokens, acts):
            tok_text = str(tok).replace("<", "&lt;").replace(">", "&gt;")
            bg = _activation_color(act, max_act)
            fw = "700" if act >= local_max and local_max > 0 else "400"
            spans.append(
                f"<span class='token-chip' title='act={act:.4f}' style='"
                f"display:inline-block;padding:2px 4px;margin:1px;border-radius:4px;"
                f"background:{bg};font-weight:{fw};font-family:monospace;font-size:12px;"
                f"transition:all .12s ease;'>"
                f"{tok_text}</span>"
            )

        blocks.append(
            "<div style='margin-bottom:10px;'>"
            f"<div style='color:#999;font-size:12px;'>Sequence {i}</div>"
            f"<div style='line-height:1.8'>{''.join(spans)}</div>"
            "</div>"
        )

    if not blocks:
        return "<p>No valid sequence records available.</p>"

    style = """
    <style>
      .token-chip:hover {
        outline: 1px solid rgba(255,140,0,.9);
        box-shadow: 0 0 0 1px rgba(255,140,0,.25) inset;
        transform: translateY(-1px);
      }
    </style>
    """
    return style + "".join(blocks)


# -------------------------
# Token -> similar features (find_features_for_token)
# -------------------------


def _to_tensor_or_none(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.tensor(x)
    except Exception:
        return None


def find_features_for_token_query(
    bundle: FeatureDataBundle,
    query_token: Optional[str],
    query_token_id: Optional[int],
    tokenizer_path: Optional[str],
    n_best: int = 10,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not bundle.token_to_feat:
        return [], "`token_to_feat` is missing in logits payload."

    indices = _to_tensor_or_none(bundle.token_to_feat.get("indices"))
    values = _to_tensor_or_none(bundle.token_to_feat.get("values"))
    if indices is None or values is None:
        return [], "Invalid `token_to_feat` structure."

    target_id: Optional[int] = query_token_id
    warn: Optional[str] = None

    if target_id is None:
        if not query_token:
            return [], "Please provide query token or token id."
        if not tokenizer_path:
            return [], "Tokenizer path is required when searching by token text."
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            enc = tok.encode(query_token, add_special_tokens=False)
            if not enc:
                return [], f"Token '{query_token}' not found by tokenizer."
            target_id = int(enc[-1])
        except Exception as exc:
            return [], f"Tokenizer load/encode failed: {exc}"

    if target_id < 0 or target_id >= indices.shape[1]:
        return [], f"token_id={target_id} out of range [0, {indices.shape[1]-1}]"

    feat_indices = indices[:, target_id]
    feat_values = values[:, target_id]

    allowed_feature_ids = set(get_feature_ids(bundle))
    rows: list[dict[str, Any]] = []
    for i in range(int(feat_indices.shape[0])):
        feat_id = int(feat_indices[i].item())
        if allowed_feature_ids and feat_id not in allowed_feature_ids:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "feature": feat_id,
                "logit": float(feat_values[i].item()),
                "token_id": int(target_id),
            }
        )
        if len(rows) >= int(n_best):
            break

    if not rows:
        return [], "No matched features in current load/infer bundle range."

    return rows, warn


# -------------------------
# Explanation generation
# -------------------------


def _normalize_provider(provider: str, api_base_url: str, model: str) -> str:
    p = (provider or "auto").strip().lower()
    if p != "auto":
        return p
    url = (api_base_url or "").lower()
    mdl = (model or "").lower()
    if "openrouter.ai" in url or mdl.endswith(":free"):
        return "openrouter"
    if "api.openai.com" in url:
        return "openai"
    return "custom"


def _extract_chat_content(resp_json: dict[str, Any]) -> str:
    choices = resp_json.get("choices", [])
    if not choices:
        raise ValueError("API response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("Cannot parse assistant content from API response")


def call_chat_completion_adaptive(
    *,
    api_base_url: str,
    api_key: str,
    model: str,
    provider: str,
    messages: list[dict[str, str]],
    openrouter_app_name: Optional[str],
    openrouter_app_url: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    provider_name = _normalize_provider(provider, api_base_url, model)
    url = api_base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider_name == "openrouter":
        # Optional OpenRouter attribution headers.
        if openrouter_app_url:
            headers["HTTP-Referer"] = openrouter_app_url
        if openrouter_app_name:
            headers["X-Title"] = openrouter_app_name

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Prefer bypassing env proxies by default (some hosts set invalid localhost proxies).
    last_exc: Optional[Exception] = None
    response = None
    for trust_env in (False, True):
        try:
            with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
                response = client.post(url, headers=headers, json=payload)
            break
        except httpx.ConnectError as exc:
            last_exc = exc
            continue

    if response is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No response received from API endpoint.")
    response.raise_for_status()
    return _extract_chat_content(response.json())


def generate_feature_explanation(
    bundle: FeatureDataBundle,
    feature_id: int,
    method: str,
    mode: str,
    model: str,
    api_base_url: str,
    api_key: Optional[str],
    api_provider: str,
    openrouter_app_name: Optional[str],
    openrouter_app_url: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: float,
    tokens_around: int,
    top_logits_k: int,
    max_records_per_feature: int,
    save_prompts: bool = False,
) -> dict[str, Any]:
    from np_max_act_logits_interpreter import build_token_decoder

    method_map = {
        "np_max-act-logits": NPMaxActLogitsMethod,
        "token-activation-pair": TokenActivationPairMethod,
        "token-space-representation": TokenSpaceRepresentationMethod,
    }
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")

    decode_token, _ = build_token_decoder(None)
    act_payload, logits_payload = get_feature_payloads(bundle, feature_id)
    if act_payload is None:
        raise ValueError(f"No activation payload for feature {feature_id}")

    activation_records = to_activation_records(
        feature_payload=act_payload,
        decode_token=decode_token,
        max_records=max_records_per_feature,
    )
    if not activation_records:
        raise ValueError(f"No valid activation records for feature {feature_id}")

    signals = extract_feature_signals(activation_records, tokens_around)
    top_positive_logits = extract_top_positive_logits(logits_payload, top_logits_k)

    method_cls = method_map[method]
    if method == "token-activation-pair":
        messages = method_cls.build_messages(
            feature_id=feature_id,
            _=signals,
            __=top_positive_logits,
            args=type("Args", (), {})(),
            activation_records=activation_records,
        )
        heuristic_text = method_cls.heuristic(signals, activation_records, top_positive_logits)
    elif method == "token-space-representation":
        # Token-space method uses tokens list as input; prefer top logits token list.
        token_space_tokens = top_positive_logits[:50] if top_positive_logits else signals.max_activating_tokens[:50]
        messages = method_cls.build_messages(
            feature_id=feature_id,
            _=signals,
            token_list=token_space_tokens,
            __=type("Args", (), {})(),
        )
        heuristic_text = method_cls.heuristic(signals, activation_records, token_space_tokens)
    else:
        messages = method_cls.build_messages(
            feature_id=feature_id,
            signals=signals,
            top_positive_logits=top_positive_logits,
            _=type("Args", (), {})(),
        )
        heuristic_text = method_cls.heuristic(signals, activation_records, top_positive_logits)

    use_api = mode == "api" or (mode == "auto" and bool(api_key))
    mode_used: str

    if use_api:
        try:
            raw = call_chat_completion_adaptive(
                api_base_url=api_base_url,
                api_key=str(api_key),
                model=model,
                provider=api_provider,
                messages=messages,
                openrouter_app_name=openrouter_app_name,
                openrouter_app_url=openrouter_app_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            explanation = postprocess_explanation(raw)
            mode_used = "api"
        except Exception as exc:
            if mode == "api":
                raise
            explanation = heuristic_text
            mode_used = f"heuristic_fallback({type(exc).__name__})"
    else:
        explanation = heuristic_text
        mode_used = "heuristic"

    result = {
        "feature_id": feature_id,
        "method": method,
        "mode_used": mode_used,
        "explanation": explanation,
        "max_activation": signals.max_activation,
        "max_activating_tokens": signals.max_activating_tokens,
        "tokens_after_max_activating_token": signals.tokens_after_max,
        "top_positive_logits": top_positive_logits,
    }
    if save_prompts:
        result["messages"] = messages
    return result


# -------------------------
# Feature steering (inference mode)
# -------------------------


def _load_hooked_model(
    model_path: str,
    tokenizer_path: Optional[str],
    device: str,
):
    resolved_tokenizer_path = tokenizer_path or model_path

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_path, local_files_only=True)
    model_name_for_tlens = Path(model_path).name or model_path
    model = HookedTransformer.from_pretrained(
        model_name_for_tlens,
        hf_model=hf_model,
        device=device,
        dtype=torch.float32,
        center_unembed=False,
        tokenizer=tokenizer,
    )
    model.eval()
    return model, tokenizer


def _prepare_dataset_for_activations_store(dataset_path: str, streaming: bool = True):
    """
    Return dataset argument for ActivationsStore.from_sae:
    - local parquet file/dir -> load via datasets 'parquet'
    - otherwise -> passthrough original string
    """
    p = Path(dataset_path).expanduser()
    info: dict[str, Any] = {"dataset_input": dataset_path, "streaming": bool(streaming)}

    try:
        from datasets import load_dataset
    except Exception:
        info["dataset_loader"] = "passthrough_no_datasets_pkg"
        return dataset_path, info

    try:
        if p.exists() and p.is_file() and p.suffix.lower() == ".parquet":
            ds = load_dataset("parquet", data_files=str(p), split="train", streaming=streaming)
            info.update({"dataset_loader": "parquet_file", "dataset_files": [str(p)]})
            return ds, info

        if p.exists() and p.is_dir():
            parquet_files = sorted(str(x) for x in p.glob("*.parquet") if x.is_file())
            if parquet_files:
                ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=streaming)
                info.update(
                    {
                        "dataset_loader": "parquet_dir",
                        "dataset_num_files": len(parquet_files),
                        "dataset_files_preview": parquet_files[:5],
                    }
                )
                return ds, info
    except Exception as exc:
        info.update({"dataset_loader": "parquet_attempt_failed", "dataset_error": str(exc)})
        return dataset_path, info

    info["dataset_loader"] = "passthrough_string"
    return dataset_path, info


def _get_unembed_weight_cpu(model, hf_model) -> torch.Tensor:
    """
    Prefer model.W_U; fallback to HF output embedding weight if W_U is meta.
    Returns shape [d_model, d_vocab].
    """
    wu = getattr(model, "W_U", None)
    if isinstance(wu, torch.Tensor) and not bool(getattr(wu, "is_meta", False)):
        return wu.detach().cpu()

    out_emb = hf_model.get_output_embeddings()
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise RuntimeError("Cannot resolve unembed matrix: model.W_U is meta and HF output embeddings missing.")
    w = out_emb.weight
    if not isinstance(w, torch.Tensor) or bool(getattr(w, "is_meta", False)):
        raise RuntimeError("Cannot resolve unembed matrix: HF output embedding weight is meta.")
    # HF lm_head weight is typically [vocab, d_model], convert to [d_model, vocab].
    return w.detach().to(dtype=torch.float32).t().cpu()


def _load_sae(
    sae_path: str,
    device: str,
):
    from sae_lens import SAE

    sae = SAE.load_from_disk(sae_path, device=device)
    return sae


def _query_gpu_uuid_to_index() -> dict[str, int]:
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return {}
    mapping: dict[str, int] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except Exception:
            continue
        uuid = parts[1]
        if uuid:
            mapping[uuid] = idx
    return mapping


def _query_gpu_free_mem_mib() -> dict[int, int]:
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return {}
    free_mem: dict[int, int] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            mib = int(float(parts[1]))
        except Exception:
            continue
        free_mem[idx] = mib
    return free_mem


def _query_busy_gpu_indices(ignore_pids: Optional[set[int]] = None) -> set[int]:
    uuid_to_index = _query_gpu_uuid_to_index()
    if not uuid_to_index:
        return set()
    ignore_pids = set(ignore_pids or set())

    try:
        cmd = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return set()

    busy: set[int] = set()
    text = (out.stdout or "").strip()
    if not text:
        return busy

    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        uuid = parts[0]
        pid: Optional[int] = None
        if len(parts) >= 2:
            try:
                pid = int(parts[1])
            except Exception:
                pid = None
        if pid is not None and pid in ignore_pids:
            # Allow reusing GPUs occupied only by current app process.
            continue
        if uuid in uuid_to_index:
            busy.add(int(uuid_to_index[uuid]))
    return busy


_PINNED_VISIBLE_PHYSICAL: Optional[list[int]] = None
_PINNED_BY_APP = False


def _pin_visible_idle_devices(physical_indices: list[int]) -> tuple[list[int], Optional[list[int]]]:
    """
    Pin CUDA_VISIBLE_DEVICES before first CUDA initialization so torch only sees selected GPUs.
    Returns (logical_indices, pinned_physical_indices_or_none).
    """
    global _PINNED_VISIBLE_PHYSICAL, _PINNED_BY_APP

    if not physical_indices:
        return [], None

    if _PINNED_VISIBLE_PHYSICAL:
        # Already pinned in this process; keep stable mapping.
        logical = list(range(len(_PINNED_VISIBLE_PHYSICAL)))
        return logical, list(_PINNED_VISIBLE_PHYSICAL)

    # If user has explicitly configured visibility, respect it.
    env_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if env_visible and not _PINNED_BY_APP:
        return physical_indices, None

    # Can only reliably change visibility before CUDA runtime initialization.
    try:
        if torch.cuda.is_initialized():
            return physical_indices, None
    except Exception:
        return physical_indices, None

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in physical_indices)
    _PINNED_VISIBLE_PHYSICAL = list(physical_indices)
    _PINNED_BY_APP = True
    logical = list(range(len(physical_indices)))
    return logical, list(physical_indices)


def _resolve_device_pair(device: Optional[str], max_devices: int = 2) -> tuple[str, str, dict[str, Any]]:
    req = str(device or "").strip().lower()
    if req.startswith("cuda:") and req != "cuda:auto":
        return req, req, {"mode": "explicit", "selected": [req], "busy_filtered": True}
    if req and req not in {"auto", "cuda:auto", "cuda", "cpu"}:
        return str(device), str(device), {"mode": "explicit", "selected": [str(device)], "busy_filtered": False}
    if req == "cpu":
        return "cpu", "cpu", {"mode": "cpu", "selected": ["cpu"], "busy_filtered": True}

    free_mem_mib = _query_gpu_free_mem_mib()
    if not free_mem_mib:
        return "cpu", "cpu", {"mode": "auto_no_cuda", "selected": ["cpu"], "busy_filtered": True}

    all_indices = sorted(free_mem_mib.keys())
    current_pid = os.getpid()
    parent_pid = os.getppid()
    ignore_pids = {int(current_pid)}
    if isinstance(parent_pid, int) and parent_pid > 0:
        ignore_pids.add(int(parent_pid))
    busy = _query_busy_gpu_indices(ignore_pids=ignore_pids)
    idle = [i for i in all_indices if i not in busy]
    if not idle:
        # User requirement: if a GPU has running process, do not use it.
        return "cpu", "cpu", {
            "mode": "auto_all_busy_to_cpu",
            "selected": ["cpu"],
            "busy_indices": sorted(list(busy)),
            "idle_indices": [],
            "busy_filtered": True,
        }

    idle_sorted = sorted(idle, key=lambda idx: int(free_mem_mib.get(idx, 0)), reverse=True)
    picked_physical = idle_sorted[: max(1, int(max_devices))]
    picked_logical, pinned_physical = _pin_visible_idle_devices(picked_physical)
    if pinned_physical:
        # After pinning, torch uses logical indices 0..N-1
        selected = [f"cuda:{i}" for i in picked_logical]
        model_device = selected[0]
        sae_device = selected[1] if len(selected) >= 2 else model_device
        plan_mode = "auto_idle_only_pinned_visible"
    else:
        selected = [f"cuda:{i}" for i in picked_physical]
        model_device = selected[0]
        sae_device = selected[1] if len(selected) >= 2 else model_device
        plan_mode = "auto_idle_only_unpinned"

    return model_device, sae_device, {
        "mode": plan_mode,
        "selected": [model_device] if sae_device == model_device else [model_device, sae_device],
        "selected_physical": pinned_physical if pinned_physical else picked_physical,
        "pinned_visible": bool(pinned_physical),
        "visible_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "busy_indices": sorted(list(busy)),
        "idle_indices": idle_sorted,
        "idle_free_mem_mib": {int(i): int(free_mem_mib.get(i, 0)) for i in idle_sorted},
        "busy_filtered": True,
    }


def _resolve_device(device: Optional[str]) -> str:
    model_device, _, _ = _resolve_device_pair(device, max_devices=1)
    return model_device


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _get_hook_name_from_sae(sae) -> str:
    hook_name = sae.cfg.metadata['hook_name']
    if hook_name:
        return hook_name
    metadata = getattr(sae.cfg, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("hook_name"):
        return str(metadata["hook_name"])
    raise ValueError("Cannot find SAE hook_name from sae.cfg")


def _top_next_tokens(logits_1d: torch.Tensor, tokenizer, k: int = 10) -> list[dict[str, Any]]:
    vals, ids = torch.topk(logits_1d, k)
    rows: list[dict[str, Any]] = []
    for rank, (v, tid) in enumerate(zip(vals.tolist(), ids.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "token_id": int(tid),
                "token": tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False),
                "logit": float(v),
            }
        )
    return rows


def _generate_with_optional_hook(
    model,
    input_tokens: torch.Tensor,
    max_new_tokens: int,
    hook_name: Optional[str] = None,
    hook_fn=None,
) -> torch.Tensor:
    out = input_tokens
    for _ in range(max_new_tokens):
        if hook_name is None or hook_fn is None:
            logits = model(out)
        else:
            logits = model.run_with_hooks(out, fwd_hooks=[(hook_name, hook_fn)])
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out = torch.cat([out, next_token], dim=1)
    return out


def steer_feature_on_text(
    *,
    model_path: str,
    tokenizer_path: Optional[str],
    sae_path: str,
    feature_id: int,
    prompt_text: str,
    steer_strength: float,
    device: str = "auto",
    max_new_tokens: int = 24,
    top_k_next_tokens: int = 10,
) -> dict[str, Any]:
    """
    Apply simple feature steering by adding `steer_strength` to the selected SAE feature activation
    at the SAE hook point, then compare baseline vs steered next-token logits and continuation.
    """
    model_device, sae_device, device_plan = _resolve_device_pair(device, max_devices=2)
    model = None
    tokenizer = None
    sae = None
    input_tokens = None
    baseline_logits = None
    steered_logits = None
    baseline_gen = None
    steered_gen = None
    baseline_next = None
    steered_next = None

    try:
        model, tokenizer = _load_hooked_model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=model_device,
        )
        sae = _load_sae(sae_path=sae_path, device=sae_device)
        hook_name = _get_hook_name_from_sae(sae)

        input_tokens = model.to_tokens(prompt_text)
        input_tokens = input_tokens.to(model.cfg.device)

        # Baseline forward
        baseline_logits = model(input_tokens)
        baseline_next = baseline_logits[0, -1, :]

        # Steering hook
        def _steer_hook(resid, hook):  # noqa: ARG001
            resid_for_sae = resid.to(sae_device) if str(sae_device) != str(resid.device) else resid
            acts = sae.encode(resid_for_sae)
            if feature_id < 0 or feature_id >= acts.shape[-1]:
                raise ValueError(f"feature_id={feature_id} out of range [0, {acts.shape[-1]-1}]")
            acts[..., feature_id] = acts[..., feature_id] + steer_strength
            decoded = sae.decode(acts)
            return decoded.to(resid.device) if str(decoded.device) != str(resid.device) else decoded

        steered_logits = model.run_with_hooks(input_tokens, fwd_hooks=[(hook_name, _steer_hook)])
        steered_next = steered_logits[0, -1, :]

        # Greedy continuation (preview)
        baseline_gen = _generate_with_optional_hook(model, input_tokens, max_new_tokens=max_new_tokens)
        steered_gen = _generate_with_optional_hook(
            model, input_tokens, max_new_tokens=max_new_tokens, hook_name=hook_name, hook_fn=_steer_hook
        )
        baseline_text = tokenizer.decode(baseline_gen[0].tolist(), clean_up_tokenization_spaces=False)
        steered_text = tokenizer.decode(steered_gen[0].tolist(), clean_up_tokenization_spaces=False)

        # Top next-token comparison
        baseline_top = _top_next_tokens(baseline_next, tokenizer, k=top_k_next_tokens)
        steered_top = _top_next_tokens(steered_next, tokenizer, k=top_k_next_tokens)

        return {
            "feature_id": feature_id,
            "prompt_text": prompt_text,
            "steer_strength": steer_strength,
            "hook_name": hook_name,
            "device": model_device,
            "sae_device": sae_device,
            "device_plan": device_plan,
            "baseline_top_next_tokens": baseline_top,
            "steered_top_next_tokens": steered_top,
            "baseline_generated_text": baseline_text,
            "steered_generated_text": steered_text,
        }
    finally:
        del model, tokenizer, sae, input_tokens, baseline_logits, steered_logits, baseline_gen, steered_gen
        del baseline_next, steered_next
        _release_cuda_memory()


@torch.no_grad()
def feature_activations_for_prompt(
    *,
    model_path: str,
    tokenizer_path: Optional[str] = None,
    sae_path: str,
    feature_id: int,
    prompt_text: str,
    device: str = "auto",
    seed: int = 42,
    match_infer_pipeline: bool = True,
) -> dict[str, Any]:
    """
    Compute per-token activation values of one SAE feature on a custom prompt.
    Primary path uses sae_dashboard's FeatureDataGenerator pipeline to align with
    SaeVis extraction behavior; fallback path uses direct run_with_cache + sae.encode.
    """
    if not prompt_text.strip():
        raise ValueError("prompt_text is empty.")

    model_device, sae_device, device_plan = _resolve_device_pair(device, max_devices=2)
    model = None
    tokenizer = None
    sae = None
    input_tokens = None
    cache = None
    resid = None
    acts = None
    backend_used = "sae_dashboard_feature_data_generator"
    backend_warning: Optional[str] = None

    try:
        # Keep prompt-activation runs reproducible across repeated calls.
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        model, tokenizer = _load_hooked_model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=model_device,
        )
        model.eval()
        sae = _load_sae(sae_path=sae_path, device=sae_device)
        if bool(match_infer_pipeline):
            sae.fold_W_dec_norm()
        hook_name = _get_hook_name_from_sae(sae)

        # Align with SAE Dashboard token handling: no BOS prepend for prompt-centric runs.
        input_tokens = model.to_tokens(prompt_text, prepend_bos=False).to(model.cfg.device)
        if input_tokens.numel() == 0:
            raise ValueError("Prompt produced empty token sequence after tokenization.")

        try:
            from sae_dashboard.sae_vis_data import SaeVisConfig
            from sae_dashboard.sae_vis_runner import FeatureDataGeneratorFactory

            vis_cfg = SaeVisConfig(
                hook_point=hook_name,
                features=[int(feature_id)],
                minibatch_size_features=1,
                minibatch_size_tokens=None,
                device=model_device,
                dtype="bfloat16",
                seed=int(seed),
                verbose=False,
            )
            fdg = FeatureDataGeneratorFactory.create(
                cfg=vis_cfg,
                model=model,
                encoder=sae,
                tokens=input_tokens,
            )
            acts, _, _, _, _, _, _ = fdg.get_feature_data([int(feature_id)], progress=None)
            if acts.ndim != 3 or acts.shape[-1] <= 0:
                raise ValueError(f"Unexpected acts shape from sae_dashboard path: {tuple(acts.shape)}")
            feat_acts_t = acts[0, :, 0].detach().float().cpu()
        except Exception as exc:
            backend_used = "fallback_run_with_cache"
            backend_warning = f"sae_dashboard path failed, fallback used: {type(exc).__name__}: {exc}"
            _, cache = model.run_with_cache(input_tokens, names_filter=lambda name: name == hook_name)
            resid = cache[hook_name]
            resid_for_sae = resid.to(sae_device) if str(sae_device) != str(resid.device) else resid
            acts = sae.encode(resid_for_sae)
            if acts.ndim != 3:
                raise ValueError(f"Unexpected fallback acts shape: {tuple(acts.shape)}")
            if feature_id < 0 or feature_id >= acts.shape[-1]:
                raise ValueError(f"feature_id={feature_id} out of range [0, {acts.shape[-1]-1}]")
            feat_acts_t = acts[0, :, feature_id].detach().float().cpu()

        token_ids = [int(x) for x in input_tokens[0].detach().cpu().tolist()]
        tokens = [
            tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False)
            for tok_id in token_ids
        ]
        feat_acts = [float(x) for x in feat_acts_t.tolist()]

        max_idx = int(torch.argmax(feat_acts_t).item()) if feat_acts else -1
        max_val = float(feat_acts_t[max_idx].item()) if max_idx >= 0 else 0.0

        return {
            "feature_id": int(feature_id),
            "hook_name": hook_name,
            "prompt_text": prompt_text,
            "device": model_device,
            "sae_device": sae_device,
            "device_plan": device_plan,
            "match_infer_pipeline": bool(match_infer_pipeline),
            "backend_used": backend_used,
            "backend_warning": backend_warning,
            "record": {
                "token_ids": token_ids,
                "tokens": tokens,
                "feat_acts": feat_acts,
            },
            "max_activation": max_val,
            "max_activation_token_index": max_idx,
            "max_activation_token": tokens[max_idx] if max_idx >= 0 else "",
        }
    finally:
        del model, tokenizer, sae, input_tokens, cache, resid, acts
        _release_cuda_memory()


# -------------------------
# Optional on-the-fly inference (model + SAE)
# -------------------------


def discover_sae_paths(root: str, max_entries: int = 500) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        has_cfg = "cfg.json" in filenames
        has_sae_weights = any(
            name.endswith(".safetensors") or name.endswith(".pt") for name in filenames
        )
        if has_cfg and has_sae_weights:
            candidates.append(dirpath)
            if len(candidates) >= max_entries:
                break

    candidates.sort()
    return candidates


def _build_dual_cache(dec_projection_onto_wu: torch.Tensor, tokenizer, top_k: int = 10, top_n_features: int = 10):
    top_pos_vals, top_pos_inds = torch.topk(dec_projection_onto_wu, top_k, dim=1)
    top_neg_vals, top_neg_inds = torch.topk(dec_projection_onto_wu, top_k, dim=1, largest=False)

    unique_ids = torch.unique(torch.cat([top_pos_inds, top_neg_inds])).tolist()
    id_to_token = {tid: tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False) for tid in unique_ids}

    feature_to_logits: dict[int, dict[str, list[dict[str, Any]]]] = {}
    pos_inds_cpu = top_pos_inds.cpu().numpy()
    pos_vals_cpu = top_pos_vals.cpu().numpy()
    neg_inds_cpu = top_neg_inds.cpu().numpy()
    neg_vals_cpu = top_neg_vals.cpu().numpy()

    n_features = dec_projection_onto_wu.shape[0]
    for i in range(n_features):
        feature_to_logits[i] = {
            "positive": [
                {"token": id_to_token.get(int(tid), f"<id:{int(tid)}>") , "logit": float(v)}
                for tid, v in zip(pos_inds_cpu[i], pos_vals_cpu[i])
            ],
            "negative": [
                {"token": id_to_token.get(int(tid), f"<id:{int(tid)}>") , "logit": float(v)}
                for tid, v in zip(neg_inds_cpu[i], neg_vals_cpu[i])
            ],
        }

    top_feat_vals, top_feat_inds = torch.topk(dec_projection_onto_wu, top_n_features, dim=0)
    token_to_features = {"indices": top_feat_inds.cpu(), "values": top_feat_vals.cpu()}
    return feature_to_logits, token_to_features


def infer_feature_data_from_model(
    model_path: str,
    tokenizer_path: str,
    sae_path: str,
    dataset_path: str,
    *,
    device: str = "auto",
    top_logits_k: int = 10,
    top_n_features_per_token: int = 10,
    max_features_for_acts: int = 1024,
    n_tokens_for_vis: int = 2048,
    n_tokens_run: int = 256,
    seq_groups_per_feature: int = 5,
    save_logits_pt: bool = False,
    save_acts_pt: bool = False,
    logits_save_path: Optional[str] = None,
    acts_save_path: Optional[str] = None,
) -> FeatureDataBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    from sae_lens import SAE, ActivationsStore
    from sae_dashboard.sae_vis_data import SaeVisConfig
    from sae_dashboard.sae_vis_runner import SaeVisRunner
    from sae_dashboard.utils_fns import get_tokens

    model_device, sae_device, device_plan = _resolve_device_pair(device, max_devices=2)
    resolved_device = model_device
    hf_model = None
    tokenizer = None
    model = None
    sae = None
    w_dec = None
    w_u = None
    dec_proj = None
    activations_store = None
    token_dataset = None
    sae_vis_data = None

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model_name_for_tlens = Path(model_path).name or model_path
        model = HookedTransformer.from_pretrained(
            model_name_for_tlens,
            hf_model=hf_model,
            device=resolved_device,
            dtype=torch.float32,
            center_unembed=False,
            tokenizer=tokenizer,
        )
        model.eval()

        sae = SAE.load_from_disk(sae_path, device=resolved_device)

        # 1) logits cache
        w_dec = sae.W_dec.detach().cpu()
        w_u = _get_unembed_weight_cpu(model, hf_model)
        dec_proj = w_dec @ w_u
        feat_to_logits, token_to_feat = _build_dual_cache(
            dec_proj, tokenizer, top_k=top_logits_k, top_n_features=top_n_features_per_token
        )
        logits_payload = {"feat_to_logits": feat_to_logits, "token_to_feat": token_to_feat}

        # 2) activation records for sequence visualization
        sae.fold_W_dec_norm()
        hook_point = _get_hook_name_from_sae(sae)

        feature_count = min(int(max_features_for_acts), int(sae.W_dec.shape[0]))
        config = SaeVisConfig(
            hook_point=hook_point,
            features=list(range(feature_count)),
            minibatch_size_features=32,
            minibatch_size_tokens=256,
            device=resolved_device,
            dtype="bfloat16",
            seed=42,
        )

        dataset_arg, dataset_info = _prepare_dataset_for_activations_store(dataset_path, streaming=True)

        activations_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=8,
            n_batches_in_buffer=8,
            device="cpu",
            dataset=dataset_arg,
        )
        token_dataset = get_tokens(activations_store, n_tokens_for_vis)
        sae_vis_data = SaeVisRunner(config).run(encoder=sae, model=model, tokens=token_dataset[:n_tokens_run])

        activations: dict[str, list[dict[str, Any]]] = {}
        for feat_idx, feat_data in sae_vis_data.feature_data_dict.items():
            rows: list[dict[str, Any]] = []
            for sg in feat_data.sequence_data.seq_group_data[:seq_groups_per_feature]:
                for seq in sg.seq_data:
                    token_ids = [int(x) for x in seq.token_ids]
                    rows.append(
                        {
                            "token_ids": token_ids,
                            "tokens": [
                                tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False)
                                for tok_id in token_ids
                            ],
                            "feat_acts": [float(x) for x in seq.feat_acts],
                        }
                    )
            activations[str(feat_idx)] = rows

        if save_logits_pt and logits_save_path:
            os.makedirs(Path(logits_save_path).parent, exist_ok=True)
            torch.save(logits_payload, logits_save_path)

        if save_acts_pt and acts_save_path:
            os.makedirs(Path(acts_save_path).parent, exist_ok=True)
            with open(acts_save_path, "w", encoding="utf-8") as f:
                json.dump(activations, f, ensure_ascii=False, indent=2)

        return FeatureDataBundle(
            activations=activations,
            logits_payload=logits_payload,
            feat_to_logits=feat_to_logits,
            token_to_feat=token_to_feat,
            activations_path=acts_save_path,
            logits_path=logits_save_path,
            source="inference",
            metadata={
                "model_path": model_path,
                "tokenizer_path": tokenizer_path,
                "sae_path": sae_path,
                "dataset_path": dataset_path,
                "dataset_info": dataset_info,
                "device": resolved_device,
                "sae_device": sae_device,
                "device_plan": device_plan,
                "top_logits_k": int(top_logits_k),
                "top_n_features_per_token": int(top_n_features_per_token),
                "max_features_for_acts": int(max_features_for_acts),
                "n_tokens_for_vis": int(n_tokens_for_vis),
                "n_tokens_run": int(n_tokens_run),
                "seq_groups_per_feature": int(seq_groups_per_feature),
                "save_logits_pt": bool(save_logits_pt),
                "save_acts_pt": bool(save_acts_pt),
                "logits_save_path": logits_save_path,
                "acts_save_path": acts_save_path,
            },
        )
    finally:
        del hf_model, tokenizer, model, sae, w_dec, w_u, dec_proj, activations_store, token_dataset, sae_vis_data
        _release_cuda_memory()
