from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

DEFAULT_TOKENIZER_PATH = "/mnt/wuyuzhang/models/gemma-2-2b"


@dataclass
class SequenceActivationResult:
    activations: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]
    acts_save_path: Optional[str]


def _resolve_device(device: str) -> str:
    req = str(device or "auto").strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return _to_serializable(obj.__dict__)
    return obj


def _get_sae_hook_name(sae: Any) -> str:
    hook_name = getattr(getattr(sae, "cfg", None), "hook_name", None)
    if hook_name:
        return str(hook_name)
    metadata = getattr(getattr(sae, "cfg", None), "metadata", None)
    if isinstance(metadata, dict) and metadata.get("hook_name"):
        return str(metadata["hook_name"])
    raise ValueError("Cannot find SAE hook_name from sae.cfg")


def _prepare_dataset_arg(dataset_path: str, streaming: bool = True) -> tuple[Any, dict[str, Any]]:
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


def _collect_tokens_from_store(
    activations_store: Any,
    n_prompts: int,
    seed: int,
) -> torch.Tensor:
    """
    Local replacement for sae_dashboard.utils_fns.get_tokens.
    Returns token tensor shape [N, seq].
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    all_tokens_list: list[torch.Tensor] = []
    for _ in range(int(n_prompts)):
        batch_tokens = activations_store.get_batch_tokens().detach().cpu()
        if batch_tokens.ndim != 2:
            raise ValueError(f"Unexpected tokens shape from ActivationsStore: {tuple(batch_tokens.shape)}")
        perm = torch.randperm(batch_tokens.shape[0], generator=generator)
        batch_tokens = batch_tokens[perm]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    perm_all = torch.randperm(all_tokens.shape[0], generator=generator)
    return all_tokens[perm_all]


@torch.no_grad()
def _compute_feature_acts_for_tokens(
    model: Any,
    sae: Any,
    tokens: torch.Tensor,
    hook_name: str,
    feature_count: int,
    minibatch_size_tokens: int,
) -> torch.Tensor:
    """
    Compute SAE feature activations over token batches.
    Returns shape [batch, seq, feature_count] on CPU.
    """
    all_acts_cpu: list[torch.Tensor] = []
    sae_device = str(getattr(sae, "device", getattr(model.cfg, "device", "cpu")))

    for batch in tokens.split(int(minibatch_size_tokens), dim=0):
        batch = batch.to(model.cfg.device)
        _, cache = model.run_with_cache(batch, names_filter=lambda name: name == hook_name)
        resid = cache[hook_name]
        resid_for_sae = resid.to(sae_device) if str(resid.device) != sae_device else resid
        acts = sae.encode(resid_for_sae)
        if acts.ndim != 3:
            raise ValueError(f"Unexpected SAE acts shape: {tuple(acts.shape)}")
        acts = acts[..., : int(feature_count)].detach().float().cpu()
        all_acts_cpu.append(acts)
        del cache, resid, resid_for_sae, acts

    return torch.cat(all_acts_cpu, dim=0)


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False)
        for tok_id in token_ids
    ]


def _extract_top_sequences_for_feature(
    feat_acts_2d: torch.Tensor,  # [batch, seq]
    token_rows: torch.Tensor,  # [batch, seq]
    tokenizer: Any,
    num_sequences: int,
    buffer_left: int,
    buffer_right: int,
) -> list[dict[str, Any]]:
    if feat_acts_2d.ndim != 2:
        return []
    bsz, seqlen = feat_acts_2d.shape
    if bsz == 0 or seqlen == 0:
        return []

    k_candidates = min(feat_acts_2d.numel(), max(int(num_sequences) * 8, int(num_sequences)))
    flat = feat_acts_2d.reshape(-1)
    top_vals, top_idx = torch.topk(flat, k=k_candidates, largest=True)

    seen_windows: set[tuple[int, int, int]] = set()
    rows: list[dict[str, Any]] = []
    for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
        b = int(idx // seqlen)
        s = int(idx % seqlen)
        if b < 0 or b >= bsz:
            continue
        start = max(0, s - int(buffer_left))
        end = min(seqlen, s + int(buffer_right) + 1)
        key = (b, start, end)
        if key in seen_windows:
            continue
        seen_windows.add(key)

        token_ids = [int(x) for x in token_rows[b, start:end].tolist()]
        tokens = _decode_token_ids(tokenizer, token_ids)
        acts = [float(x) for x in feat_acts_2d[b, start:end].tolist()]
        if not tokens or not acts or len(tokens) != len(acts):
            continue
        rows.append(
            {
                "tokens": tokens,
                "feat_acts": acts,
                "max_act_in_window": float(max(acts)),
                "center_pos": int(s),
                "center_act": float(val),
            }
        )
        if len(rows) >= int(num_sequences):
            break

    for row in rows:
        row.pop("max_act_in_window", None)
        row.pop("center_pos", None)
        row.pop("center_act", None)
    return rows


@torch.no_grad()
def extract_sequence_activations_from_dataset(
    model_path: str,
    tokenizer_path: str,
    sae_path: str,
    dataset_path: str,
    *,
    device: str = "auto",
    max_features_for_acts: int = 1024,
    n_tokens_for_vis: int = 2048,
    n_tokens_run: int = 256,
    seq_groups_per_feature: int = 5,
    minibatch_size_features: int = 32,  # kept for compatibility, unused in this local impl
    minibatch_size_tokens: int = 256,
    store_batch_size_prompts: int = 8,
    n_batches_in_buffer: int = 8,
    seed: int = 42,
    save_acts: bool = False,
    acts_save_path: Optional[str] = None,
    buffer_left: int = 5,
    buffer_right: int = 5,
) -> SequenceActivationResult:
    """
    Local non-sae_dashboard implementation:
    dataset -> tokens -> hook activations -> SAE encode -> top sequence windows.
    Output rows: {"tokens": [...], "feat_acts": [...]}
    """
    from sae_lens import SAE, ActivationsStore
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ = minibatch_size_features
    resolved_device = _resolve_device(device)
    _set_seed(int(seed))

    hf_model = None
    tokenizer = None
    model = None
    sae = None
    activations_store = None
    tokens_all = None
    tokens_run = None
    acts_all = None

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
        sae.fold_W_dec_norm()
        hook_name = _get_sae_hook_name(sae)

        dataset_arg, dataset_info = _prepare_dataset_arg(dataset_path, streaming=True)
        activations_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=int(store_batch_size_prompts),
            n_batches_in_buffer=int(n_batches_in_buffer),
            device="cpu",
            dataset=dataset_arg,
        )

        tokens_all = _collect_tokens_from_store(
            activations_store=activations_store,
            n_prompts=int(n_tokens_for_vis),
            seed=int(seed),
        )
        tokens_run = tokens_all[: int(n_tokens_run)].clone()
        if tokens_run.ndim != 2 or tokens_run.shape[0] == 0:
            raise ValueError(f"Invalid tokens for run: shape={tuple(tokens_run.shape)}")

        feature_count = min(int(max_features_for_acts), int(sae.W_dec.shape[0]))
        acts_all = _compute_feature_acts_for_tokens(
            model=model,
            sae=sae,
            tokens=tokens_run,
            hook_name=hook_name,
            feature_count=feature_count,
            minibatch_size_tokens=int(minibatch_size_tokens),
        )

        activations: dict[str, list[dict[str, Any]]] = {}
        for feat_idx in range(feature_count):
            feat_acts_2d = acts_all[:, :, feat_idx]
            rows = _extract_top_sequences_for_feature(
                feat_acts_2d=feat_acts_2d,
                token_rows=tokens_run,
                tokenizer=tokenizer,
                num_sequences=int(seq_groups_per_feature),
                buffer_left=int(buffer_left),
                buffer_right=int(buffer_right),
            )
            activations[str(feat_idx)] = rows

        if bool(save_acts) and acts_save_path:
            out_path = str(acts_save_path)
            os.makedirs(Path(out_path).parent, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(activations, f, ensure_ascii=False, indent=2, default=_to_serializable)
        else:
            out_path = None

        metadata = {
            "implementation": "local_non_sae_dashboard",
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "sae_path": sae_path,
            "dataset_path": dataset_path,
            "dataset_info": dataset_info,
            "device": resolved_device,
            "hook_name": hook_name,
            "max_features_for_acts": int(max_features_for_acts),
            "n_tokens_for_vis": int(n_tokens_for_vis),
            "n_tokens_run": int(n_tokens_run),
            "seq_groups_per_feature": int(seq_groups_per_feature),
            "minibatch_size_tokens": int(minibatch_size_tokens),
            "store_batch_size_prompts": int(store_batch_size_prompts),
            "n_batches_in_buffer": int(n_batches_in_buffer),
            "buffer_left": int(buffer_left),
            "buffer_right": int(buffer_right),
            "seed": int(seed),
            "save_acts": bool(save_acts),
            "acts_save_path": out_path,
        }
        return SequenceActivationResult(
            activations=activations,
            metadata=metadata,
            acts_save_path=out_path,
        )
    finally:
        del hf_model, tokenizer, model, sae, activations_store, tokens_all, tokens_run, acts_all
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


def infer_feature_data_from_model(
    model_path: str,
    tokenizer_path: str,
    sae_path: str,
    dataset_path: str,
    *,
    device: str = "auto",
    top_logits_k: int = 10,  # kept for compatibility, unused in this local file
    top_n_features_per_token: int = 10,  # kept for compatibility, unused in this local file
    max_features_for_acts: int = 1024,
    n_tokens_for_vis: int = 2048,
    n_tokens_run: int = 256,
    seq_groups_per_feature: int = 5,
    save_logits_pt: bool = False,  # kept for compatibility, unused in this local file
    save_acts_pt: bool = False,
    logits_save_path: Optional[str] = None,  # kept for compatibility, unused in this local file
    acts_save_path: Optional[str] = None,
) -> SequenceActivationResult:
    """
    Compatibility wrapper with existing backend signature.
    This local module only returns sequence activations.
    """
    _ = top_logits_k, top_n_features_per_token, save_logits_pt, logits_save_path
    return extract_sequence_activations_from_dataset(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        sae_path=sae_path,
        dataset_path=dataset_path,
        device=device,
        max_features_for_acts=max_features_for_acts,
        n_tokens_for_vis=n_tokens_for_vis,
        n_tokens_run=n_tokens_run,
        seq_groups_per_feature=seq_groups_per_feature,
        save_acts=bool(save_acts_pt),
        acts_save_path=acts_save_path,
    )


if __name__ == "__main__":
    print("Local sequence activation extractor (non-sae_dashboard) is ready.")
    print("Use: extract_sequence_activations_from_dataset(...)")
