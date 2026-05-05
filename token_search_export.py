from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch


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


def load_feature_data(
    activations_path: str,
    logits_path: Optional[str],
) -> FeatureDataBundle:
    activations = load_activations(activations_path)
    logits_payload: dict[str, Any] = {}
    feat_to_logits: dict[Any, Any] = {}
    token_to_feat: Optional[dict[str, Any]] = None

    if logits_path and Path(logits_path).exists():
        logits_payload = load_logits(logits_path)
        maybe_feat_to_logits = logits_payload.get("feat_to_logits", {})
        if isinstance(maybe_feat_to_logits, dict):
            feat_to_logits = maybe_feat_to_logits
        maybe_token_to_feat = logits_payload.get("token_to_feat")
        if isinstance(maybe_token_to_feat, dict):
            token_to_feat = maybe_token_to_feat

    return FeatureDataBundle(
        activations=activations,
        logits_payload=logits_payload,
        feat_to_logits=feat_to_logits,
        token_to_feat=token_to_feat,
        activations_path=activations_path,
        logits_path=logits_path,
        source="load_existing",
        metadata={"activations_path": activations_path, "logits_path": logits_path},
    )


def get_feature_ids(bundle: FeatureDataBundle) -> list[int]:
    act_ids = {int(k) for k in bundle.activations.keys()}
    if bundle.feat_to_logits:
        logit_ids = {int(k) for k in bundle.feat_to_logits.keys()}
        ids = sorted(act_ids & logit_ids)
        if ids:
            return ids
    return sorted(act_ids)


def _safe_feature_dict_get(data: dict[Any, Any], feature_id: int) -> Any:
    if feature_id in data:
        return data[feature_id]
    key = str(feature_id)
    if key in data:
        return data[key]
    return None


def _to_tensor_or_none(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.tensor(x)
    except Exception:
        return None


class ModelInterface:
    """
    模型/分词器读取接口（可替换）。
    当前 token search 仅依赖 tokenizer.encode；model 加载接口预留给后续扩展。
    """

    def load_tokenizer(self, tokenizer_path: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def load_model(self, model_path: str):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, low_cpu_mem_usage=False, device_map=None
        )


class DataInterface:
    """
    数据读取接口（可替换）。
    """

    def load_bundle(self, activations_path: str, logits_path: Optional[str]) -> FeatureDataBundle:
        return load_feature_data(activations_path=activations_path, logits_path=logits_path)


def find_features_for_token_query(
    bundle: FeatureDataBundle,
    query_token: Optional[str],
    query_token_id: Optional[int],
    tokenizer_path: Optional[str],
    n_best: int = 10,
    model_interface: Optional[ModelInterface] = None,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """
    迁移自当前 UI 后端逻辑：
    1) token -> token_id（若用户没直接给 id）
    2) 在 token_to_feat 的该 token 列上取 feature 排名
    3) 过滤到当前 bundle 的特征范围
    """
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
            model_interface = model_interface or ModelInterface()
            tok = model_interface.load_tokenizer(tokenizer_path)
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
        return [], "No matched features in current bundle range."
    return rows, warn


def _feature_top_token_hint(bundle: FeatureDataBundle, feature_id: int) -> str:
    payload = _safe_feature_dict_get(bundle.feat_to_logits, int(feature_id))
    if not isinstance(payload, dict):
        return ""
    pos = payload.get("positive", [])
    if isinstance(pos, list) and pos:
        top = pos[0]
        if isinstance(top, dict):
            return str(top.get("token", ""))
    return ""


def save_token_search_pdf(
    bundle: FeatureDataBundle,
    rows: list[dict[str, Any]],
    query_desc: str,
    save_pdf_path: str,
    *,
    fig_width: float = 12.0,
    base_height: float = 2.2,
    per_bar_height: float = 0.34,
    bar_height: float = 0.44,
    x_padding_ratio: float = 0.24,
    annotation_fontsize: int = 8,
    title_fontsize: int = 12,
) -> str:
    if not rows:
        raise ValueError("rows is empty, nothing to visualize.")

    chart_rows = []
    for r in rows:
        feature_id = int(r.get("feature", -1))
        score = float(r.get("logit", 0.0))
        rank = int(r.get("rank", 0))
        top_token = _feature_top_token_hint(bundle, feature_id) or "(no top token)"
        chart_rows.append(
            {
                "feature": feature_id,
                "score": score,
                "rank": rank,
                "top_token": top_token,
            }
        )

    chart_rows = sorted(chart_rows, key=lambda x: x["rank"])
    labels = [f"Feature {x['feature']}" for x in chart_rows]
    scores = [x["score"] for x in chart_rows]
    ann = [f"{x['score']:.3f} | {x['top_token'][:28]}" for x in chart_rows]

    fig_h = max(base_height, base_height + per_bar_height * len(chart_rows))
    fig, ax = plt.subplots(figsize=(fig_width, fig_h), frameon=False)
    bars = ax.barh(labels, scores, color="#1f77b4", alpha=0.92, height=bar_height)
    ax.invert_yaxis()
    ax.set_title(f"Top {len(rows)} Features Driving Token: {query_desc}", fontsize=title_fontsize, pad=8)
    ax.set_xlabel("Logit Contribution", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    # Add dynamic x padding to avoid clipping for large-magnitude values and annotations.
    x_min = min(scores)
    x_max = max(scores)
    span = max(1e-8, x_max - x_min)
    pad = max(abs(x_max), abs(x_min), span) * float(x_padding_ratio)
    left = x_min - (pad * 0.55)
    right = x_max + pad
    if left == right:
        right = left + 1.0
    ax.set_xlim(left, right)

    for bar, txt in zip(bars, ann):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        if x >= 0:
            tx = x + pad * 0.04
            ha = "left"
        else:
            tx = x - pad * 0.04
            ha = "right"
        ax.text(tx, y, txt, va="center", ha=ha, fontsize=annotation_fontsize, color="#0f2f57")

    fig.tight_layout()
    out = Path(save_pdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format="pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return str(out)


def run_token_search_and_export_pdf(
    *,
    activations_path: str,
    logits_path: str,
    query_token: Optional[str] = None,
    query_token_id: Optional[int] = None,
    tokenizer_path: Optional[str] = None,
    n_best: int = 10,
    save_pdf_path: str = "token_search_result.pdf",
    data_interface: Optional[DataInterface] = None,
    model_interface: Optional[ModelInterface] = None,
) -> dict[str, Any]:
    data_interface = data_interface or DataInterface()
    model_interface = model_interface or ModelInterface()

    bundle = data_interface.load_bundle(activations_path=activations_path, logits_path=logits_path)
    rows, warn = find_features_for_token_query(
        bundle=bundle,
        query_token=query_token,
        query_token_id=query_token_id,
        tokenizer_path=tokenizer_path,
        n_best=int(n_best),
        model_interface=model_interface,
    )
    if not rows:
        return {"ok": False, "warn": warn, "rows": [], "pdf_path": None}

    query_desc = query_token if query_token else f"id={query_token_id}"
    pdf_path = save_token_search_pdf(
        bundle=bundle,
        rows=rows,
        query_desc=str(query_desc),
        save_pdf_path=save_pdf_path,
    )
    return {
        "ok": True,
        "warn": warn,
        "rows": rows,
        "pdf_path": pdf_path,
        "query_desc": query_desc,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Token search (non-UI) and export visualization to PDF.")
    p.add_argument("--activations-path", required=True)
    p.add_argument("--logits-path", required=True)
    p.add_argument("--query-token", default=None)
    p.add_argument("--query-token-id", type=int, default=None)
    p.add_argument("--tokenizer-path", default="/mnt/wuyuzhang/models/gemma-2-2b")
    p.add_argument("--n-best", type=int, default=10)
    p.add_argument("--save-pdf-path", default="token_search_result.pdf")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_token_search_and_export_pdf(
        activations_path=args.activations_path,
        logits_path=args.logits_path,
        query_token=args.query_token,
        query_token_id=args.query_token_id,
        tokenizer_path=args.tokenizer_path,
        n_best=args.n_best,
        save_pdf_path=args.save_pdf_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
