from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import streamlit as st
    import streamlit.components.v1 as st_components
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is required. Install it first: pip install streamlit\n"
        f"Import error: {exc}"
    )

from sae_ui_backend import (
    DEFAULT_TOKENIZER_PATH,
    FeatureDataBundle,
    build_sequence_activation_html,
    discover_sae_paths,
    evaluate_feature_explanation_quality,
    feature_activations_for_prompt,
    find_features_for_token_query,
    generate_feature_explanation,
    get_feature_ids,
    get_feature_payloads,
    get_top_logits_for_feature,
    infer_feature_data_from_model,
    load_feature_data,
    steer_feature_on_text,
)


DEFAULT_ACTS = "/home/liuyiyang/graduate/gemma2_standard_layer19_activations.pt"
DEFAULT_LOGITS = "/home/liuyiyang/graduate/gemma2_standard_layer19_logits.pt"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          .main-title {
            font-size: 56px;
            font-weight: 800;
            letter-spacing: .5px;
            line-height: 1.1;
            margin-bottom: 0;
            text-align: center;
            background: linear-gradient(90deg,#1452cc,#00a6a6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
          .sub-title {
            color: #5f6b7a;
            text-align: center;
            margin-top: .25rem;
            margin-bottom: 1.2rem;
          }
          .function-menu-shell {
            border: 1px solid rgba(20,82,204,.18);
            border-radius: 12px;
            padding: 12px 14px 14px 14px;
            background: linear-gradient(180deg, rgba(20,82,204,.04), rgba(0,166,166,.04));
            margin-bottom: 1rem;
          }
          .function-menu-title {
            font-weight: 700;
            margin-bottom: .5rem;
            color: #113a8f;
          }
          .sidebar-mode-shell {
            border: 1px solid rgba(20,82,204,.18);
            border-radius: 12px;
            padding: 10px 10px 12px 10px;
            background: linear-gradient(180deg, rgba(20,82,204,.05), rgba(0,166,166,.03));
            margin-bottom: .8rem;
          }
          .sidebar-mode-title {
            font-weight: 700;
            margin-bottom: .4rem;
            color: #113a8f;
            font-size: 13px;
          }
          .cache-stat-shell {
            border: 1px solid rgba(20,82,204,.14);
            border-radius: 12px;
            padding: 10px 10px 8px 10px;
            background: rgba(255,255,255,.76);
            margin-bottom: .8rem;
          }
          .cache-row {
            margin-bottom: 8px;
          }
          .cache-row:last-child {
            margin-bottom: 0;
          }
          .cache-head {
            display:flex;
            justify-content:space-between;
            align-items:center;
            font-size:12px;
            color:#284063;
            margin-bottom:4px;
          }
          .cache-head .active {
            color:#0c2f7a;
            font-weight:700;
          }
          .cache-bar {
            width:100%;
            height:8px;
            border-radius:999px;
            background:rgba(17,58,143,.12);
            overflow:hidden;
          }
          .cache-fill {
            height:100%;
            border-radius:999px;
            background:linear-gradient(90deg,#1452cc,#00a6a6);
          }
          .metric-card {
            border: 1px solid rgba(20,82,204,.2);
            border-radius: 12px;
            padding: 10px 12px;
            background: linear-gradient(180deg, rgba(255,255,255,.7), rgba(20,82,204,.04));
            margin-bottom: .8rem;
          }
          .metric-card .k {
            color:#5f6b7a;
            font-size:12px;
          }
          .metric-card .v {
            color:#0c2f7a;
            font-size:22px;
            font-weight:800;
            line-height:1.2;
            word-break: break-word;
          }
          .panel-shell {
            border: 1px solid rgba(20,82,204,.18);
            border-radius: 12px;
            padding: 12px;
            background: rgba(255,255,255,.75);
            margin-bottom: .8rem;
          }
          .explain-chip {
            display:inline-block;
            padding:4px 8px;
            margin:3px 4px 3px 0;
            border-radius:999px;
            background: rgba(20,82,204,.10);
            color:#0f3f97;
            font-size:12px;
            font-weight:600;
          }
          .explain-card {
            border: 1px solid rgba(20,82,204,.18);
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 8px;
            background: linear-gradient(180deg, rgba(20,82,204,.04), rgba(0,166,166,.03));
          }
          .explain-method {
            color:#0f3f97;
            font-size:12px;
            font-weight:700;
            margin-bottom:4px;
          }
          .explain-text {
            color:#1f2d3d;
            font-size:13px;
            line-height:1.45;
          }
          .kv-shell {
            border: 1px solid rgba(20,82,204,.18);
            border-radius: 12px;
            background: rgba(255,255,255,.76);
            overflow: hidden;
            margin-bottom: .8rem;
          }
          .kv-row {
            display:flex;
            justify-content:space-between;
            gap:10px;
            padding:9px 12px;
            border-bottom:1px solid rgba(20,82,204,.10);
          }
          .kv-row:last-child {
            border-bottom:none;
          }
          .kv-k {
            color:#5f6b7a;
            font-size:12px;
          }
          .kv-v {
            color:#0f3f97;
            font-size:12px;
            font-weight:700;
            text-align:right;
            word-break:break-all;
          }
          .status-chip {
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            font-size:11px;
            font-weight:700;
          }
          .status-ok {
            color:#086f3a;
            background:rgba(12,170,93,.16);
          }
          .status-no {
            color:#8c1f1f;
            background:rgba(220,58,58,.14);
          }
          div[data-testid="stButton"] > button {
            border-radius: 12px;
            font-weight: 700;
            min-height: 54px;
          }
          div[data-testid="stRadio"] > label p,
          div[data-testid="stSelectbox"] label p {
            font-weight: 700 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults = {
        "bundle_load": None,
        "bundle_infer": None,
        "show_load_setup": True,
        "show_infer_setup": True,
        "explain_history": [],
        "sae_candidates": [],
        "explain_cache_load": {},
        "explain_cache_infer": {},
        "menu_load": "Feature Explorer",
        "menu_infer": "Feature Explorer",
        "token_search_rows_load": [],
        "token_search_rows_infer": [],
        "token_search_meta_load": {},
        "token_search_meta_infer": {},
        "explain_result_cache_load": {},
        "explain_result_cache_infer": {},
        "prompt_act_cache_load": {},
        "prompt_act_cache_infer": {},
        "api_model": "openai/gpt-oss-120b:free",
        "api_base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "api_provider": "auto",
        "api_temperature": 0.0,
        "api_max_tokens": 128,
        "api_timeout": 120,
        "api_tokens_around": 24,
        "api_top_logits_k": 10,
        "api_max_records_per_feature": 25,
        "api_save_prompts": False,
        "api_or_name": "SAE Interpreter",
        "api_or_url": "",
        "runtime_sae_path": "",
        "runtime_device": "auto",
        "source_mode": "Load existing data",
        "eval_result_cache_load": {},
        "eval_result_cache_infer": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _esc_html(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _materialize_uploaded_file(uploaded_file, suffix: str) -> str:
    tmp_dir = Path("/home/liuyiyang/graduate/.uploaded_cache")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    out.write_bytes(uploaded_file.getbuffer())
    return str(out)


def _sync_picker_to_manual(sel_key: str, manual_key: str) -> None:
    sel = str(st.session_state.get(sel_key, "") or "").strip()
    if sel:
        st.session_state[manual_key] = sel


def _server_file_picker(
    *,
    label: str,
    key: str,
    default_dir: str,
    patterns: tuple[str, ...],
) -> str:
    st.markdown(f"**{label}**")
    pick_dir = st.text_input("Directory", value=default_dir, key=f"{key}_dir")

    files: list[str] = []
    p = Path(pick_dir).expanduser()
    if p.exists() and p.is_dir():
        for pattern in patterns:
            files.extend([str(x) for x in p.glob(pattern) if x.is_file()])
    files = sorted(set(files))[:1000]

    sel_key = f"{key}_sel"
    manual_key = f"{key}_manual"
    if manual_key not in st.session_state:
        st.session_state[manual_key] = ""

    selected = st.selectbox(
        "Pick file",
        options=[""] + files,
        key=sel_key,
        on_change=_sync_picker_to_manual,
        args=(sel_key, manual_key),
    )
    manual = st.text_input("Or full path", key=manual_key)
    return (manual.strip() or str(selected).strip())


def _server_dir_picker(*, label: str, key: str, default_dir: str) -> str:
    st.markdown(f"**{label}**")
    pick_dir = st.text_input("Root directory", value=default_dir, key=f"{key}_dir")

    dirs: list[str] = []
    p = Path(pick_dir).expanduser()
    if p.exists() and p.is_dir():
        dirs = sorted([str(x) for x in p.iterdir() if x.is_dir()])[:1000]

    sel_key = f"{key}_sel"
    manual_key = f"{key}_manual"
    if manual_key not in st.session_state:
        st.session_state[manual_key] = ""

    selected = st.selectbox(
        "Pick directory",
        options=[""] + dirs,
        key=sel_key,
        on_change=_sync_picker_to_manual,
        args=(sel_key, manual_key),
    )
    manual = st.text_input("Or full directory", key=manual_key)
    return (manual.strip() or str(selected).strip())


def _set_bundle_for_mode(mode_key: str, bundle: FeatureDataBundle) -> None:
    if mode_key == "load":
        st.session_state.bundle_load = bundle
        st.session_state.show_load_setup = False
    else:
        st.session_state.bundle_infer = bundle
        st.session_state.show_infer_setup = False


def _load_pt_data_to_mode(acts_path: str, logits_path: str, mode_key: str = "load") -> None:
    with st.spinner("Loading data..."):
        bundle = load_feature_data(acts_path, logits_path)
    _set_bundle_for_mode(mode_key, bundle)


def _infer_top_k_from_bundle(bundle: FeatureDataBundle) -> int:
    if bundle.metadata and isinstance(bundle.metadata.get("top_logits_k"), int):
        return int(bundle.metadata["top_logits_k"])
    if bundle.metadata and isinstance(bundle.metadata.get("inferred_top_logits_k"), int):
        return int(bundle.metadata["inferred_top_logits_k"])
    # fallback from payload
    if bundle.feat_to_logits:
        try:
            first = next(iter(bundle.feat_to_logits.keys()))
            row = bundle.feat_to_logits[first]
            if isinstance(row, dict) and isinstance(row.get("positive"), list):
                return len(row["positive"])
        except Exception:
            pass
    return 10


def _cache_names(mode_key: str) -> tuple[str, str]:
    if mode_key == "load":
        return "explain_cache_load", "explain_result_cache_load"
    return "explain_cache_infer", "explain_result_cache_infer"


def _eval_cache_name(mode_key: str) -> str:
    if mode_key == "load":
        return "eval_result_cache_load"
    return "eval_result_cache_infer"


def _ensure_api_defaults() -> None:
    if not str(st.session_state.get("api_model", "")).strip():
        st.session_state["api_model"] = "openai/gpt-oss-120b:free"
    if not str(st.session_state.get("api_base_url", "")).strip():
        st.session_state["api_base_url"] = "https://openrouter.ai/api/v1"
    if not str(st.session_state.get("api_or_name", "")).strip():
        st.session_state["api_or_name"] = "SAE Interpreter"


def _init_api_widget_state() -> None:
    _ensure_api_defaults()
    pairs = [
        ("api_model_ui", "api_model"),
        ("api_base_url_ui", "api_base_url"),
        ("api_key_ui", "api_key"),
        ("api_provider_ui", "api_provider"),
        ("api_temperature_ui", "api_temperature"),
        ("api_max_tokens_ui", "api_max_tokens"),
        ("api_timeout_ui", "api_timeout"),
        ("api_tokens_around_ui", "api_tokens_around"),
        ("api_top_logits_k_ui", "api_top_logits_k"),
        ("api_max_records_per_feature_ui", "api_max_records_per_feature"),
        ("api_save_prompts_ui", "api_save_prompts"),
        ("api_or_name_ui", "api_or_name"),
        ("api_or_url_ui", "api_or_url"),
    ]
    for ui_key, store_key in pairs:
        if ui_key not in st.session_state:
            st.session_state[ui_key] = st.session_state.get(store_key)
            continue
        ui_val = st.session_state.get(ui_key)
        store_val = st.session_state.get(store_key)
        if isinstance(ui_val, str) and isinstance(store_val, str):
            if (not ui_val.strip()) and store_val.strip():
                st.session_state[ui_key] = store_val


def _sync_api_widget_to_store() -> None:
    pairs = [
        ("api_model_ui", "api_model"),
        ("api_base_url_ui", "api_base_url"),
        ("api_key_ui", "api_key"),
        ("api_provider_ui", "api_provider"),
        ("api_temperature_ui", "api_temperature"),
        ("api_max_tokens_ui", "api_max_tokens"),
        ("api_timeout_ui", "api_timeout"),
        ("api_tokens_around_ui", "api_tokens_around"),
        ("api_top_logits_k_ui", "api_top_logits_k"),
        ("api_max_records_per_feature_ui", "api_max_records_per_feature"),
        ("api_save_prompts_ui", "api_save_prompts"),
        ("api_or_name_ui", "api_or_name"),
        ("api_or_url_ui", "api_or_url"),
    ]
    for ui_key, store_key in pairs:
        if ui_key in st.session_state:
            st.session_state[store_key] = st.session_state[ui_key]


def _ensure_runtime_defaults() -> None:
    if "runtime_sae_path" not in st.session_state:
        st.session_state["runtime_sae_path"] = ""
    if not str(st.session_state.get("runtime_device", "")).strip():
        st.session_state["runtime_device"] = "auto"


def _init_runtime_widget_state(default_sae_path: Optional[str] = None, default_device: str = "auto") -> None:
    _ensure_runtime_defaults()
    if "runtime_sae_path_ui" not in st.session_state:
        st.session_state["runtime_sae_path_ui"] = st.session_state.get("runtime_sae_path", "")
    if not str(st.session_state.get("runtime_sae_path_ui", "")).strip():
        if str(st.session_state.get("runtime_sae_path", "")).strip():
            st.session_state["runtime_sae_path_ui"] = st.session_state["runtime_sae_path"]
        elif default_sae_path and str(default_sae_path).strip():
            st.session_state["runtime_sae_path_ui"] = str(default_sae_path).strip()

    if "runtime_device_ui" not in st.session_state:
        st.session_state["runtime_device_ui"] = st.session_state.get("runtime_device", default_device)
    if not str(st.session_state.get("runtime_device_ui", "")).strip():
        st.session_state["runtime_device_ui"] = st.session_state.get("runtime_device", default_device)


def _sync_runtime_widget_to_store() -> None:
    if "runtime_sae_path_ui" in st.session_state:
        st.session_state["runtime_sae_path"] = str(st.session_state["runtime_sae_path_ui"] or "").strip()
    if "runtime_device_ui" in st.session_state:
        st.session_state["runtime_device"] = str(st.session_state["runtime_device_ui"] or "auto").strip() or "auto"


def _cached_explanations_for_feature(mode_key: str, feature_id: int) -> dict[str, str]:
    txt_cache_name, _ = _cache_names(mode_key)
    cache = st.session_state.get(txt_cache_name, {})
    return {
        method: txt
        for (fid, method), txt in cache.items()
        if int(fid) == int(feature_id)
    }


def _render_top_cards(mode_key: str, bundle: FeatureDataBundle, selected_feature: int, num_features: int) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Selected Feature</div><div class='v'>{selected_feature}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Current Mode</div><div class='v'>{mode_key.upper()}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Features In Bundle</div><div class='v'>{num_features}</div></div>",
            unsafe_allow_html=True,
    )


def _get_device_options() -> list[str]:
    options: list[str] = ["auto", "cpu"]
    gpu_indices: list[int] = []

    # Prefer nvidia-smi for lightweight detection in UI process.
    try:
        cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=2)
        for line in (out.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                gpu_indices.append(int(line))
            except Exception:
                continue
    except Exception:
        # Fallback to torch probing.
        try:
            import torch

            if torch.cuda.is_available():
                gpu_indices = list(range(int(torch.cuda.device_count())))
        except Exception:
            gpu_indices = []

    for idx in sorted(set(gpu_indices)):
        options.append(f"cuda:{idx}")
    return options


def _device_selectbox(label: str, key: str, default: str = "auto") -> str:
    options = _get_device_options()
    current = str(st.session_state.get(key, default) or default)
    if current not in options:
        options.append(current)
    index = options.index(current if current in options else default)
    return str(st.selectbox(label, options=options, index=index, key=key))


def _render_sidebar_mode_switch() -> str:
    options = ["Load existing data", "Infer now"]
    selected = st.session_state.get("source_mode", options[0])
    if selected not in options:
        selected = options[0]
        st.session_state["source_mode"] = selected

    st.markdown("<div class='sidebar-mode-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-mode-title'>Choose Mode</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Load existing data",
            key="sidebar_mode_load",
            use_container_width=True,
            type="primary" if selected == options[0] else "secondary",
        ):
            if selected != options[0]:
                st.session_state["source_mode"] = options[0]
                st.rerun()
    with c2:
        if st.button(
            "Infer now",
            key="sidebar_mode_infer",
            use_container_width=True,
            type="primary" if selected == options[1] else "secondary",
        ):
            if selected != options[1]:
                st.session_state["source_mode"] = options[1]
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.get("source_mode", options[0])


def _render_mode_cache_visual(load_count: int, infer_count: int, source_mode: str) -> None:
    total = max(1, int(load_count) + int(infer_count))
    load_pct = max(0, min(100, int(round(100.0 * int(load_count) / total))))
    infer_pct = max(0, min(100, int(round(100.0 * int(infer_count) / total))))

    load_active = "active" if source_mode == "Load existing data" else ""
    infer_active = "active" if source_mode == "Infer now" else ""

    st.markdown(
        (
            "<div class='cache-stat-shell'>"
            "<div class='sidebar-mode-title'>Cached Data Status</div>"
            f"<div class='cache-row'><div class='cache-head'><span class='{load_active}'>Load</span>"
            f"<span>{load_count} features</span></div>"
            f"<div class='cache-bar'><div class='cache-fill' style='width:{load_pct}%;'></div></div></div>"
            f"<div class='cache-row'><div class='cache-head'><span class='{infer_active}'>Infer</span>"
            f"<span>{infer_count} features</span></div>"
            f"<div class='cache-bar'><div class='cache-fill' style='width:{infer_pct}%;'></div></div></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_load_existing_ui() -> None:
    if st.session_state.bundle_load is not None and not st.session_state.show_load_setup:
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("Re-select Files"):
                st.session_state.show_load_setup = True
                st.rerun()
        with c2:
            st.success("Load mode data is ready. Dashboard is shown below.")
        return

    st.markdown("### Load Existing Data")
    mode = st.radio(
        "Load mode",
        options=["Upload from local computer", "Browse server files"],
        horizontal=True,
        key="load_mode_radio",
    )

    if mode == "Upload from local computer":
        st.caption("Browser popup upload (best for local running).")
        up_acts = st.file_uploader("Activations file (.pt/.json)", type=["pt", "json"], key="up_acts")
        up_logits = st.file_uploader("Logits file (.pt) - optional", type=["pt"], key="up_logits")

        if st.button("Load Uploaded Files"):
            if up_acts is None:
                st.error("Please upload activations first.")
                return
            acts_path = _materialize_uploaded_file(up_acts, "_acts.pt")
            logits_path = ""
            if up_logits is not None:
                logits_path = _materialize_uploaded_file(up_logits, "_logits.pt")
            try:
                _load_pt_data_to_mode(acts_path, logits_path, mode_key="load")
                st.success("Loaded uploaded files.")
                st.rerun()
            except Exception as exc:
                st.error(f"Load failed: {exc}")
    else:
        st.caption("Server-side browse (best for remote server deployment).")
        c1, c2 = st.columns(2)
        with c1:
            acts_path = _server_file_picker(
                label="Activations file",
                key="browse_acts",
                default_dir="/home/liuyiyang/graduate",
                patterns=("*.pt", "*.json"),
            )
        with c2:
            logits_path = _server_file_picker(
                label="Logits file (optional)",
                key="browse_logits",
                default_dir="/home/liuyiyang/graduate",
                patterns=("*logits*.pt", "*.pt"),
            )

        if st.button("Load Server Files"):
            if not acts_path:
                st.error("Please select activations file.")
                return
            try:
                _load_pt_data_to_mode(acts_path, logits_path, mode_key="load")
                st.success("Loaded server files.")
                st.rerun()
            except Exception as exc:
                st.error(f"Load failed: {exc}")


def _render_infer_data_ui() -> None:
    if st.session_state.bundle_infer is not None and not st.session_state.show_infer_setup:
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("Re-run Inference Setup"):
                st.session_state.show_infer_setup = True
                st.rerun()
        with c2:
            st.success("Infer mode data is ready. Dashboard is shown below.")
        return

    st.markdown("### Infer Now (Model + SAE)")
    _init_runtime_widget_state(default_sae_path=None, default_device="auto")

    ca, cb = st.columns(2)
    with ca:
        model_path = st.text_input("Model path/name", value="/mnt/wuyuzhang/models/gemma-2-2b")
        tokenizer_path = st.text_input("Tokenizer path/name", value="/mnt/wuyuzhang/models/gemma-2-2b")
    with cb:
        dataset_path = st.text_input("Dataset path", value="/mnt/wuyuzhang/datasets/openwebtext_tokenized_gemma-2-9b/data")
        device = _device_selectbox(
            "Device",
            key="runtime_device_ui",
            default=str(st.session_state.get("runtime_device", "auto")),
        )

    with st.expander("Browse server paths (optional)"):
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            model_pick = _server_dir_picker(label="Model directory", key="pick_model", default_dir=".")
            if st.button("Use model dir") and model_pick:
                st.session_state["_fill_model"] = model_pick
        with cp2:
            tok_pick = _server_dir_picker(label="Tokenizer directory", key="pick_tok", default_dir=".")
            if st.button("Use tokenizer dir") and tok_pick:
                st.session_state["_fill_tok"] = tok_pick
        with cp3:
            data_pick = _server_dir_picker(label="Dataset directory", key="pick_data", default_dir=".")
            if st.button("Use dataset dir") and data_pick:
                st.session_state["_fill_data"] = data_pick

    if st.session_state.get("_fill_model"):
        model_path = st.session_state["_fill_model"]
    if st.session_state.get("_fill_tok"):
        tokenizer_path = st.session_state["_fill_tok"]
    if st.session_state.get("_fill_data"):
        dataset_path = st.session_state["_fill_data"]

    st.markdown("#### SAE Selection")
    sae_root = st.text_input("SAE root (scan)", value="")
    cs1, cs2 = st.columns([1, 1])
    with cs1:
        if st.button("Scan SAE"):
            if not sae_root:
                st.warning("Please set SAE root first.")
            else:
                with st.spinner("Scanning SAE directories..."):
                    st.session_state.sae_candidates = discover_sae_paths(sae_root)
    with cs2:
        if st.button("Clear SAE list"):
            st.session_state.sae_candidates = []

    sae_path = st.text_input(
        "SAE path",
        key="runtime_sae_path_ui",
        value=str(st.session_state.get("runtime_sae_path", "")),
    )
    if st.session_state.sae_candidates:
        picked = st.selectbox("Discovered SAE", options=[""] + st.session_state.sae_candidates)
        if picked:
            st.session_state["runtime_sae_path_ui"] = picked
            sae_path = picked

    _sync_runtime_widget_to_store()

    st.markdown("#### Inference Controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        top_logits_k = st.number_input("TopLogitsK", min_value=1, max_value=200, value=10)
        top_n_features_per_token = st.number_input("TopFeat/Token", min_value=1, max_value=200, value=10)
    with c2:
        max_features_for_acts = st.number_input("MaxFeatActs", min_value=1, max_value=65536, value=1024)
        n_tokens_for_vis = st.number_input("TokenBuffer", min_value=128, max_value=100000, value=2048)
    with c3:
        n_tokens_run = st.number_input("TokensRun", min_value=32, max_value=4096, value=256)
        seq_groups_per_feature = st.number_input("SeqGroups", min_value=1, max_value=20, value=5)
    with c4:
        st.caption("Outputs")
        save_logits_pt = st.checkbox("Save logits", value=False)
        save_acts_pt = st.checkbox("Save acts", value=False)

    logits_save_path = None
    acts_save_path = None
    if save_logits_pt:
        logits_save_path = st.text_input("Logits save path", value="/home/liuyiyang/graduate/inferred_logits.pt")
    if save_acts_pt:
        acts_save_path = st.text_input("Acts save path", value="/home/liuyiyang/graduate/inferred_activations.pt")

    if st.button("Run Inference"):
        if not sae_path or not dataset_path:
            st.error("SAE path and dataset path are required.")
            return
        try:
            with st.spinner("Running inference (can be slow)..."):
                bundle = infer_feature_data_from_model(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    sae_path=sae_path,
                    dataset_path=dataset_path,
                    device=device,
                    top_logits_k=int(top_logits_k),
                    top_n_features_per_token=int(top_n_features_per_token),
                    max_features_for_acts=int(max_features_for_acts),
                    n_tokens_for_vis=int(n_tokens_for_vis),
                    n_tokens_run=int(n_tokens_run),
                    seq_groups_per_feature=int(seq_groups_per_feature),
                    save_logits_pt=save_logits_pt,
                    save_acts_pt=save_acts_pt,
                    logits_save_path=logits_save_path,
                    acts_save_path=acts_save_path,
                )
            _set_bundle_for_mode("infer", bundle)
            st.success("Inference completed and data loaded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Inference failed: {exc}")


def _plot_top_logits(pos: list[dict], neg: list[dict], title: str) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Positive Logits", "Negative Logits"),
        horizontal_spacing=0.12,
    )

    pos_tokens = [str(x.get("token", "")) for x in pos]
    pos_vals = [float(x.get("logit", 0.0)) for x in pos]
    pos_y = list(range(len(pos_tokens)))
    if pos_tokens:
        fig.add_trace(
            go.Bar(
                x=pos_vals,
                y=pos_y,
                orientation="h",
                marker_color="#1f77b4",
                customdata=pos_tokens,
                hovertemplate="token=%{customdata}<br>logit=%{x:.4f}<extra>positive</extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=pos_y,
            ticktext=pos_tokens,
            autorange="reversed",
            row=1,
            col=1,
        )

    neg_tokens = [str(x.get("token", "")) for x in neg]
    neg_vals = [float(x.get("logit", 0.0)) for x in neg]
    neg_y = list(range(len(neg_tokens)))
    if neg_tokens:
        fig.add_trace(
            go.Bar(
                x=neg_vals,
                y=neg_y,
                orientation="h",
                marker_color="#d62728",
                customdata=neg_tokens,
                hovertemplate="token=%{customdata}<br>logit=%{x:.4f}<extra>negative</extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=neg_y,
            ticktext=neg_tokens,
            autorange="reversed",
            row=1,
            col=2,
        )

    fig.update_layout(
        title=title,
        hovermode="closest",
        height=380,
        margin=dict(l=20, r=20, t=44, b=20),
    )
    fig.update_xaxes(title_text="Logit", row=1, col=1)
    fig.update_xaxes(title_text="Logit", row=1, col=2)
    return fig


def _feature_top_token_hint(bundle: FeatureDataBundle, feature_id: int) -> str:
    pos, _ = get_top_logits_for_feature(bundle, int(feature_id), top_k=1)
    if pos:
        return str(pos[0].get("token", ""))
    return ""


def _render_sequence_html(records: list[dict], max_records: int) -> None:
    html = build_sequence_activation_html(records, max_records=max_records)
    n = min(len(records), max_records)
    estimated_lines = 0
    for rec in records[:max_records]:
        token_count = len(rec.get("tokens", [])) if isinstance(rec, dict) else 0
        estimated_lines += max(1, (int(token_count) + 11) // 12)
    height = int(min(1200, max(180, 70 + n * 26 + estimated_lines * 22)))
    try:
        st_components.html(html, height=height, scrolling=True)
    except Exception:
        st.markdown(html, unsafe_allow_html=True)


def _plot_token_search_attribution(bundle: FeatureDataBundle, rows: list[dict], query_desc: str) -> go.Figure:
    chart_rows = []
    for r in rows:
        feature_id = int(r.get("feature", -1))
        score = float(r.get("logit", 0.0))
        rank = int(r.get("rank", 0))
        top_token = _feature_top_token_hint(bundle, feature_id)
        chart_rows.append(
            {
                "feature": feature_id,
                "score": score,
                "rank": rank,
                "top_token": top_token or "(no top token)",
            }
        )

    chart_rows = sorted(chart_rows, key=lambda x: x["rank"], reverse=True)
    y_labels = [f"Feature {x['feature']}" for x in chart_rows]
    x_vals = [x["score"] for x in chart_rows]
    hover_tokens = [x["top_token"] for x in chart_rows]
    rank_labels = [f"rank {x['rank']}" for x in chart_rows]
    text_labels = [f"{x['score']:.3f} | {x['top_token'][:28]}" for x in chart_rows]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_vals,
                y=y_labels,
                orientation="h",
                marker_color="#1f77b4",
                customdata=list(zip(rank_labels, hover_tokens)),
                hovertemplate=(
                    "%{y}<br>Contribution=%{x:.4f}<br>%{customdata[0]}"
                    "<br>Top positive token=%{customdata[1]}<extra></extra>"
                ),
                text=text_labels,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=f"Top {len(rows)} Features Driving Token: {query_desc}",
        height=max(360, 120 + 36 * len(rows)),
        margin=dict(l=20, r=20, t=56, b=20),
        xaxis_title="Logit Contribution",
        yaxis_title="Feature",
    )
    return fig


def _render_feature_explorer(bundle: FeatureDataBundle, feature_id: int, mode_key: str) -> None:
    st.subheader(f"Feature {feature_id}")

    left, right = st.columns([2.3, 1.2], gap="medium")
    with left:
        pos, neg = get_top_logits_for_feature(bundle, feature_id, top_k=None)
        k = max(len(pos), len(neg))
        if pos or neg:
            st.plotly_chart(
                _plot_top_logits(pos, neg, title=f"Top-{k} Positive / Negative Logits"),
                use_container_width=True,
            )
        else:
            st.info("No logits payload available for this feature.")

        st.markdown("#### Prompt Activation (Custom Input)")
        _render_prompt_activation_panel(bundle, feature_id, mode_key=mode_key, compact=True)

        st.markdown("#### Sequence Token Activations")
        max_records = st.slider(
            "Max sequences",
            min_value=1,
            max_value=12,
            value=6,
            key=f"seq_slider_{mode_key}_{feature_id}",
        )
        act_payload, _ = get_feature_payloads(bundle, feature_id)
        if act_payload:
            _render_sequence_html(act_payload, max_records=max_records)
        else:
            st.info("No activation sequences available.")

    with right:
        st.markdown("<div class='panel-shell'>", unsafe_allow_html=True)
        st.markdown("**Cached Explanations**")
        cached = _cached_explanations_for_feature(mode_key, feature_id)
        if cached:
            for method, txt in cached.items():
                txt_html = (
                    str(txt)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                st.markdown(
                    "<div class='explain-card'>"
                    f"<div class='explain-method'>{method}</div>"
                    f"<div class='explain-text'>{txt_html}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No saved explanation for this feature yet.")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_429_guidance() -> None:
    st.warning("429 Too Many Requests: current request rate or quota exceeded.")
    st.info(
        "Fix options:\n"
        "1. Reduce concurrency / request frequency and retry after a few seconds.\n"
        "2. Use a cheaper model, or lower `max_tokens`.\n"
        "3. Check API account balance, billing, and rate limits.\n"
        "4. Temporarily switch to `heuristic` mode for debugging."
    )


def _render_explanation_panel(bundle: FeatureDataBundle, feature_id: int, mode_key: str) -> None:
    st.subheader("Feature Explanation")
    _init_api_widget_state()

    method = st.selectbox(
        "Method",
        options=["np_max-act-logits", "token-activation-pair", "token-space-representation"],
        key=f"explain_method_{mode_key}",
    )
    mode = st.selectbox("Explanation mode", options=["auto", "heuristic", "api"], index=0, key=f"explain_mode_{mode_key}")

    col1, col2 = st.columns(2)
    with col1:
        model = st.text_input("LLM model", key="api_model_ui")
        api_base_url = st.text_input("API base URL", key="api_base_url_ui")
    with col2:
        api_key = st.text_input("API key", type="password", key="api_key_ui", help="Kept in current session only.")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, key="api_temperature_ui")

    col_provider_1, col_provider_2 = st.columns(2)
    with col_provider_1:
        api_provider = st.selectbox(
            "API provider",
            options=["auto", "openai", "openrouter", "custom"],
            key="api_provider_ui",
            help="auto: infer provider from base URL or model (e.g., :free -> openrouter).",
        )
    with col_provider_2:
        st.caption(
            "OpenRouter free model example: `google/gemma-4-26b-a4b-it:free` "
            "with base URL `https://openrouter.ai/api/v1`."
        )

    with st.expander("OpenRouter Optional Headers (Leaderboard attribution)"):
        openrouter_app_name = st.text_input(
            "X-Title",
            key="api_or_name_ui",
            help="Optional app title shown in OpenRouter attribution.",
        )
        openrouter_app_url = st.text_input(
            "HTTP-Referer",
            key="api_or_url_ui",
            help="Optional public app URL for OpenRouter attribution.",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        max_tokens = st.number_input("Max tokens", min_value=8, max_value=1024, key="api_max_tokens_ui")
    with col4:
        timeout = st.number_input("Timeout (s)", min_value=10, max_value=600, key="api_timeout_ui")
    with col5:
        tokens_around = st.number_input("Tokens around max act", min_value=2, max_value=64, key="api_tokens_around_ui")

    top_logits_k = st.number_input("Top logits K (prompt)", min_value=1, max_value=50, key="api_top_logits_k_ui")
    max_records_per_feature = st.number_input("Max records/feature", min_value=1, max_value=64, key="api_max_records_per_feature_ui")
    save_prompts = st.checkbox("Show prompt messages", key="api_save_prompts_ui")
    _sync_api_widget_to_store()

    _, result_cache_name = _cache_names(mode_key)
    result_cache = st.session_state.get(result_cache_name, {})
    cached_result = result_cache.get((int(feature_id), method))
    if cached_result:
        st.markdown("#### Cached Result For This Feature")
        st.success(f"Explanation: {cached_result.get('explanation', '')}")
        with st.expander("Cached raw payload"):
            st.json(cached_result)

    if st.button("Generate Explanation"):
        try:
            result = generate_feature_explanation(
                bundle=bundle,
                feature_id=int(feature_id),
                method=method,
                mode=mode,
                model=model,
                api_base_url=api_base_url,
                api_key=api_key if api_key else None,
                api_provider=api_provider,
                openrouter_app_name=openrouter_app_name.strip() or None,
                openrouter_app_url=openrouter_app_url.strip() or None,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                timeout=float(timeout),
                tokens_around=int(tokens_around),
                top_logits_k=int(top_logits_k),
                max_records_per_feature=int(max_records_per_feature),
                save_prompts=save_prompts,
            )
            st.session_state.explain_history.insert(0, result)

            cache_name, result_cache_name = _cache_names(mode_key)
            st.session_state[cache_name][(int(feature_id), method)] = result["explanation"]
            st.session_state[result_cache_name][(int(feature_id), method)] = result

            st.success("Explanation generated.")
            st.markdown(f"**Explanation:** `{result['explanation']}`")
            st.json(result)
        except Exception as exc:
            err_text = str(exc)
            st.error(f"Explanation failed: {err_text}")
            if "429" in err_text:
                _render_429_guidance()

    if st.session_state.explain_history:
        with st.expander("History"):
            st.json(st.session_state.explain_history[:10])


def _render_token_search(bundle: FeatureDataBundle, mode_key: str) -> None:
    st.subheader("Token -> Similar Features")
    st.caption("Based on `find_features_for_token`: use token -> top contributing SAE features.")
    st.caption("Search scope: only features in the current mode bundle (load/infer).")

    default_tokenizer = DEFAULT_TOKENIZER_PATH
    if bundle.metadata and isinstance(bundle.metadata.get("tokenizer_path"), str):
        default_tokenizer = str(bundle.metadata.get("tokenizer_path", ""))

    col1, col2 = st.columns(2)
    with col1:
        query_token = st.text_input("Query token", value="")
        tokenizer_path = st.text_input("Tokenizer path (if using token text)", value=default_tokenizer)
    with col2:
        query_token_id_str = st.text_input("Query token id (optional)", value="")
        n_best = st.slider("Top N features", min_value=1, max_value=50, value=10)

    if st.button("Find Similar Features"):
        token_id = None
        if query_token_id_str.strip():
            try:
                token_id = int(query_token_id_str.strip())
            except Exception:
                st.error("Invalid token id.")
                return

        rows, warn = find_features_for_token_query(
            bundle=bundle,
            query_token=query_token.strip() or None,
            query_token_id=token_id,
            tokenizer_path=tokenizer_path.strip() or None,
            n_best=int(n_best),
        )
        st.session_state[f"token_search_rows_{mode_key}"] = rows
        st.session_state[f"token_search_meta_{mode_key}"] = {
            "query_token": query_token.strip(),
            "query_token_id": token_id,
            "n_best": int(n_best),
            "warn": warn,
        }

    rows = st.session_state.get(f"token_search_rows_{mode_key}", [])
    meta = st.session_state.get(f"token_search_meta_{mode_key}", {})
    warn = meta.get("warn")
    if warn:
        st.warning(warn)
    if rows:
        token_desc = meta.get("query_token") or f"id={meta.get('query_token_id')}"
        st.plotly_chart(
            _plot_token_search_attribution(bundle, rows, str(token_desc)),
            use_container_width=True,
        )
    elif meta:
        st.info("No results.")


def _plot_prompt_acts(record: dict, title: str) -> go.Figure:
    tokens = [str(x) for x in record.get("tokens", [])]
    acts = [float(x) for x in record.get("feat_acts", [])]
    idx = list(range(len(tokens)))
    short_tokens = [t.replace("\n", "\\n")[:14] for t in tokens]

    fig = go.Figure(
        data=[
            go.Bar(
                x=idx,
                y=acts,
                marker_color="#ff8c00",
                customdata=tokens,
                hovertemplate="idx=%{x}<br>token=%{customdata}<br>act=%{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Token Position",
        yaxis_title="Feature Activation",
        xaxis=dict(
            tickmode="array",
            tickvals=idx,
            ticktext=short_tokens,
            tickangle=-35,
        ),
        height=380,
        margin=dict(l=20, r=20, t=56, b=90),
    )
    return fig


def _render_prompt_activation_panel(
    bundle: FeatureDataBundle,
    feature_id: int,
    mode_key: str,
    *,
    compact: bool = False,
) -> None:
    if not compact:
        st.subheader("Prompt Activation")
        st.caption("Run local model + SAE on custom text and visualize this feature's token activations.")

    meta = bundle.metadata or {}
    default_model = str(meta.get("model_path", DEFAULT_TOKENIZER_PATH))
    default_tokenizer = str(meta.get("tokenizer_path", DEFAULT_TOKENIZER_PATH))
    default_sae = str(meta.get("sae_path", ""))
    default_device = str(meta.get("device", "auto"))
    _init_runtime_widget_state(default_sae_path=default_sae, default_device=default_device)

    c1, c2 = st.columns(2)
    with c1:
        model_path = st.text_input("Model path/name", value=default_model, key=f"prompt_model_{mode_key}")
        tokenizer_path = st.text_input("Tokenizer path/name", value=default_tokenizer, key=f"prompt_tok_{mode_key}")
    with c2:
        sae_path = st.text_input(
            "SAE path",
            key="runtime_sae_path_ui",
            value=str(st.session_state.get("runtime_sae_path", default_sae)),
        )
        device = _device_selectbox(
            "Device",
            key="runtime_device_ui",
            default=str(st.session_state.get("runtime_device", default_device)),
        )
    _sync_runtime_widget_to_store()

    prompt_text = st.text_area(
        "Input text",
        value="The Eiffel Tower is located in",
        height=120,
        key=f"prompt_text_{mode_key}",
    )

    cache_key = f"prompt_act_cache_{mode_key}"
    if st.button("Run Prompt Activation"):
        if not model_path or not sae_path:
            st.error("Please provide model path and sae path.")
            return
        try:
            with st.spinner("Computing feature activations on prompt..."):
                result = feature_activations_for_prompt(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    sae_path=sae_path,
                    feature_id=int(feature_id),
                    prompt_text=prompt_text,
                    device=device,
                    seed=42,
                    match_infer_pipeline=True,
                )
            st.session_state[cache_key][int(feature_id)] = result
        except Exception as exc:
            st.error(f"Prompt activation failed: {exc}")

    result = st.session_state.get(cache_key, {}).get(int(feature_id))
    if not result:
        st.info("No prompt activation result for this feature yet.")
        return

    st.success(
        f"Max activation {result.get('max_activation', 0.0):.4f} at token "
        f"#{result.get('max_activation_token_index', -1)} ({result.get('max_activation_token', '')})"
    )
    record = result["record"]
    st.plotly_chart(
        _plot_prompt_acts(record, title=f"Feature {feature_id} Activation Over Prompt Tokens"),
        use_container_width=True,
    )
    st.markdown("#### Token Heat View")
    _render_sequence_html([record], max_records=1)
    with st.expander("Raw result"):
        st.json(result)


def _render_function_menu(mode_key: str) -> str:
    options = ["Feature Explorer", "Explain", "Token Search", "Steer", "Explanation Eval", "Data Summary"]
    state_key = f"menu_{mode_key}"
    selected = st.session_state.get(state_key, options[0])
    if selected not in options:
        selected = options[0]
        st.session_state[state_key] = selected

    st.markdown("<div class='function-menu-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='function-menu-title'>Function Menu</div>", unsafe_allow_html=True)

    row1 = options[:3]
    row2 = options[3:6]

    cols1 = st.columns(3)
    for idx, opt in enumerate(row1):
        with cols1[idx]:
            if st.button(
                opt,
                key=f"menu_btn_{mode_key}_{opt}",
                use_container_width=True,
                type="primary" if selected == opt else "secondary",
            ):
                if selected != opt:
                    st.session_state[state_key] = opt
                    st.rerun()

    cols2 = st.columns(3)
    for idx, opt in enumerate(row2):
        with cols2[idx]:
            if st.button(
                opt,
                key=f"menu_btn_{mode_key}_{opt}",
                use_container_width=True,
                type="primary" if selected == opt else "secondary",
            ):
                if selected != opt:
                    st.session_state[state_key] = opt
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.get(state_key, selected)


def _render_explanation_eval_panel(bundle: FeatureDataBundle, feature_id: int, mode_key: str) -> None:
    st.subheader("Explanation Quality Eval")
    st.caption("Default mode: rule-based sample generation + Prompt Activation and Steer scoring. LLM generation interface is reserved.")

    meta = bundle.metadata or {}
    default_model = str(meta.get("model_path", DEFAULT_TOKENIZER_PATH))
    default_tokenizer = str(meta.get("tokenizer_path", DEFAULT_TOKENIZER_PATH))
    default_sae = str(meta.get("sae_path", ""))
    default_device = str(meta.get("device", "auto"))
    _init_runtime_widget_state(default_sae_path=default_sae, default_device=default_device)

    cache_name, _ = _cache_names(mode_key)
    explain_cache = st.session_state.get(cache_name, {})
    cached_explanations = {
        method: str(txt)
        for (fid, method), txt in explain_cache.items()
        if int(fid) == int(feature_id)
    }

    if cached_explanations:
        methods = list(cached_explanations.keys())
        selected_method = st.selectbox(
            "Explanation source",
            options=methods + ["manual"],
            index=0,
            key=f"eval_source_{mode_key}_{feature_id}",
        )
        default_expl = cached_explanations.get(selected_method, "") if selected_method != "manual" else ""
    else:
        st.info("No cached explanation found for this feature. You can paste one manually.")
        default_expl = ""

    explanation_text = st.text_area(
        "Explanation text",
        value=default_expl,
        height=90,
        key=f"eval_expl_text_{mode_key}_{feature_id}",
    )

    c1, c2 = st.columns(2)
    with c1:
        model_path = st.text_input("Model path/name", value=default_model, key=f"eval_model_{mode_key}")
        tokenizer_path = st.text_input("Tokenizer path/name", value=default_tokenizer, key=f"eval_tok_{mode_key}")
    with c2:
        sae_path = st.text_input(
            "SAE path",
            key="runtime_sae_path_ui",
            value=str(st.session_state.get("runtime_sae_path", default_sae)),
        )
        device = _device_selectbox(
            "Device",
            key="runtime_device_ui",
            default=str(st.session_state.get("runtime_device", default_device)),
        )
    _sync_runtime_widget_to_store()

    cg1, cg2, cg3 = st.columns(3)
    with cg1:
        n_positive = st.number_input("Positive samples", min_value=1, max_value=8, value=3, key=f"eval_npos_{mode_key}")
        n_negative = st.number_input("Negative samples", min_value=1, max_value=8, value=3, key=f"eval_nneg_{mode_key}")
    with cg2:
        n_neutral = st.number_input("Neutral samples", min_value=1, max_value=6, value=2, key=f"eval_nneu_{mode_key}")
        steer_strength = st.number_input("Steer strength", min_value=-300.0, max_value=300.0, value=100.0, step=1.0, key=f"eval_strength_{mode_key}")
    with cg3:
        steer_max_new_tokens = st.number_input("Steer max new tokens", min_value=1, max_value=96, value=32, key=f"eval_newtok_{mode_key}")
        sample_gen_mode = st.selectbox(
            "Sample generation",
            options=["default", "llm"],
            index=0,
            key=f"eval_gen_mode_{mode_key}",
            help="`llm` is a reserved interface. If not configured in backend, it will fallback to default.",
        )
    steer_method = st.selectbox(
        "Steer method",
        options=["simple_additive", "orthogonal_decomp", "projection_cap"],
        index=0,
        key=f"eval_steer_method_{mode_key}",
        format_func=lambda x: {
            "simple_additive": "simple additive",
            "orthogonal_decomp": "orthogonal decomp",
            "projection_cap": "projection cap",
        }.get(str(x), str(x)),
    )

    eval_cache_name = _eval_cache_name(mode_key)
    eval_cache = st.session_state.get(eval_cache_name, {})
    cached_eval = eval_cache.get(int(feature_id))
    if cached_eval:
        m = cached_eval.get("metrics", {})
        st.markdown(
            f"<div class='metric-card'><div class='k'>Cached Final Score</div><div class='v'>{float(m.get('final_score', 0.0)):.1f} / 100 ({_esc_html(m.get('grade', 'N/A'))})</div></div>",
            unsafe_allow_html=True,
        )

    if st.button("Run Explanation Eval"):
        if not explanation_text.strip():
            st.error("Please provide explanation text (or generate one in Explain panel first).")
            return
        if not model_path.strip() or not sae_path.strip():
            st.error("Model path and SAE path are required.")
            return
        try:
            with st.spinner("Running explanation quality evaluation..."):
                result = evaluate_feature_explanation_quality(
                    feature_id=int(feature_id),
                    explanation_text=explanation_text,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    sae_path=sae_path,
                    device=device,
                    sample_generation_mode=sample_gen_mode,
                    n_positive=int(n_positive),
                    n_negative=int(n_negative),
                    n_neutral=int(n_neutral),
                    steer_strength=float(steer_strength),
                    steer_method=str(steer_method),
                    steer_max_new_tokens=int(steer_max_new_tokens),
                    steer_top_k_next_tokens=10,
                    seed=42,
                )
            st.session_state[eval_cache_name][int(feature_id)] = result
            cached_eval = result
            st.success("Explanation evaluation completed.")
        except Exception as exc:
            st.error(f"Explanation evaluation failed: {exc}")

    cached_eval = st.session_state.get(eval_cache_name, {}).get(int(feature_id))
    if not cached_eval:
        st.info("No evaluation result for this feature yet.")
        return

    metrics = cached_eval.get("metrics", {})
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Final Score</div><div class='v'>{float(metrics.get('final_score', 0.0)):.1f}</div></div>",
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Activation Score</div><div class='v'>{float(metrics.get('activation_score', 0.0)):.1f}</div></div>",
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Steer Score</div><div class='v'>{float(metrics.get('steer_score', 0.0)):.1f} ({_esc_html(metrics.get('grade', 'N/A'))})</div></div>",
            unsafe_allow_html=True,
        )

    warnings = cached_eval.get("warnings", [])
    if warnings:
        with st.expander("Warnings"):
            for w in warnings:
                st.warning(str(w))

    samples = cached_eval.get("samples", {})
    st.markdown("#### Generated Test Samples")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Positive**")
        for p in samples.get("positive", []):
            st.caption(p)
    with t2:
        st.markdown("**Negative**")
        for p in samples.get("negative", []):
            st.caption(p)
    with t3:
        st.markdown("**Neutral**")
        for p in samples.get("neutral", []):
            st.caption(p)

    act_rows = cached_eval.get("activation_tests", [])
    steer_rows = cached_eval.get("steer_tests", [])
    if act_rows:
        st.markdown("#### Prompt Activation Tests")
        st.dataframe(pd.DataFrame(act_rows), use_container_width=True)
    if steer_rows:
        st.markdown("#### Steer Direction Tests")
        st.dataframe(pd.DataFrame(steer_rows), use_container_width=True)

    with st.expander("Raw eval payload"):
        st.json(cached_eval)


def _render_steer_panel(bundle: FeatureDataBundle, feature_id: int, mode_key: str) -> None:
    st.subheader("Feature Steering")

    meta = bundle.metadata or {}
    default_model = str(meta.get("model_path", DEFAULT_TOKENIZER_PATH))
    default_tokenizer = str(meta.get("tokenizer_path", DEFAULT_TOKENIZER_PATH))
    default_sae = str(meta.get("sae_path", ""))
    default_device = str(meta.get("device", "auto"))
    _init_runtime_widget_state(default_sae_path=default_sae, default_device=default_device)

    # Show cached explanation for this feature if exists.
    st.markdown("**Cached explanations for this feature**")
    cache_name, _ = _cache_names(mode_key)
    explain_cache = st.session_state.get(cache_name, {})
    cached = {
        method: txt
        for (fid, method), txt in explain_cache.items()
        if fid == int(feature_id)
    }
    if cached:
        st.markdown("<div class='panel-shell'>", unsafe_allow_html=True)
        for method, txt in cached.items():
            st.markdown(
                "<div class='explain-card'>"
                f"<div class='explain-method'>{_esc_html(method)}</div>"
                f"<div class='explain-text'>{_esc_html(txt)}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("No cached explanation yet. Generate one in the Explain panel first.")

    prompt_text = st.text_area("Input prompt", value="The city of Paris is", height=120)
    steer_strength = st.slider("Activation steering strength", min_value=-300.0, max_value=300.0, value=100.0, step=1.0)
    steer_method = st.selectbox(
        "Steer method",
        options=["simple_additive", "orthogonal_decomp", "projection_cap"],
        index=0,
        key=f"steer_method_{mode_key}",
        format_func=lambda x: {
            "simple_additive": "simple additive",
            "orthogonal_decomp": "orthogonal decomp",
            "projection_cap": "projection cap",
        }.get(str(x), str(x)),
    )

    c1, c2 = st.columns(2)
    with c1:
        model_path = st.text_input("Model path/name", value=default_model)
        tokenizer_path = st.text_input("Tokenizer path/name", value=default_tokenizer)
    with c2:
        sae_path = st.text_input(
            "SAE path",
            key="runtime_sae_path_ui",
            value=str(st.session_state.get("runtime_sae_path", default_sae)),
        )
        device = _device_selectbox(
            "Device",
            key="runtime_device_ui",
            default=str(st.session_state.get("runtime_device", default_device)),
        )
    _sync_runtime_widget_to_store()

    c3, c4 = st.columns(2)
    with c3:
        max_new_tokens = st.number_input("Max new tokens", min_value=1, max_value=128, value=32)
    with c4:
        top_k_next = st.number_input("Top next-token K", min_value=1, max_value=50, value=10)

    if st.button("Run Steering"):
        if not model_path or not tokenizer_path or not sae_path or not prompt_text.strip():
            st.error("Please provide model/tokenizer/sae paths and prompt text.")
            return
        try:
            with st.spinner("Running steering..."):
                result = steer_feature_on_text(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    sae_path=sae_path,
                    feature_id=int(feature_id),
                    prompt_text=prompt_text,
                    steer_strength=float(steer_strength),
                    steer_method=str(steer_method),
                    device=device,
                    max_new_tokens=int(max_new_tokens),
                    top_k_next_tokens=int(top_k_next),
                )
            st.success("Steering done.")
            st.markdown("**Baseline continuation**")
            st.code(result["baseline_generated_text"])
            st.markdown("**Steered continuation**")
            st.code(result["steered_generated_text"])

            st.markdown("**Top next tokens (baseline)**")
            st.dataframe(pd.DataFrame(result["baseline_top_next_tokens"]), use_container_width=True)
            st.markdown("**Top next tokens (steered)**")
            st.dataframe(pd.DataFrame(result["steered_top_next_tokens"]), use_container_width=True)

            with st.expander("Raw steering result"):
                st.json(result)
        except Exception as exc:
            st.error(f"Steering failed: {exc}")


def _render_data_summary(bundle: FeatureDataBundle, mode_key: str) -> None:
    st.subheader("Data Summary")
    feature_ids = get_feature_ids(bundle)
    feature_range = f"{feature_ids[0]} .. {feature_ids[-1]}" if feature_ids else "N/A"
    has_logits = bool(bundle.feat_to_logits)
    has_token_map = bool(bundle.token_to_feat)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Mode</div><div class='v'>{_esc_html(mode_key.upper())}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Source</div><div class='v'>{_esc_html(bundle.source)}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='k'>Features</div><div class='v'>{len(feature_ids)}</div></div>",
            unsafe_allow_html=True,
        )

    logits_chip = "<span class='status-chip status-ok'>YES</span>" if has_logits else "<span class='status-chip status-no'>NO</span>"
    token_chip = "<span class='status-chip status-ok'>YES</span>" if has_token_map else "<span class='status-chip status-no'>NO</span>"
    st.markdown(
        (
            "<div class='kv-shell'>"
            f"<div class='kv-row'><span class='kv-k'>Feature ID Range</span><span class='kv-v'>{_esc_html(feature_range)}</span></div>"
            f"<div class='kv-row'><span class='kv-k'>Has feat_to_logits</span><span class='kv-v'>{logits_chip}</span></div>"
            f"<div class='kv-row'><span class='kv-k'>Has token_to_feat</span><span class='kv-v'>{token_chip}</span></div>"
            f"<div class='kv-row'><span class='kv-k'>Activations Path</span><span class='kv-v'>{_esc_html(bundle.activations_path or '')}</span></div>"
            f"<div class='kv-row'><span class='kv-k'>Logits Path</span><span class='kv-v'>{_esc_html(bundle.logits_path or '')}</span></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if bundle.metadata:
        with st.expander("Runtime / Inference Config (Raw)"):
            st.json(bundle.metadata)


def main() -> None:
    st.set_page_config(page_title="SAE Interpreter", layout="wide")
    _inject_styles()
    _init_state()

    st.markdown("<div class='main-title'>SAE Interpreter</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Sparse Autoencoder feature analysis, explanation, and steering.</div>",
        unsafe_allow_html=True,
    )

    load_count = len(get_feature_ids(st.session_state.bundle_load)) if st.session_state.bundle_load else 0
    infer_count = len(get_feature_ids(st.session_state.bundle_infer)) if st.session_state.bundle_infer else 0

    with st.sidebar:
        st.header("Mode")
        source_mode = _render_sidebar_mode_switch()
        _render_mode_cache_visual(load_count, infer_count, source_mode)

    mode_key = "load" if source_mode == "Load existing data" else "infer"

    # Show setup panel for current mode only.
    if mode_key == "load":
        _render_load_existing_ui()
        bundle = st.session_state.bundle_load
    else:
        _render_infer_data_ui()
        bundle = st.session_state.bundle_infer

    if bundle is None:
        st.info(f"No data in current mode ({source_mode}). Configure and load/infer first.")
        return

    feature_ids = get_feature_ids(bundle)
    if not feature_ids:
        st.error("No feature ids available in current mode data.")
        return

    # Feature picker
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_feature = st.selectbox("Selected feature", options=feature_ids, index=0)
    with c2:
        st.metric("TopK in data", _infer_top_k_from_bundle(bundle))

    _render_top_cards(mode_key, bundle, int(selected_feature), len(feature_ids))
    menu = _render_function_menu(mode_key)

    if menu == "Feature Explorer":
        _render_feature_explorer(bundle, int(selected_feature), mode_key=mode_key)
    elif menu == "Explain":
        _render_explanation_panel(bundle, int(selected_feature), mode_key=mode_key)
    elif menu == "Token Search":
        _render_token_search(bundle, mode_key=mode_key)
    elif menu == "Steer":
        _render_steer_panel(bundle, int(selected_feature), mode_key=mode_key)
    elif menu == "Explanation Eval":
        _render_explanation_eval_panel(bundle, int(selected_feature), mode_key=mode_key)
    else:
        _render_data_summary(bundle, mode_key=mode_key)


if __name__ == "__main__":
    main()
