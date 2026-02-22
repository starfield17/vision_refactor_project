"""Train Web UI (Streamlit, port 7794)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from share.config.config_loader import load_config, save_resolved_config
from share.config.editing import load_merged_user_config, persist_config_overrides
from share.config.schema import FASTER_RCNN_VARIANTS, QUANTIZE_MODES, TRAIN_BACKENDS
from share.kernel.kernel import VisionKernel
from share.kernel.registry import KernelRegistry
from share.kernel.trainer.faster_rcnn import run_faster_rcnn_train
from share.kernel.trainer.yolo import run_yolo_train
from share.kernel.utils.logging import StructuredLogger
from share.types.errors import ConfigError

WEB_HOST = "0.0.0.0"
WEB_PORT = 7794
TRAIN_WEB_EDITABLE_PREFIXES = ("workspace", "data.yolo_dataset_dir", "train", "export")

# ---------------------------------------------------------------------------
# Custom CSS — Modern Light Theme (consistent with statistics dashboard)
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #f4f6f8;
    --bg-card: #ffffff;
    --border: rgba(0, 0, 0, 0.08);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --text-primary: #0f172a;
    --text-muted: #64748b;
    --radius: 16px;
}

.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #0f172a 0%, var(--accent-blue) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.5rem !important;
}
h2, h3, h4, h5, .stSubheader {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Glassmorphism Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--bg-card) 0%, rgba(255,255,255,0.7) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(59, 130, 246, 0.12);
    border-color: rgba(59, 130, 246, 0.3);
}
[data-testid="stMetricLabel"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-size: 2rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(15px) !important;
    border-right: 1px solid var(--border) !important;
}

/* Modern Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 12px 20px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary) !important; }
.stT-selected="true"] {
    background: transparent !important;
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* Status Pills */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 30px;
    font-size: 0.8rem; font-weight: 600;
    font-family: 'Outfit', sans-serif; letter-spacing: 0.05em;
}
.status-pill.ok {
    background: rgba(16,185,129,0.1); color: #059669;
    border: 1px solid rgba(16,185,129,0.3);
}
.status-pill.error {
    background: rgba(239,68,68,0.1); color: #dc2626;
    border: 1px solid rgba(239,68,68,0.3);
}
.status-pill.dry-run {
    background: rgba(245,158,11,0.1); color: #d97706;
    border: 1px solid rgba(245,158,11,0.3);
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background-color: currentColor; }

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 2rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    background: var(--bg-card) !important;
    border-radius: 12px !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 4px; }

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train web UI")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Config override, repeatable.",
    )
    parser.add_argument("--web-mode", action="store_true", help=argparse.SUPPRESS)
    return parser


def _resolve_log_path(cfg: dict[str, Any]) -> Path:
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    return log_file if log_file.is_absolute() else workdir / log_file


def _parse_override_text(raw: str) -> list[str]:
    items: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue  # skip malformed lines
        items.append(text)
    return items


def _read_tail(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _ui_device_to_runtime(choice: str) -> str:
    return "cuda:0" if choice == "gpu" else "cpu"


def _validate_config_path(path_str: str) -> Path | None:
    """Return resolved Path if valid, else None."""
    if not path_str.strip():
        return None
    p = Path(path_str.strip()).resolve()
    if not p.exists():
        return None
    return p


def _format_elapsed(ms: float) -> str:
    """Human-friendly elapsed time."""
    if ms < 1000:
        return f"{ms:.0f} ms"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.1f} s"
    mins = secs / 60
    return f"{mins:.1f} min"


def _choice_index(options: list[str], value: str, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def _cfg_get(cfg: dict[str, Any], path: tuple[str, ...], fallback: Any) -> Any:
    cur: Any = cfg
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_train_once(
    config_path: Path,
    workdir_override: str | None,
    overrides: list[str],
) -> dict[str, Any]:
    cfg = load_config(
        config_path=config_path,
        overrides=overrides,
        workdir_override=workdir_override,
    )

    logger = StructuredLogger(
        log_path=_resolve_log_path(cfg),
        level=cfg["workspace"]["log_level"],
    )
    logger.info("train.web.start", "Train web run started", config_path=str(config_path))

    registry = KernelRegistry()
    registry.register_trainer("yolo", run_yolo_train)
    registry.register_trainer("faster_rcnn", run_faster_rcnn_train)

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=registry)
    result = kernel.run_train()

    resolved_path = result.run_context.run_dir / "config.resolved.toml"
    save_resolved_config(cfg, resolved_path)
    shutil.copyfile(resolved_path, Path(cfg["workspace"]["root"]) / "config.resolved.toml")

    artifacts_path = result.run_context.run_dir / "artifacts.json"
    artifacts_payload: dict[str, Any] = {}
    if artifacts_path.exists():
        artifacts_payload = json.loads(artifacts_path.read_text(encoding="utf-8"))

    return {
        "status": result.status,
        "error": result.error,
        "run_id": result.run_context.run_id,
        "elapsed_ms": result.elapsed_ms,
        "run_dir": str(result.run_context.run_dir),
        "resolved_config": str(resolved_path),
        "artifacts_path": str(artifacts_path),
        "artifacts": artifacts_payload,
        "log_path": str(_resolve_log_path(cfg)),
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def _render_streamlit_ui(
    initial_config: str,
    initial_workdir: str | None,
    initial_overrides: list[str],
) -> int:
    try:
        import streamlit as st
    except Exception as exc:
        print(f"[RUNTIME ERROR] streamlit is required: {exc}", file=sys.stderr)
        return 3

    st.set_page_config(page_title="Train Web", page_icon="🎯", layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # ---- Session state defaults ----
    if "train_web_last" not in st.session_state:
        st.session_state["train_web_last"] = None
    if "train_run_count" not in st.session_state:
        st.session_state["train_run_count"] = 0
    if "train_web_flash" not in st.session_state:
        st.session_state["train_web_flash"] = ""

    # ---- Sidebar: Configuration ----
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        config_value = st.text_input(
            "Config Path",
            value=initial_config,
            help="Path to config.toml file",
        )
        workdir_value = st.text_input(
            "Workdir Override",
            value=initial_workdir or "",
            placeholder="Leave empty to use config default",
        )
        st.divider()

        st.markdown("### 📝 Extra Overrides")
        extra_default = "\n".join(initial_overrides)
        extra_overrides = st.text_area(
            "One KEY=VALUE per line",
            value=extra_default,
            height=120,
            placeholder="train.yolo.weights=../yolo26n.pt\nexport.quantize=true",
            label_visibility="collapsed",
        )

        st.divider()
        run_count = st.session_state.get("train_run_count", 0)
        st.caption(f"Total runs this session: **{run_count}**")

    config_path_resolved = _validate_config_path(config_value)
    editable_cfg: dict[str, Any] | None = None
    config_load_error = ""
    if config_path_resolved is not None:
        try:
            editable_cfg = load_merged_user_config(config_path_resolved)
        except ConfigError as exc:
            config_load_error = str(exc)

    cfg = editable_cfg or {}
    backend_options = sorted(TRAIN_BACKENDS)
    frcnn_options = sorted(FASTER_RCNN_VARIANTS)
    quantize_options = sorted(QUANTIZE_MODES)
    default_backend = str(_cfg_get(cfg, ("train", "backend"), "yolo"))
    default_device = str(_cfg_get(cfg, ("train", "device"), "cpu"))
    default_device_ui = "gpu" if default_device.startswith("cuda") else "cpu"

    # ---- Main Area ----
    st.title("🎯 Train Web")
    st.caption("Run the training pipeline from your browser — kernel-backed, same behavior as CLI.")
    flash = st.session_state.get("train_web_flash", "")
    if flash:
        st.success(flash)
        st.session_state["train_web_flash"] = ""

    tab_params, tab_results, tab_logs = st.tabs(["⚡ Parameters", "📊 Results", "📜 Logs"])

    with tab_params:
        st.markdown("#### Workspace & Data")
        w1, w2 = st.columns(2)
        run_name = w1.text_input(
            "Run Name",
            value=str(_cfg_get(cfg, ("workspace", "run_name"), "exp001")),
            help="workspace.run_name",
        )
        dataset_dir = w2.text_input(
            "YOLO Dataset Dir",
            value=str(_cfg_get(cfg, ("data", "yolo_dataset_dir"), "")),
            help="data.yolo_dataset_dir",
        )

        st.divider()

        # ---- Backend & Device ----
        st.markdown("#### Backend & Device")
        c1, c2, c3 = st.columns(3)
        backend = c1.selectbox(
            "Backend",
            backend_options,
            index=_choice_index(backend_options, default_backend),
            help="Training backend: YOLO or Faster R-CNN.",
        )
        device_choice = c2.selectbox(
            "Device",
            ["cpu", "gpu"],
            index=0 if default_device_ui == "cpu" else 1,
            help="gpu → NVIDIA CUDA (train.device=cuda:0)",
        )
        dry_run = c3.checkbox(
            "Dry Run",
            value=bool(_cfg_get(cfg, ("train", "dry_run"), False)),
            help="When enabled, validate the config without actually training.",
        )
        runtime_device = _ui_device_to_runtime(device_choice)

        # ---- Training Hyperparameters ----
        st.markdown("#### Training Hyperparameters")
        c4, c5, c6, c7 = st.columns(4)
        epochs = c4.number_input(
            "Epochs",
            min_value=1,
            value=int(_cfg_get(cfg, ("train", "epochs"), 1)),
            step=1,
        )
        batch_size = c5.number_input(
            "Batch Size",
            min_value=1,
            value=int(_cfg_get(cfg, ("train", "batch_size"), 4)),
            step=1,
        )
        img_size = c6.number_input(
            "Image Size",
            min_value=32,
            value=int(_cfg_get(cfg, ("train", "img_size"), 640)),
            step=32,
        )
        seed = c7.number_input(
            "Random Seed",
            min_value=0,
            value=int(_cfg_get(cfg, ("train", "seed"), 42)),
            step=1,
        )

        st.divider()

        # ---- Backend-specific Settings ----
        if backend == "yolo":
            st.markdown("#### 🟢 YOLO Settings")
            yolo_weights = st.text_input(
                "YOLO Weights",
                value=str(_cfg_get(cfg, ("train", "yolo", "weights"), "")),
                placeholder="./weights/yolo.pt",
                help="Path to pretrained YOLO weights file.",
            )
            # Faster R-CNN defaults (not used)
            frcnn_variant = str(_cfg_get(cfg, ("train", "faster_rcnn", "variant"), "mobilenet_v3"))
            frcnn_lr = float(_cfg_get(cfg, ("train", "faster_rcnn", "lr"), 0.005))
            frcnn_momentum = float(_cfg_get(cfg, ("train", "faster_rcnn", "momentum"), 0.9))
            frcnn_weight_decay = float(
                _cfg_get(cfg, ("train", "faster_rcnn", "weight_decay"), 0.0005)
            )
            frcnn_num_workers = int(_cfg_get(cfg, ("train", "faster_rcnn", "num_workers"), 0))
            frcnn_max_samples = int(_cfg_get(cfg, ("train", "faster_rcnn", "max_samples"), 0))
        else:
            st.markdown("#### 🔵 Faster R-CNN Settings")
            fc1, fc2 = st.columns(2)
            frcnn_variant = fc1.selectbox(
                "Variant",
                frcnn_options,
                index=_choice_index(
                    frcnn_options,
                    str(_cfg_get(cfg, ("train", "faster_rcnn", "variant"), "mobilenet_v3")),
                ),
                help="Backbone architecture for Faster R-CNN.",
            )
            frcnn_lr = fc2.number_input(
                "Learning Rate",
                min_value=0.0001,
                value=float(_cfg_get(cfg, ("train", "faster_rcnn", "lr"), 0.005)),
                step=0.001, format="%.4f",
            )
            fc3, fc4, fc5 = st.columns(3)
            frcnn_momentum = fc3.number_input(
                "Momentum", min_value=0.0, max_value=1.0,
                value=float(_cfg_get(cfg, ("train", "faster_rcnn", "momentum"), 0.9)),
                step=0.01,
                format="%.2f",
            )
            frcnn_weight_decay = fc4.number_input(
                "Weight Decay",
                min_value=0.0,
                value=float(_cfg_get(cfg, ("train", "faster_rcnn", "weight_decay"), 0.0005)),
                step=0.0001,
                format="%.4f",
            )
            frcnn_num_workers = fc5.number_input(
                "Num Workers",
                min_value=0,
                value=int(_cfg_get(cfg, ("train", "faster_rcnn", "num_workers"), 0)),
                step=1,
                help="DataLoader workers. 0 = main process.",
            )
            frcnn_max_samples = st.number_input(
                "Max Samples (0 = all)",
                min_value=0,
                value=int(_cfg_get(cfg, ("train", "faster_rcnn", "max_samples"), 0)),
                step=100,
                help="Limit the dataset size for debugging. 0 means use all data.",
            )
            # YOLO defaults (not used)
            yolo_weights = str(_cfg_get(cfg, ("train", "yolo", "weights"), ""))

        st.divider()

        # ---- Export Settings ----
        st.markdown("#### Export Settings")
        e1, e2, e3, e4, e5 = st.columns(5)
        export_onnx = e1.checkbox(
            "Export ONNX",
            value=bool(_cfg_get(cfg, ("export", "onnx"), True)),
            help="export.onnx",
        )
        export_quantize = e2.checkbox(
            "Quantize",
            value=bool(_cfg_get(cfg, ("export", "quantize"), True)),
            help="export.quantize",
        )
        export_opset = e3.number_input(
            "ONNX Opset",
            min_value=1,
            value=int(_cfg_get(cfg, ("export", "opset"), 17)),
            step=1,
        )
        export_quantize_mode = e4.selectbox(
            "Quantize Mode",
            quantize_options,
            index=_choice_index(
                quantize_options,
                str(_cfg_get(cfg, ("export", "quantize_mode"), "dynamic")),
            ),
        )
        export_calib_samples = e5.number_input(
            "Calib Samples",
            min_value=0,
            value=int(_cfg_get(cfg, ("export", "calib_samples"), 32)),
            step=1,
        )

        st.divider()

        # ---- Actions ----
        save_col, run_col, info_col = st.columns([1, 1, 3])
        save_clicked = save_col.button("💾 Save Config", use_container_width=True)
        run_clicked = run_col.button("🚀 Run Train", type="primary", use_container_width=True)

        # Config validation feedback
        if not config_value.strip():
            info_col.warning("⚠️ Please provide a config path.")
        elif config_path_resolved is None:
            info_col.error(f"❌ Config file not found: `{config_value}`")
        elif config_load_error:
            info_col.error(f"❌ Cannot read config TOML: `{config_load_error}`")
        elif dry_run:
            info_col.info("ℹ️ Dry-run mode — config will be validated only.")

        cli_overrides = [
            f"workspace.run_name={run_name.strip()}",
            f"data.yolo_dataset_dir={dataset_dir.strip()}",
            f"train.backend={backend}",
            f"train.device={runtime_device}",
            f"train.dry_run={str(bool(dry_run)).lower()}",
            f"train.epochs={int(epochs)}",
            f"train.batch_size={int(batch_size)}",
            f"train.img_size={int(img_size)}",
            f"train.seed={int(seed)}",
            f"export.onnx={str(bool(export_onnx)).lower()}",
            f"export.opset={int(export_opset)}",
            f"export.quantize={str(bool(export_quantize)).lower()}",
            f"export.quantize_mode={export_quantize_mode}",
            f"export.calib_samples={int(export_calib_samples)}",
        ]
        if backend == "yolo":
            cli_overrides.append(f"train.yolo.weights={yolo_weights.strip()}")
        else:
            cli_overrides.extend(
                [
                    f"train.faster_rcnn.variant={frcnn_variant}",
                    f"train.faster_rcnn.lr={frcnn_lr}",
                    f"train.faster_rcnn.momentum={frcnn_momentum}",
                    f"train.faster_rcnn.weight_decay={frcnn_weight_decay}",
                    f"train.faster_rcnn.num_workers={int(frcnn_num_workers)}",
                    f"train.faster_rcnn.max_samples={int(frcnn_max_samples)}",
                ]
            )
        cli_overrides.extend(_parse_override_text(extra_overrides))

        if save_clicked:
            if config_path_resolved is None:
                st.error("Cannot save: config file path is invalid or does not exist.")
            elif config_load_error:
                st.error(f"Cannot save: {config_load_error}")
            else:
                try:
                    persist_config_overrides(
                        config_path=config_path_resolved,
                        overrides=cli_overrides,
                        allowed_prefixes=TRAIN_WEB_EDITABLE_PREFIXES,
                    )
                    st.session_state["train_web_flash"] = (
                        f"Saved training sections to {config_path_resolved}"
                    )
                    st.rerun()
                except ConfigError as exc:
                    st.error(f"**Config Error:** {exc}")

        if run_clicked:
            if config_path_resolved is None:
                st.error("Cannot run: config file path is invalid or does not exist.")
            elif config_load_error:
                st.error(f"Cannot run: {config_load_error}")
            else:
                try:
                    with st.spinner("Running training pipeline..."):
                        result = _run_train_once(
                            config_path=config_path_resolved,
                            workdir_override=workdir_value.strip() or None,
                            overrides=cli_overrides,
                        )
                    st.session_state["train_web_last"] = result
                    st.session_state["train_run_count"] = run_count + 1
                    st.rerun()
                except ConfigError as exc:
                    st.error(f"**Config Error:** {exc}")
                except Exception as exc:
                    st.error(f"**Runtime Error:** {exc}")
                    with st.expander("Traceback", expanded=False):
                        st.code(traceback.format_exc())

    # ---- Results Tab ----
    result = st.session_state.get("train_web_last")

    with tab_results:
        if not result:
            st.info("No results yet — run the training pipeline first.")
        else:
            is_ok = result.get("status") == "ok"

            # Status pill
            pill_cls = "ok" if is_ok else "error"
            pill_text = "SUCCESS" if is_ok else "FAILED"
            st.markdown(
                f'<span class="status-pill {pill_cls}">'
                f'<span class="status-dot"></span>{pill_text}</span>',
                unsafe_allow_html=True,
            )
            st.write("")

            # Metrics row
            r1, r2, r3 = st.columns(3)
            r1.metric("Run ID", str(result["run_id"]))
            r2.metric("Elapsed", _format_elapsed(float(result["elapsed_ms"])))
            r3.metric("Status", str(result["status"]).upper())

            if not is_ok and result.get("error"):
                st.error(f"**Error:** {result['error']}")

            # Run paths
            st.markdown("#### 📂 Output Paths")
            st.code(
                f"run_dir:         {result['run_dir']}\n"
                f"resolved_config: {result['resolved_config']}\n"
                f"artifacts:       {result['artifacts_path']}",
                language="text",
            )

            # Artifacts
            st.markdown("#### 🗂️ Artifacts")
            artifacts = result.get("artifacts", {})
            if artifacts:
                st.json(artifacts)
            else:
                st.caption("No artifacts produced.")

    # ---- Logs Tab ----
    with tab_logs:
        if not result:
            st.info("No logs available yet — run the pipeline first.")
        else:
            log_path = Path(result["log_path"])
            st.markdown(f"**Log file:** `{log_path}`")

            max_lines = st.select_slider(
                "Tail lines", options=[50, 120, 300, 500], value=120,
            )
            log_text = _read_tail(log_path, max_lines=max_lines)
            st.code(log_text or "(empty log)", language="log")

    return 0


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def _launch_streamlit(config_path: str, workdir: str | None, overrides: list[str]) -> int:
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).resolve()),
        "--server.address", WEB_HOST,
        "--server.port", str(WEB_PORT),
        "--", "--web-mode", "--config", config_path,
    ]
    if workdir:
        cmd.extend(["--workdir", workdir])
    for item in overrides:
        cmd.extend(["--set", item])
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, _unknown = parser.parse_known_args(argv)

    if args.web_mode:
        config = args.config or "./work-dir/config.toml"
        return _render_streamlit_ui(
            initial_config=config,
            initial_workdir=args.workdir,
            initial_overrides=list(args.set),
        )

    if not args.config:
        print("[USAGE ERROR] --config is required when launching train web.", file=sys.stderr)
        return 2

    return _launch_streamlit(
        config_path=str(Path(args.config).resolve()),
        workdir=str(Path(args.workdir).resolve()) if args.workdir else None,
        overrides=list(args.set),
    )


if __name__ == "__main__":
    raise SystemExit(main())
