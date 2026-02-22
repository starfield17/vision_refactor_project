"""Autolabel Web UI (Streamlit, port 7795)."""

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
from share.kernel.autolabel.llm_autolabel import run_llm_autolabel
from share.kernel.autolabel.model_autolabel import run_model_autolabel
from share.kernel.kernel import VisionKernel
from share.kernel.registry import KernelRegistry
from share.kernel.utils.logging import StructuredLogger
from share.types.errors import ConfigError

WEB_HOST = "0.0.0.0"
WEB_PORT = 7795

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
    --accent-purple: #8b5cf6;
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
    background: linear-gradient(135deg, #0f172a 0%, var(--accent-purple) 100%) !important;
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
    box-shadow: 0 12px 28px rgba(139, 92, 246, 0.12);
    border-color: rgba(139, 92, 246, 0.3);
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
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent-purple) !important;
    border-bottom: 2px solid var(--accent-purple) !important;
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
.status-dot { width: 8px; height: 8px; border-radius: 50%; background-color: currentColor; }

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
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
    box-shadow: 0 8px 20px rgba(139, 92, 246, 0.3) !important;
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
    parser = argparse.ArgumentParser(description="Autolabel web UI")
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
            continue  # skip malformed lines silently
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


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_autolabel_once(
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
    logger.info("autolabel.web.start", "Autolabel web run started", config_path=str(config_path))

    registry = KernelRegistry()
    registry.register_autolabeler("model", run_model_autolabel)
    registry.register_autolabeler("llm", run_llm_autolabel)

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=registry)
    result = kernel.run_autolabel()

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

    st.set_page_config(page_title="Autolabel Web", page_icon="🧷", layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # ---- Session state defaults ----
    if "autolabel_web_last" not in st.session_state:
        st.session_state["autolabel_web_last"] = None
    if "autolabel_run_count" not in st.session_state:
        st.session_state["autolabel_run_count"] = 0

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
            placeholder="autolabel.model.onnx_model=./work-dir/models/exp/model.onnx\nautolabel.llm.base_url=http://...",
            label_visibility="collapsed",
        )

        # Sidebar status
        st.divider()
        run_count = st.session_state.get("autolabel_run_count", 0)
        st.caption(f"Total runs this session: **{run_count}**")

    # ---- Main Area ----
    st.title("🧷 Autolabel Web")
    st.caption("Run the autolabel pipeline from your browser — model or LLM mode.")

    # ---- Parameters ----
    tab_params, tab_results, tab_logs = st.tabs(["⚡ Parameters", "📊 Results", "📜 Logs"])

    with tab_params:
        st.markdown("#### Pipeline Mode & Device")
        c1, c2 = st.columns(2)
        mode = c1.selectbox("Mode", ["model", "llm"], index=0, help="model = ONNX inference, llm = LLM API")
        device_choice = c2.selectbox(
            "Device", ["cpu", "gpu"], index=0,
            help="gpu → NVIDIA CUDA (train.device=cuda:0)",
        )
        runtime_device = _ui_device_to_runtime(device_choice)

        st.markdown("#### General Settings")
        c3, c4, c5 = st.columns(3)
        confidence = c3.slider(
            "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Minimum confidence for accepting a prediction.",
        )
        batch_size = c4.number_input("Batch Size", min_value=1, value=2, step=1)
        on_conflict = c5.selectbox(
            "On Conflict", ["skip", "overwrite", "merge"], index=0,
            help="How to handle labels that already exist.",
        )

        c6, _ = st.columns([1, 2])
        visualize = c6.checkbox("Visualize Results", value=True)

        st.divider()

        # ---- Mode-specific settings ----
        if mode == "model":
            st.markdown("#### 🤖 Model Settings")
            cm1, cm2 = st.columns(2)
            model_backend = cm1.selectbox(
                "Model Backend", ["yolo", "faster_rcnn"], index=0,
                help="Which model backend to use for inference.",
            )
            model_path = cm2.text_input(
                "Model Path (ONNX)",
                value="",
                placeholder="./work-dir/models/exp001/model-int8.onnx",
            )
            llm_max_images = 0
        else:
            st.markdown("#### 🧠 LLM Settings")
            llm_max_images = st.number_input(
                "Max Images (0 = all)", min_value=0, value=0, step=1,
                help="Limit the number of images sent to the LLM. 0 means no limit.",
            )
            model_backend = "yolo"
            model_path = ""

        st.divider()

        # ---- Run Button ----
        run_col, info_col = st.columns([1, 3])
        run_clicked = run_col.button("🚀 Run Autolabel", type="primary", use_container_width=True)

        # Config validation feedback
        config_path_resolved = _validate_config_path(config_value)
        if not config_value.strip():
            info_col.warning("⚠️ Please provide a config path.")
        elif config_path_resolved is None:
            info_col.error(f"❌ Config file not found: `{config_value}`")

        if run_clicked:
            if config_path_resolved is None:
                st.error("Cannot run: config file path is invalid or does not exist.")
            else:
                try:
                    cli_overrides = [
                        f"autolabel.mode={mode}",
                        f"train.device={runtime_device}",
                        f"autolabel.confidence={float(confidence)}",
                        f"autolabel.visualize={str(bool(visualize)).lower()}",
                        f"autolabel.batch_size={int(batch_size)}",
                        f"autolabel.on_conflict={on_conflict}",
                    ]
                    if mode == "model":
                        cli_overrides.append(f"autolabel.model.backend={model_backend}")
                        if model_path.strip():
                            cli_overrides.append(f"autolabel.model.onnx_model={model_path.strip()}")
                    else:
                        cli_overrides.append(f"autolabel.llm.max_images={int(llm_max_images)}")

                    cli_overrides.extend(_parse_override_text(extra_overrides))

                    with st.spinner("Running autolabel pipeline..."):
                        result = _run_autolabel_once(
                            config_path=config_path_resolved,
                            workdir_override=workdir_value.strip() or None,
                            overrides=cli_overrides,
                        )
                    st.session_state["autolabel_web_last"] = result
                    st.session_state["autolabel_run_count"] = run_count + 1
                    st.rerun()
                except ConfigError as exc:
                    st.error(f"**Config Error:** {exc}")
                except Exception as exc:
                    st.error(f"**Runtime Error:** {exc}")
                    with st.expander("Traceback", expanded=False):
                        st.code(traceback.format_exc())

    # ---- Results Tab ----
    result = st.session_state.get("autolabel_web_last")

    with tab_results:
        if not result:
            st.info("No results yet — run the autolabel pipeline first.")
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
        print("[USAGE ERROR] --config is required when launching autolabel web.", file=sys.stderr)
        return 2

    return _launch_streamlit(
        config_path=str(Path(args.config).resolve()),
        workdir=str(Path(args.workdir).resolve()) if args.workdir else None,
        overrides=list(args.set),
    )


if __name__ == "__main__":
    raise SystemExit(main())

