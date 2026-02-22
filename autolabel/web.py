"""Autolabel Web UI (Streamlit, port 7795)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autolabel web UI")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable.",
    )
    parser.add_argument("--web-mode", action="store_true", help=argparse.SUPPRESS)
    return parser


def _resolve_log_path(cfg: dict[str, Any]) -> Path:
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    if log_file.is_absolute():
        return log_file
    return workdir / log_file


def _parse_override_text(raw: str) -> list[str]:
    items: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        items.append(text)
    return items


def _read_tail(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _ui_device_to_runtime(choice: str) -> str:
    if choice == "gpu":
        return "cuda:0"
    return "cpu"


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
    st.title("Autolabel Web")
    st.caption("Run autolabel pipeline from browser (model or llm mode).")

    config_value = st.text_input("Config Path", value=initial_config)
    workdir_value = st.text_input("Workdir Override (optional)", value=initial_workdir or "")

    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("Mode", ["model", "llm"], index=0)
    device_choice = c2.selectbox(
        "Device",
        ["cpu", "gpu"],
        index=0,
        help="gpu will use NVIDIA CUDA (train.device=cuda:0).",
    )
    confidence = c3.slider("Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    visualize = c4.checkbox("Visualize", value=True)
    runtime_device = _ui_device_to_runtime(device_choice)

    c5, c6 = st.columns(2)
    batch_size = c5.number_input("Batch Size", min_value=1, value=2, step=1)
    on_conflict = c6.selectbox("On Conflict", ["skip", "overwrite", "merge"], index=0)

    model_backend = "yolo"
    model_path = ""
    llm_max_images = 0

    if mode == "model":
        c7, c8 = st.columns(2)
        model_backend = c7.selectbox("Model Backend", ["yolo", "faster_rcnn"], index=0)
        model_path = c8.text_input("Model Path", value="")
    else:
        llm_max_images = st.number_input("LLM max_images (0 means all)", min_value=0, value=0, step=1)

    extra_default = "\n".join(initial_overrides)
    extra_overrides = st.text_area(
        "Extra Overrides (one KEY=VALUE per line, optional)",
        value=extra_default,
        height=140,
        placeholder="autolabel.model.onnx_model=./work-dir/models/exp/model.onnx\nautolabel.llm.base_url=http://...",
    )

    if "autolabel_web_last" not in st.session_state:
        st.session_state["autolabel_web_last"] = None

    if st.button("🚀 Run Autolabel", type="primary"):
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

            with st.spinner("Running autolabel..."):
                result = _run_autolabel_once(
                    config_path=Path(config_value).resolve(),
                    workdir_override=workdir_value.strip() or None,
                    overrides=cli_overrides,
                )
            st.session_state["autolabel_web_last"] = result
        except ConfigError as exc:
            st.error(f"[CONFIG ERROR] {exc}")
        except Exception as exc:
            st.error(f"[RUNTIME ERROR] {exc}")

    result = st.session_state.get("autolabel_web_last")
    if result:
        st.subheader("Last Run")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Status", str(result["status"]).upper())
        r2.metric("Elapsed (ms)", f"{float(result['elapsed_ms']):.3f}")
        r3.metric("Run ID", str(result["run_id"]))
        r4.metric("Has Error", "YES" if result.get("error") else "NO")

        if result.get("status") != "ok":
            st.error(f"Error: {result.get('error')}")
        else:
            st.success("Autolabel completed successfully.")

        st.code(
            f"run_dir: {result['run_dir']}\n"
            f"resolved_config: {result['resolved_config']}\n"
            f"artifacts: {result['artifacts_path']}"
        )

        with st.expander("Artifacts JSON", expanded=True):
            st.json(result.get("artifacts", {}))

        log_text = _read_tail(Path(result["log_path"]), max_lines=120)
        with st.expander("Recent Log Tail (120 lines)", expanded=False):
            st.code(log_text or "(empty)")

    return 0


def _launch_streamlit(config_path: str, workdir: str | None, overrides: list[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        WEB_HOST,
        "--server.port",
        str(WEB_PORT),
        "--",
        "--web-mode",
        "--config",
        config_path,
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
