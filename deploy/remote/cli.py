"""Deploy remote CLI entrypoint (Phase 5)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from share.config.config_loader import load_config, save_resolved_config
from share.kernel.deploy.remote_server import run_remote_deploy
from share.kernel.kernel import VisionKernel
from share.kernel.registry import KernelRegistry
from share.kernel.utils.logging import StructuredLogger
from share.types.errors import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 deploy remote CLI")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", required=True, help="Path to config.toml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable. Example: --set deploy.remote.listen_port=60052",
    )
    return parser


def _resolve_log_path(cfg: dict) -> Path:
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    if log_file.is_absolute():
        return log_file
    return workdir / log_file


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None

    try:
        cfg = load_config(
            config_path=config_path,
            overrides=args.set,
            workdir_override=workdir_override,
        )
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    log_path = _resolve_log_path(cfg)
    logger = StructuredLogger(log_path=log_path, level=cfg["workspace"]["log_level"])
    logger.info("deploy.remote.cli.start", "CLI started", config_path=str(config_path))

    registry = KernelRegistry()
    registry.register_deployer("remote", run_remote_deploy)

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=registry)
    result = kernel.run_deploy_remote()

    resolved_path = result.run_context.run_dir / "config.resolved.toml"
    save_resolved_config(cfg, resolved_path)
    latest_link = Path(cfg["workspace"]["root"]) / "config.resolved.toml"
    shutil.copyfile(resolved_path, latest_link)

    logger.info(
        "deploy.remote.cli.done",
        "CLI finished",
        run_id=result.run_context.run_id,
        status=result.status,
        resolved_config=str(resolved_path),
    )

    print(f"run_id={result.run_context.run_id}")
    print(f"status={result.status}")
    print(f"resolved_config={resolved_path}")
    print(f"artifacts={result.run_context.run_dir / 'artifacts.json'}")

    if result.status != "ok":
        print(f"error={result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
