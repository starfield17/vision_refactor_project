"""Launcher notice for the React deploy/statistics web app."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy/statistics web UI launcher")
    parser.add_argument("--config", default="./work-dir/config.toml")
    parser.add_argument("--workdir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print("Streamlit deploy.statistics.ui has been replaced by the React deploy/statistics app.")
    print("Start backend:")
    print("  python -m services.deploy_statistics.api --config ./work-dir/config.toml")
    print("Start web:")
    print("  cd web/deploy_statistics && npm install && npm run dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
