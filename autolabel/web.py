"""Launcher notice for the React train/autolabel web app."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autolabel web UI launcher")
    parser.add_argument("--config", default="./work-dir/config.toml")
    parser.add_argument("--workdir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print("Streamlit autolabel.web has been replaced by the React train/autolabel app.")
    print("Start backend:")
    print("  python -m services.train_autolabel.api --config ./work-dir/config.toml")
    print("Start web:")
    print("  cd web/train_autolabel && npm install && npm run dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
