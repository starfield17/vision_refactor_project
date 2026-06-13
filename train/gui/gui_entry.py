"""Train GUI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from train.gui.main_window import TrainMainWindow


def run_gui(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision Train GUI")
    parser.add_argument("--config", default="train/config/config.example.toml")
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1] + (argv or []))
    window = TrainMainWindow(
        config_path=Path(args.config).resolve(),
        workdir_override=str(Path(args.workdir).resolve()) if args.workdir else None,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_gui())
