"""Compatibility launcher for the deploy/statistics backend service."""

from __future__ import annotations

from services.deploy_statistics.api import build_parser, create_app, main

__all__ = ["build_parser", "create_app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
