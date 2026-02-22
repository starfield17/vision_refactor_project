"""Unified kernel entrypoints (Phase 1 skeleton)."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from share.kernel.registry import KernelRegistry
from share.kernel.utils.logging import StructuredLogger


@dataclass(slots=True)
class RunContext:
    run_id: str
    mode: str
    workdir: Path
    run_dir: Path


@dataclass(slots=True)
class RunResult:
    run_context: RunContext
    status: str
    backend: str
    elapsed_ms: float
    artifacts: dict[str, Any]
    error: str | None = None


class VisionKernel:
    def __init__(self, cfg: dict[str, Any], logger: StructuredLogger, registry: KernelRegistry) -> None:
        self.cfg = cfg
        self.logger = logger
        self.registry = registry
        self.workdir = Path(cfg["workspace"]["root"])

    def _ensure_workdir_layout(self) -> None:
        required_dirs = [
            self.workdir / "datasets",
            self.workdir / "models",
            self.workdir / "runs",
            self.workdir / "outputs",
            self.workdir / "stats",
            self.workdir / "tmp",
        ]
        for d in required_dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _make_run_context(self, mode: str) -> RunContext:
        now = datetime.now(tz=timezone.utc)
        run_id = f"{self.cfg['workspace']['run_name']}-{now.strftime('%Y%m%d-%H%M%S')}"
        run_dir = self.workdir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunContext(run_id=run_id, mode=mode, workdir=self.workdir, run_dir=run_dir)

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def run_train(self) -> RunResult:
        self._ensure_workdir_layout()
        run_ctx = self._make_run_context(mode="train")

        backend = self.cfg["train"]["backend"]
        trainer = self.registry.get_trainer(backend)

        self.logger.info(
            "train.start",
            "Train pipeline started",
            run_id=run_ctx.run_id,
            backend=backend,
            workdir=str(self.workdir),
        )

        start = time.perf_counter()
        status = "ok"
        error: str | None = None
        artifacts: dict[str, Any] = {}

        try:
            artifacts = trainer(
                self.cfg,
                {
                    "run_id": run_ctx.run_id,
                    "run_dir": str(run_ctx.run_dir),
                    "logger": self.logger,
                },
            )
        except Exception as exc:  # Keep kernel resilient and let CLI decide exit code.
            status = "failed"
            error = str(exc)
            error_traceback = traceback.format_exc()
            artifacts = {
                "backend": backend,
                "status": status,
                "error": error,
                "traceback": error_traceback,
            }
            self.logger.error(
                "train.failed",
                "Train pipeline failed",
                run_id=run_ctx.run_id,
                backend=backend,
                error=error,
                traceback=error_traceback,
            )

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)

        self._save_json(
            run_ctx.run_dir / "metrics.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "error": error,
            },
        )

        self._save_json(
            run_ctx.run_dir / "artifacts.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "backend": backend,
                "status": status,
                "error": error,
                "artifacts": artifacts,
            },
        )

        self.logger.info(
            "train.done",
            "Train pipeline finished",
            run_id=run_ctx.run_id,
            backend=backend,
            status=status,
            elapsed_ms=elapsed_ms,
        )

        return RunResult(
            run_context=run_ctx,
            status=status,
            backend=backend,
            elapsed_ms=elapsed_ms,
            artifacts=artifacts,
            error=error,
        )

    def run_autolabel(self) -> RunResult:
        self._ensure_workdir_layout()
        run_ctx = self._make_run_context(mode="autolabel")

        mode = self.cfg["autolabel"]["mode"]
        runner = self.registry.get_autolabeler(mode)

        self.logger.info(
            "autolabel.start",
            "Autolabel pipeline started",
            run_id=run_ctx.run_id,
            mode=mode,
            workdir=str(self.workdir),
        )

        start = time.perf_counter()
        status = "ok"
        error: str | None = None
        artifacts: dict[str, Any] = {}

        try:
            artifacts = runner(
                self.cfg,
                {
                    "run_id": run_ctx.run_id,
                    "run_dir": str(run_ctx.run_dir),
                    "logger": self.logger,
                },
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            error_traceback = traceback.format_exc()
            artifacts = {
                "mode": mode,
                "status": status,
                "error": error,
                "traceback": error_traceback,
            }
            self.logger.error(
                "autolabel.failed",
                "Autolabel pipeline failed",
                run_id=run_ctx.run_id,
                mode=mode,
                error=error,
                traceback=error_traceback,
            )

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)

        self._save_json(
            run_ctx.run_dir / "metrics.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "error": error,
            },
        )

        self._save_json(
            run_ctx.run_dir / "artifacts.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "autolabel_mode": mode,
                "status": status,
                "error": error,
                "artifacts": artifacts,
            },
        )

        self.logger.info(
            "autolabel.done",
            "Autolabel pipeline finished",
            run_id=run_ctx.run_id,
            mode=mode,
            status=status,
            elapsed_ms=elapsed_ms,
        )

        return RunResult(
            run_context=run_ctx,
            status=status,
            backend=mode,
            elapsed_ms=elapsed_ms,
            artifacts=artifacts,
            error=error,
        )

    def run_deploy_edge(self) -> RunResult:
        self._ensure_workdir_layout()
        run_ctx = self._make_run_context(mode="deploy-edge")

        mode = self.cfg["deploy"]["edge"]["mode"]
        runner = self.registry.get_deployer(mode)

        self.logger.info(
            "deploy.edge.start",
            "Deploy edge pipeline started",
            run_id=run_ctx.run_id,
            mode=mode,
            workdir=str(self.workdir),
        )

        start = time.perf_counter()
        status = "ok"
        error: str | None = None
        artifacts: dict[str, Any] = {}

        try:
            artifacts = runner(
                self.cfg,
                {
                    "run_id": run_ctx.run_id,
                    "run_dir": str(run_ctx.run_dir),
                    "logger": self.logger,
                },
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            error_traceback = traceback.format_exc()
            artifacts = {
                "mode": mode,
                "status": status,
                "error": error,
                "traceback": error_traceback,
            }
            self.logger.error(
                "deploy.edge.failed",
                "Deploy edge pipeline failed",
                run_id=run_ctx.run_id,
                mode=mode,
                error=error,
                traceback=error_traceback,
            )

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)

        self._save_json(
            run_ctx.run_dir / "metrics.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "error": error,
            },
        )

        self._save_json(
            run_ctx.run_dir / "artifacts.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "deploy_mode": mode,
                "status": status,
                "error": error,
                "artifacts": artifacts,
            },
        )

        self.logger.info(
            "deploy.edge.done",
            "Deploy edge pipeline finished",
            run_id=run_ctx.run_id,
            mode=mode,
            status=status,
            elapsed_ms=elapsed_ms,
        )

        return RunResult(
            run_context=run_ctx,
            status=status,
            backend=mode,
            elapsed_ms=elapsed_ms,
            artifacts=artifacts,
            error=error,
        )

    def run_infer(self) -> RunResult:
        return self.run_deploy_edge()

    def run_deploy_remote(self) -> RunResult:
        self._ensure_workdir_layout()
        run_ctx = self._make_run_context(mode="deploy-remote")

        backend = "remote"
        runner = self.registry.get_deployer(backend)

        self.logger.info(
            "deploy.remote.start",
            "Deploy remote pipeline started",
            run_id=run_ctx.run_id,
            backend=backend,
            workdir=str(self.workdir),
        )

        start = time.perf_counter()
        status = "ok"
        error: str | None = None
        artifacts: dict[str, Any] = {}

        try:
            artifacts = runner(
                self.cfg,
                {
                    "run_id": run_ctx.run_id,
                    "run_dir": str(run_ctx.run_dir),
                    "logger": self.logger,
                },
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            error_traceback = traceback.format_exc()
            artifacts = {
                "backend": backend,
                "status": status,
                "error": error,
                "traceback": error_traceback,
            }
            self.logger.error(
                "deploy.remote.failed",
                "Deploy remote pipeline failed",
                run_id=run_ctx.run_id,
                backend=backend,
                error=error,
                traceback=error_traceback,
            )

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)

        self._save_json(
            run_ctx.run_dir / "metrics.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "error": error,
            },
        )

        self._save_json(
            run_ctx.run_dir / "artifacts.json",
            {
                "run_id": run_ctx.run_id,
                "mode": run_ctx.mode,
                "backend": backend,
                "status": status,
                "error": error,
                "artifacts": artifacts,
            },
        )

        self.logger.info(
            "deploy.remote.done",
            "Deploy remote pipeline finished",
            run_id=run_ctx.run_id,
            backend=backend,
            status=status,
            elapsed_ms=elapsed_ms,
        )

        return RunResult(
            run_context=run_ctx,
            status=status,
            backend=backend,
            elapsed_ms=elapsed_ms,
            artifacts=artifacts,
            error=error,
        )
