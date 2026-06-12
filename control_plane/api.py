"""Control plane API service."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.application.api_common import (
    get_json,
    json_response,
    post_json,
    request_json,
    require_bearer_token,
    resolve_api_token,
    validate_service_security,
)
from common.types.errors import ConfigError, TransportError
from control_plane.config.schema import load_config
from control_plane.store import ControlPlaneStore


FINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled", "interrupted"}
JOB_ROLE_MAP = {
    "train": "train_worker",
    "autolabel": "autolabel_worker",
    "edge_run": "edge",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision control plane API")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def _read_model_manifests(registry_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not registry_dir.exists():
        return manifests
    for path in sorted(registry_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("manifest_path", str(path))
            manifests.append(payload)
    return manifests


def _node_dispatch_token(node: dict[str, Any]) -> str:
    payload = node.get("payload")
    if isinstance(payload, dict):
        return str(payload.get("dispatch_token") or "")
    return ""


def _node_age_sec(node: dict[str, Any]) -> float:
    try:
        seen = datetime.fromisoformat(str(node["last_seen_utc"]))
    except ValueError:
        return float("inf")
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return (datetime.now(tz=timezone.utc) - seen).total_seconds()


def _apply_node_ttl(node: dict[str, Any], offline_ttl_sec: int) -> dict[str, Any]:
    updated = dict(node)
    if offline_ttl_sec > 0 and _node_age_sec(updated) > offline_ttl_sec:
        updated["status"] = "offline"
    return updated


def _list_nodes_with_ttl(store: ControlPlaneStore, offline_ttl_sec: int) -> list[dict[str, Any]]:
    return [_apply_node_ttl(node, offline_ttl_sec) for node in store.list_nodes()]


def _choose_node(
    store: ControlPlaneStore,
    target_role: str,
    node_id: str = "",
    offline_ttl_sec: int = 45,
) -> dict[str, Any] | None:
    nodes = _list_nodes_with_ttl(store, offline_ttl_sec)
    if node_id:
        return next(
            (
                node
                for node in nodes
                if node["node_id"] == node_id
                and node["role"] == target_role
                and node["status"] == "online"
                and node["endpoint"]
            ),
            None,
        )
    return next(
        (
            node
            for node in nodes
            if node["role"] == target_role and node["status"] == "online" and node["endpoint"]
        ),
        None,
    )


def _extract_upstream_job(response: dict[str, Any]) -> dict[str, Any]:
    job = response.get("job")
    if not isinstance(job, dict):
        raise TransportError("upstream response did not include job")
    return dict(job)


def create_app(cfg: dict[str, Any]):
    try:
        from fastapi import FastAPI, Header, Query
    except Exception as exc:
        raise ConfigError(f"fastapi is required for control plane API: {exc}") from exc

    token = resolve_api_token(cfg["server"])
    offline_ttl_sec = int(cfg.get("nodes", {}).get("offline_ttl_sec", 45))
    store = ControlPlaneStore(Path(cfg["storage"]["db_path"]))
    app = FastAPI(title="Vision Control Plane", version="1.0.0")

    def auth(authorization: str | None):
        return require_bearer_token(authorization, token)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "control_plane", "db_path": str(store.db_path)}

    @app.get("/api/v1/config")
    def get_config(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "config": cfg}

    @app.get("/api/v1/nodes")
    def list_nodes(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "nodes": _list_nodes_with_ttl(store, offline_ttl_sec)}

    @app.get("/api/v1/nodes/{node_id}")
    def get_node(node_id: str, authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        node = store.get_node(node_id)
        if node is None:
            return json_response({"ok": False, "error": "not_found"}, 404)
        node = _apply_node_ttl(node, offline_ttl_sec)
        return {"ok": True, "node": node}

    @app.post("/api/v1/nodes/heartbeat")
    def heartbeat(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        try:
            node = store.upsert_node(payload)
        except ValueError as exc:
            return json_response({"ok": False, "error": "validation_failed", "detail": str(exc)}, 400)
        return {"ok": True, "node": node}

    @app.get("/api/v1/models")
    def models(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "models": _read_model_manifests(Path(cfg["storage"]["model_registry"]))}

    @app.get("/api/v1/workers/status")
    def workers_status(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        statuses: list[dict[str, Any]] = []
        for node in _list_nodes_with_ttl(store, offline_ttl_sec):
            if node["status"] != "online":
                statuses.append({"node": node, "ok": False, "error": "offline"})
                continue
            if not node["endpoint"]:
                statuses.append({"node": node, "ok": False, "error": "missing_endpoint"})
                continue
            try:
                status = get_json(
                    node["endpoint"],
                    "/api/v1/status",
                    token=_node_dispatch_token(node),
                    timeout_sec=3.0,
                )
                statuses.append({"node": node, "ok": True, "status": status})
            except TransportError as exc:
                statuses.append({"node": node, "ok": False, "error": str(exc)})
        return {"ok": True, "workers": statuses}

    def refresh_job(job: dict[str, Any]) -> dict[str, Any]:
        if not job["upstream_endpoint"] or not job["upstream_job_id"]:
            return job
        if job["status"] in FINAL_JOB_STATUSES:
            return job
        node = store.get_node(job["target_node_id"]) if job["target_node_id"] else None
        token = _node_dispatch_token(node) if node else ""
        try:
            response = get_json(
                job["upstream_endpoint"],
                f"/api/v1/jobs/{job['upstream_job_id']}",
                token=token,
                timeout_sec=5.0,
            )
            upstream_job = _extract_upstream_job(response)
        except TransportError as exc:
            updated = store.update_job_status(
                job["job_id"],
                status="upstream_unreachable",
                error=str(exc),
            )
            return updated or job
        status = str(upstream_job.get("status") or job["status"])
        updated = store.update_job_status(
            job["job_id"],
            status=status,
            result=upstream_job.get("result") if isinstance(upstream_job.get("result"), dict) else None,
            error=str(upstream_job.get("error") or ""),
            log_path=str(upstream_job.get("log_path") or ""),
            finished=status in FINAL_JOB_STATUSES,
        )
        return updated or job

    @app.post("/api/v1/jobs")
    def create_job(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        kind = str(payload.get("kind") or "")
        target_role = str(payload.get("target_role") or JOB_ROLE_MAP.get(kind, ""))
        target_node_id = str(payload.get("target_node_id") or "")
        if not kind:
            return json_response({"ok": False, "error": "validation_failed", "detail": "kind is required"}, 400)
        if not target_role:
            return json_response(
                {
                    "ok": False,
                    "error": "validation_failed",
                    "detail": f"no target role mapping for kind={kind}",
                },
                400,
            )
        node = _choose_node(
            store,
            target_role=target_role,
            node_id=target_node_id,
            offline_ttl_sec=offline_ttl_sec,
        )
        if node is None:
            return json_response(
                {
                    "ok": False,
                    "error": "no_compatible_worker",
                    "detail": f"no online node for role={target_role}",
                },
                503,
            )
        control_job = store.create_job(
            kind=kind,
            target_role=target_role,
            target_node_id=node["node_id"],
            payload=payload,
        )
        upstream_payload = dict(payload.get("payload") if isinstance(payload.get("payload"), dict) else payload)
        upstream_payload.pop("kind", None)
        upstream_payload.pop("target_role", None)
        upstream_payload.pop("target_node_id", None)
        try:
            response = post_json(
                node["endpoint"],
                "/api/v1/jobs",
                payload=upstream_payload,
                token=_node_dispatch_token(node),
                timeout_sec=10.0,
            )
            upstream_job = _extract_upstream_job(response)
        except TransportError as exc:
            job = store.update_job_status(
                control_job["job_id"],
                status="failed",
                error=str(exc),
                finished=True,
            )
            return json_response(
                {"ok": False, "error": "upstream_failed", "detail": str(exc), "job": job},
                502,
            )
        job = store.attach_upstream(
            control_job["job_id"],
            target_node_id=node["node_id"],
            upstream_endpoint=node["endpoint"],
            upstream_job_id=str(upstream_job["job_id"]),
            status=str(upstream_job.get("status") or "running"),
            result=upstream_job.get("result") if isinstance(upstream_job.get("result"), dict) else {},
            error=str(upstream_job.get("error") or ""),
            log_path=str(upstream_job.get("log_path") or ""),
        )
        return {"ok": True, "job": job, "upstream_job": upstream_job, "node": node}

    @app.get("/api/v1/jobs")
    def jobs(
        limit: int = Query(default=100),
        kind: str = Query(default=""),
        status: str = Query(default=""),
        refresh: bool = Query(default=False),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        jobs_payload = store.list_jobs(limit=limit, kind=kind, status=status)
        if refresh:
            jobs_payload = [refresh_job(job) for job in jobs_payload]
        return {"ok": True, "jobs": jobs_payload}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(
        job_id: str,
        refresh: bool = Query(default=True),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        job = store.get_job(job_id)
        if job is None:
            return json_response({"ok": False, "error": "not_found"}, 404)
        if refresh:
            job = refresh_job(job)
        return {"ok": True, "job": job}

    @app.get("/api/v1/jobs/{job_id}/logs")
    def get_job_logs(
        job_id: str,
        tail: int = Query(default=200),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        job = store.get_job(job_id)
        if job is None:
            return json_response({"ok": False, "error": "not_found"}, 404)
        if not job["upstream_endpoint"] or not job["upstream_job_id"]:
            return {"ok": True, "job_id": job_id, "text": ""}
        node = store.get_node(job["target_node_id"]) if job["target_node_id"] else None
        try:
            return get_json(
                job["upstream_endpoint"],
                f"/api/v1/jobs/{job['upstream_job_id']}/logs",
                query={"tail": tail},
                token=_node_dispatch_token(node) if node else "",
                timeout_sec=5.0,
            )
        except TransportError as exc:
            return json_response({"ok": False, "error": "upstream_failed", "detail": str(exc)}, 502)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        job = store.get_job(job_id)
        if job is None:
            return json_response({"ok": False, "error": "not_found"}, 404)
        if not job["upstream_endpoint"] or not job["upstream_job_id"]:
            updated = store.update_job_status(
                job_id,
                status="cancelled",
                error="Cancelled before dispatch",
                finished=True,
            )
            return {"ok": True, "job": updated}
        node = store.get_node(job["target_node_id"]) if job["target_node_id"] else None
        try:
            upstream = request_json(
                method="POST",
                base_url=job["upstream_endpoint"],
                path=f"/api/v1/jobs/{job['upstream_job_id']}/cancel",
                payload={},
                token=_node_dispatch_token(node) if node else "",
                timeout_sec=5.0,
            )
        except TransportError as exc:
            return json_response({"ok": False, "error": "upstream_failed", "detail": str(exc)}, 502)
        updated = refresh_job(job)
        return {"ok": True, "job": updated, "upstream": upstream}

    return app


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(
            Path(args.config).resolve(),
            overrides=args.set,
            workdir_override=str(Path(args.workdir).resolve()) if args.workdir else None,
        )
        validate_service_security("control_plane", cfg["server"])
        app = create_app(cfg)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except Exception as exc:
        print(f"[RUNTIME ERROR] uvicorn is required: {exc}", file=sys.stderr)
        return 3

    uvicorn.run(app, host=str(cfg["server"]["host"]), port=int(cfg["server"]["port"]), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
