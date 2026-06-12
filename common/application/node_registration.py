"""Control-plane node registration helpers."""

from __future__ import annotations

import threading
import time
from typing import Any

from common.application.api_common import post_json, resolve_api_token
from common.types.errors import TransportError


def build_service_endpoint(role_cfg: dict[str, Any], server_cfg: dict[str, Any]) -> str:
    node_cfg = role_cfg.get("node", {})
    if isinstance(node_cfg, dict) and node_cfg.get("endpoint"):
        return str(node_cfg["endpoint"])
    if server_cfg.get("advertise_url"):
        return str(server_cfg["advertise_url"])
    host = str(server_cfg["host"])
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{int(server_cfg['port'])}"


def build_node_payload(
    role_cfg: dict[str, Any],
    *,
    role: str,
    service: str,
    endpoint: str,
    dispatch_token: str = "",
    status: str = "online",
) -> dict[str, Any]:
    node_cfg = role_cfg.get("node", {})
    node_id = ""
    if isinstance(node_cfg, dict):
        node_id = str(node_cfg.get("id") or "")
    if not node_id:
        node_id = role
    payload = {
        "node_id": node_id,
        "role": role,
        "service": service,
        "status": status,
        "endpoint": endpoint,
        "version": "1",
        "capabilities": role_cfg.get("capabilities", {}),
    }
    if dispatch_token:
        payload["dispatch_token"] = dispatch_token
    return payload


def register_with_control_plane(
    role_cfg: dict[str, Any],
    *,
    role: str,
    service: str,
    server_cfg: dict[str, Any],
    timeout_sec: float = 2.0,
) -> dict[str, Any]:
    control_plane = role_cfg.get("control_plane", {})
    if not isinstance(control_plane, dict) or not control_plane.get("url"):
        return {"ok": False, "skipped": True, "reason": "control_plane.url is empty"}
    endpoint = build_service_endpoint(role_cfg, server_cfg)
    payload = build_node_payload(
        role_cfg,
        role=role,
        service=service,
        endpoint=endpoint,
        dispatch_token=resolve_api_token(server_cfg),
    )
    try:
        return post_json(
            base_url=str(control_plane["url"]),
            path="/api/v1/nodes/heartbeat",
            payload=payload,
            token=resolve_api_token(control_plane),
            timeout_sec=timeout_sec,
        )
    except TransportError as exc:
        return {"ok": False, "error": "control_plane_unreachable", "detail": str(exc)}


def start_control_plane_heartbeat(
    role_cfg: dict[str, Any],
    *,
    role: str,
    service: str,
    server_cfg: dict[str, Any],
    default_interval_sec: int = 15,
) -> threading.Thread | None:
    control_plane = role_cfg.get("control_plane", {})
    if not isinstance(control_plane, dict) or not control_plane.get("url"):
        return None
    interval = int(control_plane.get("heartbeat_interval_sec", default_interval_sec))
    if interval <= 0:
        register_with_control_plane(
            role_cfg,
            role=role,
            service=service,
            server_cfg=server_cfg,
        )
        return None

    def _loop() -> None:
        while True:
            register_with_control_plane(
                role_cfg,
                role=role,
                service=service,
                server_cfg=server_cfg,
            )
            time.sleep(interval)

    thread = threading.Thread(
        target=_loop,
        name=f"{service}-control-plane-heartbeat",
        daemon=True,
    )
    thread.start()
    return thread
