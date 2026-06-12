"""Shared helpers for service APIs and API clients."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common.types.errors import ConfigError, TransportError


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def resolve_api_token(service_cfg: dict[str, Any]) -> str:
    direct = str(service_cfg.get("api_token", ""))
    if direct:
        return direct
    env_name = str(service_cfg.get("api_token_env_name", ""))
    if env_name:
        return os.environ.get(env_name, "")
    return ""


def validate_service_security(service_name: str, service_cfg: dict[str, Any]) -> None:
    host = str(service_cfg["host"])
    token = resolve_api_token(service_cfg)
    if host not in LOOPBACK_HOSTS and not token:
        raise ConfigError(
            f"{service_name} binds to {host}; api_token or api_token_env_name is required"
        )


def make_service_url(service_cfg: dict[str, Any]) -> str:
    return f"http://{service_cfg['host']}:{int(service_cfg['port'])}"


def read_tail(path: Path | str, max_lines: int = 200) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, int(max_lines)) :])


def json_response(payload: dict[str, Any], status_code: int = 200):
    try:
        from fastapi.responses import JSONResponse
    except Exception as exc:
        raise ConfigError(f"fastapi is required for JSON responses: {exc}") from exc
    return JSONResponse(status_code=status_code, content=payload)


def require_bearer_token(header_value: str | None, expected_token: str):
    if not expected_token:
        return None
    expected = f"Bearer {expected_token}"
    if header_value != expected_token and header_value != expected:
        return json_response({"ok": False, "error": "unauthorized"}, status_code=401)
    return None


def require_api_key_or_bearer(
    x_api_key: str | None,
    authorization: str | None,
    expected_token: str,
):
    if not expected_token:
        return None
    expected_bearer = f"Bearer {expected_token}"
    if (
        x_api_key == expected_token
        or authorization == expected_token
        or authorization == expected_bearer
    ):
        return None
    return json_response({"ok": False, "error": "unauthorized"}, status_code=401)


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    token: str = "",
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    return request_json(
        method="POST",
        base_url=base_url,
        path=path,
        payload=payload,
        token=token,
        timeout_sec=timeout_sec,
    )


def get_json(
    base_url: str,
    path: str,
    query: dict[str, Any] | None = None,
    token: str = "",
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    suffix = path
    if query:
        clean = {k: v for k, v in query.items() if v is not None}
        if clean:
            suffix = f"{path}?{urlencode(clean)}"
    return request_json(
        method="GET",
        base_url=base_url,
        path=suffix,
        payload=None,
        token=token,
        timeout_sec=timeout_sec,
    )


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None,
    token: str = "",
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    data = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url=url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = int(resp.status)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise TransportError(f"API request failed: status_code={exc.code}, body={body}") from exc
    except URLError as exc:
        raise TransportError(f"API request failed: {exc.reason}") from exc

    try:
        decoded = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise TransportError(f"API returned non-json response: {body[:200]}") from exc
    if not isinstance(decoded, dict):
        raise TransportError("API returned invalid payload type")
    if status >= 400:
        raise TransportError(f"API request failed: status_code={status}, body={body}")
    return decoded
