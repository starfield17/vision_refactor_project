"""HTTP client helpers for service-backed frontends."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from share.application.api_common import get_json, make_service_url, post_json, resolve_api_token
from share.config.config_loader import load_config
from share.types.errors import ConfigError, TransportError


FINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled", "interrupted"}


def load_service_connection(
    config_path: Path,
    service_name: str,
    workdir_override: str | None = None,
    overrides: list[str] | None = None,
    api_url_override: str | None = None,
    api_token_override: str | None = None,
) -> tuple[str, str]:
    cfg = load_config(
        config_path=config_path,
        overrides=overrides or [],
        workdir_override=workdir_override,
    )
    if service_name not in cfg["services"]:
        raise ConfigError(f"unknown service '{service_name}'")
    service_cfg = cfg["services"][service_name]
    api_url = api_url_override or make_service_url(service_cfg)
    token = api_token_override if api_token_override is not None else resolve_api_token(service_cfg)
    return api_url, token


def submit_job(
    api_url: str,
    token: str,
    path: str,
    payload: dict[str, Any],
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    response = post_json(
        base_url=api_url,
        path=path,
        payload=payload,
        token=token,
        timeout_sec=timeout_sec,
    )
    if response.get("ok") is not True:
        raise TransportError(str(response))
    job = response.get("job")
    if not isinstance(job, dict):
        raise TransportError("API response did not include job")
    return job


def get_job(api_url: str, token: str, job_id: str, timeout_sec: float = 10.0) -> dict[str, Any]:
    response = get_json(
        base_url=api_url,
        path=f"/api/v1/jobs/{job_id}",
        token=token,
        timeout_sec=timeout_sec,
    )
    if response.get("ok") is not True or not isinstance(response.get("job"), dict):
        raise TransportError(str(response))
    return dict(response["job"])


def wait_for_job(
    api_url: str,
    token: str,
    job_id: str,
    poll_sec: float = 1.0,
    timeout_sec: float = 0.0,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        job = get_job(api_url=api_url, token=token, job_id=job_id)
        if str(job["status"]) in FINAL_JOB_STATUSES:
            return job
        if timeout_sec > 0 and time.monotonic() - started > timeout_sec:
            raise TransportError(f"timed out waiting for job_id={job_id}")
        time.sleep(max(0.1, float(poll_sec)))


def patch_config(
    api_url: str,
    token: str,
    area: str,
    overrides: list[str],
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    return patch_config_request(
        api_url=api_url,
        token=token,
        payload={"area": area, "overrides": overrides},
        timeout_sec=timeout_sec,
    )


def patch_config_request(
    api_url: str,
    token: str,
    payload: dict[str, Any],
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    from share.application.api_common import request_json

    return request_json(
        method="PATCH",
        base_url=api_url,
        path="/api/v1/config",
        payload=payload,
        token=token,
        timeout_sec=timeout_sec,
    )
