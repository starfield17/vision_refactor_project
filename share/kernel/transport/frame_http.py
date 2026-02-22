"""HTTP helpers for JPEG frame transport between edge and remote."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from share.types.errors import DataValidationError, TransportError


def encode_jpeg_base64(frame_bgr: Any, jpeg_quality: int) -> str:
    try:
        import cv2
    except Exception as exc:
        raise DataValidationError(f"opencv is required for frame encoding: {exc}") from exc

    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise DataValidationError("failed to encode frame as jpeg")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_jpeg_base64(encoded: str) -> Any:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise DataValidationError(f"opencv/numpy is required for frame decoding: {exc}") from exc

    try:
        frame_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise DataValidationError(f"invalid base64 image payload: {exc}") from exc

    np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise DataValidationError("failed to decode jpeg payload")
    return frame


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    timeout_sec: float,
    api_key: str,
) -> dict[str, Any]:
    request_payload = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    request = Request(url=endpoint, data=request_payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            status_code = int(response.status)
            body = response.read().decode("utf-8", errors="replace")
            if status_code != 200:
                raise TransportError(
                    f"remote call failed: status_code={status_code}, body={body[:400]}"
                )
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise TransportError(
            f"remote call failed: status_code={exc.code}, body={body[:400]}"
        ) from exc
    except URLError as exc:
        raise TransportError(f"remote call failed: {exc.reason}") from exc

    try:
        payload_obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TransportError(f"remote call returned non-json response: {body[:200]}") from exc
    if not isinstance(payload_obj, dict):
        raise TransportError("remote call returned invalid payload type")
    return payload_obj
