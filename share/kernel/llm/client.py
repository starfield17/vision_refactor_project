"""Shared OpenAI-compatible LLM request and parsing utilities."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from share.types.detection import Detection
from share.types.errors import DataValidationError


@dataclass(slots=True)
class HTTPCallError(Exception):
    status_code: int | None
    detail: str

    @property
    def retryable(self) -> bool:
        if self.status_code is None:
            return True
        return self.status_code in {408, 409, 425, 429} or self.status_code >= 500


def load_api_key(api_key_env: str) -> str:
    if not api_key_env:
        raise DataValidationError("llm api_key_env must not be empty")
    value = os.getenv(api_key_env)
    if value:
        return value
    # Backward-compatible: accept direct key string for non-env-name values.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        return api_key_env
    if api_key_env.startswith("sk-"):
        return api_key_env
    raise DataValidationError(
        f"env var not found for api_key_env: {api_key_env}. "
        "Either export this env var or fill api_key_env with a direct key."
    )


def build_system_prompt(base_prompt: str, class_names: list[str], min_conf: float) -> str:
    names_text = ", ".join(class_names)
    guide = (
        "Return JSON only (no markdown/code fences). "
        "Schema: {\"detections\":[{\"class_name\":\"...\",\"score\":0.0-1.0,"
        "\"bbox_xyxy\":[x1,y1,x2,y2]}]}. "
        "bbox_xyxy must use pixel coordinates on the original image. "
        f"Only use classes from this list: [{names_text}]. "
        f"Do not output detections with score < {min_conf}."
    )
    base = base_prompt.strip()
    return f"{base}\n\n{guide}" if base else guide


def post_chat_completion(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_sec: float,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request = Request(
        url=url,
        method="POST",
        headers=headers,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
    )
    try:
        with urlopen(request, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPCallError(status_code=exc.code, detail=detail) from exc
    except URLError as exc:
        raise HTTPCallError(status_code=None, detail=str(exc.reason)) from exc
    except TimeoutError as exc:
        raise HTTPCallError(status_code=None, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPCallError(status_code=None, detail=f"invalid json response: {exc}") from exc


def extract_message_content(response_json: dict[str, Any]) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError) as exc:
        raise DataValidationError(f"invalid LLM response format: {exc}") from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                parts.append(str(item["text"]))
        text = "\n".join(parts).strip()
        if text:
            return text
    raise DataValidationError("LLM response message.content is empty")


def _extract_json_text(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text.strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end >= start:
        return candidate[start : end + 1]
    raise DataValidationError("LLM response does not contain a JSON object")


def _resolve_bbox(raw: dict[str, Any], width: int, height: int) -> list[float]:
    bbox_obj = raw.get("bbox_xyxy")
    if bbox_obj is None and "bbox" in raw:
        bbox_obj = raw["bbox"]
    if bbox_obj is None:
        if all(k in raw for k in ("x1", "y1", "x2", "y2")):
            bbox_obj = [raw["x1"], raw["y1"], raw["x2"], raw["y2"]]
        else:
            raise DataValidationError("detection missing bbox_xyxy")

    bbox = [float(v) for v in bbox_obj]
    if len(bbox) != 4:
        raise DataValidationError("bbox_xyxy must contain 4 values")

    # If model returns normalized coordinates, convert to pixels.
    if all(0.0 <= v <= 1.0 for v in bbox):
        bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]

    x1 = max(0.0, min(float(width), bbox[0]))
    y1 = max(0.0, min(float(height), bbox[1]))
    x2 = max(0.0, min(float(width), bbox[2]))
    y2 = max(0.0, min(float(height), bbox[3]))
    if x2 <= x1 or y2 <= y1:
        raise DataValidationError("invalid bbox_xyxy after clipping")
    return [x1, y1, x2, y2]


def _build_detection(
    raw_det: dict[str, Any],
    class_names: list[str],
    class_id_map: dict[str, int],
    width: int,
    height: int,
    min_confidence: float,
) -> Detection | None:
    class_name_raw = raw_det.get("class_name", "")
    class_name = str(class_name_raw).strip()
    class_id_raw = raw_det.get("class_id")

    if class_name:
        if class_name not in class_id_map:
            return None
        class_id = int(class_id_map[class_name])
    else:
        if class_id_raw is None:
            return None
        class_id = int(class_id_raw)
        if class_id < 0 or class_id >= len(class_names):
            return None
        class_name = class_names[class_id]

    score = float(raw_det.get("score", 1.0))
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    if score < min_confidence:
        return None
    if score > 1.0:
        score = 1.0
    if score < 0.0:
        score = 0.0

    bbox_xyxy = _resolve_bbox(raw_det, width=width, height=height)
    det = Detection(
        schema_version=1,
        class_id=class_id,
        class_name=class_name,
        score=score,
        bbox_xyxy=bbox_xyxy,
    )
    det.validate()
    return det


def parse_llm_detection_payload(
    response_text: str,
    class_names: list[str],
    class_id_map: dict[str, int],
    width: int,
    height: int,
    min_confidence: float,
) -> list[Detection]:
    json_text = _extract_json_text(response_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"failed to parse LLM JSON payload: {exc}") from exc

    raw_detections: list[Any]
    if isinstance(payload, dict):
        if isinstance(payload.get("detections"), list):
            raw_detections = payload["detections"]
        elif isinstance(payload.get("objects"), list):
            raw_detections = payload["objects"]
        else:
            raw_detections = []
    elif isinstance(payload, list):
        raw_detections = payload
    else:
        raise DataValidationError("LLM JSON payload must be an object or list")

    detections: list[Detection] = []
    for raw_det in raw_detections:
        if not isinstance(raw_det, dict):
            continue
        try:
            det = _build_detection(
                raw_det=raw_det,
                class_names=class_names,
                class_id_map=class_id_map,
                width=width,
                height=height,
                min_confidence=min_confidence,
            )
            if det is not None:
                detections.append(det)
        except DataValidationError:
            # Skip malformed box from LLM output, keep remaining boxes usable.
            continue
    return detections
