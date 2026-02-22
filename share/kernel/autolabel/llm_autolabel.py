"""LLM-based autolabel pipeline."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from share.kernel.llm import client as llm_client
from share.kernel.media.preview import save_side_by_side_preview
from share.types.detection import Detection
from share.types.errors import DataValidationError
from share.types.label import LabelRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class _LLMResponse:
    detections: list[Detection]
    raw_text: str


@dataclass(slots=True)
class _HTTPCallError(Exception):
    status_code: int | None
    detail: str

    @property
    def retryable(self) -> bool:
        if self.status_code is None:
            return True
        return self.status_code in {408, 409, 425, 429} or self.status_code >= 500


def _list_images(images_dir: Path, max_images: int) -> list[Path]:
    if not images_dir.exists():
        raise DataValidationError(f"unlabeled images directory not found: {images_dir}")

    images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise DataValidationError(f"no images found under: {images_dir}")
    return images


def _load_api_key(api_key_env: str) -> str:
    if not api_key_env:
        raise DataValidationError("autolabel.llm.api_key_env must not be empty in llm mode")
    value = os.getenv(api_key_env)
    if value:
        return value
    # Backward-compatible: if user filled the key directly into api_key_env, accept it.
    # Env var names are typically [A-Za-z_][A-Za-z0-9_]*, so other forms are treated as raw key.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        return api_key_env
    if api_key_env.startswith("sk-"):
        return api_key_env
    raise DataValidationError(
        f"env var not found for autolabel.llm.api_key_env: {api_key_env}. "
        "Either export this env var or fill api_key_env with a direct key."
    )


def _build_system_prompt(base_prompt: str, class_names: list[str], min_conf: float) -> str:
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


def _guess_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _encode_image_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{_guess_mime_type(image_path)};base64,{encoded}"


def _post_chat_completion(
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
        raise _HTTPCallError(status_code=exc.code, detail=detail) from exc
    except URLError as exc:
        raise _HTTPCallError(status_code=None, detail=str(exc.reason)) from exc
    except TimeoutError as exc:
        raise _HTTPCallError(status_code=None, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise _HTTPCallError(status_code=None, detail=f"invalid json response: {exc}") from exc


def _extract_message_content(response_json: dict[str, Any]) -> str:
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


def _parse_llm_detection_payload(
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


# Keep autolabel and deploy llm behavior aligned by rebinding to shared helpers.
_HTTPCallError = llm_client.HTTPCallError
_load_api_key = llm_client.load_api_key
_build_system_prompt = llm_client.build_system_prompt
_post_chat_completion = llm_client.post_chat_completion
_extract_message_content = llm_client.extract_message_content
_parse_llm_detection_payload = llm_client.parse_llm_detection_payload


def _call_llm_for_image(
    image_path: Path,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    class_names: list[str],
    class_id_map: dict[str, int],
    confidence: float,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
) -> _LLMResponse:
    try:
        import cv2
    except Exception as exc:
        raise DataValidationError(f"opencv is required for llm autolabel image reading: {exc}") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise DataValidationError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    data_url = _encode_image_data_url(image_path)
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this image and output JSON only."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response_json = _post_chat_completion(
                url=endpoint,
                payload=payload,
                api_key=api_key,
                timeout_sec=timeout_sec,
            )
            text = _extract_message_content(response_json)
            detections = _parse_llm_detection_payload(
                response_text=text,
                class_names=class_names,
                class_id_map=class_id_map,
                width=width,
                height=height,
                min_confidence=confidence,
            )
            return _LLMResponse(detections=detections, raw_text=text)
        except _HTTPCallError as exc:
            last_error = f"http_error(status={exc.status_code}): {exc.detail[:300]}"
            if attempt < max_retries and exc.retryable:
                time.sleep(retry_backoff_sec * (2**attempt))
                continue
            break
        except DataValidationError as exc:
            # If payload is malformed, retry may still help when provider returns transient junk.
            last_error = f"response_parse_error: {exc}"
            if attempt < max_retries:
                time.sleep(retry_backoff_sec * (2**attempt))
                continue
            break

    raise DataValidationError(
        f"LLM request failed for image={image_path.name} after retries, detail={last_error}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_label(path: Path) -> LabelRecord:
    return LabelRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _dedupe_detections(detections: list[Detection]) -> list[Detection]:
    seen: set[tuple[int, float, float, float, float, float]] = set()
    merged: list[Detection] = []
    for det in detections:
        key = (
            det.class_id,
            round(det.score, 4),
            round(det.bbox_xyxy[0], 2),
            round(det.bbox_xyxy[1], 2),
            round(det.bbox_xyxy[2], 2),
            round(det.bbox_xyxy[3], 2),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(det)
    return merged


def _resolve_label_record(
    label_path: Path,
    incoming: LabelRecord,
    on_conflict: str,
) -> tuple[LabelRecord | None, str]:
    if not label_path.exists():
        return incoming, "created"

    if on_conflict == "skip":
        return None, "skipped"
    if on_conflict == "overwrite":
        return incoming, "overwritten"
    if on_conflict == "merge":
        existing = _load_label(label_path)
        merged = LabelRecord(
            schema_version=1,
            image_path=incoming.image_path,
            source=incoming.source,
            detections=_dedupe_detections([*existing.detections, *incoming.detections]),
        )
        return merged, "merged"
    raise DataValidationError(f"unsupported on_conflict strategy: {on_conflict}")


def _save_visualization(image_path: Path, label: LabelRecord, save_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False

    original = cv2.imread(str(image_path))
    if original is None:
        return False
    annotated = original.copy()

    for det in label.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 180, 16), 2)
        cv2.putText(
            annotated,
            f"{det.class_name}:{det.score:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (16, 180, 16),
            1,
            cv2.LINE_AA,
        )

    return save_side_by_side_preview(
        original_bgr=original,
        annotated_bgr=annotated,
        save_path=save_path,
        left_title="Original",
        right_title="LLM Labels",
    )


def run_llm_autolabel(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    class_names = list(cfg["class_map"]["names"])
    class_id_map = {str(k): int(v) for k, v in dict(cfg["class_map"]["id_map"]).items()}

    autolabel_cfg = cfg["autolabel"]
    llm_cfg = autolabel_cfg["llm"]
    images_dir = Path(cfg["data"]["unlabeled_dir"])
    labeled_dir = Path(cfg["data"]["labeled_dir"])
    max_images = int(llm_cfg["max_images"])
    image_paths = _list_images(images_dir=images_dir, max_images=max_images)

    confidence = float(autolabel_cfg["confidence"])
    on_conflict = str(autolabel_cfg["on_conflict"])
    visualize = bool(autolabel_cfg["visualize"])
    base_url = str(llm_cfg["base_url"])
    model = str(llm_cfg["model"])
    api_key = _load_api_key(str(llm_cfg["api_key_env"]))
    timeout_sec = float(llm_cfg["timeout_sec"])
    max_retries = int(llm_cfg["max_retries"])
    retry_backoff_sec = float(llm_cfg["retry_backoff_sec"])
    qps_limit = float(llm_cfg["qps_limit"])
    prompt = _build_system_prompt(
        base_prompt=str(llm_cfg["prompt"]),
        class_names=class_names,
        min_conf=confidence,
    )

    run_id = str(run_ctx["run_id"])
    run_outputs_dir = Path(cfg["workspace"]["root"]) / "outputs" / run_id
    annotated_dir = run_outputs_dir / "annotated_frames"
    logger = run_ctx["logger"]

    created = 0
    overwritten = 0
    merged = 0
    skipped = 0
    viz_saved = 0
    detections_total = 0
    llm_failures = 0
    label_files: list[str] = []
    qps_interval = 1.0 / qps_limit
    next_call_ts = 0.0

    for image_path in image_paths:
        now = time.perf_counter()
        if now < next_call_ts:
            time.sleep(next_call_ts - now)

        try:
            llm_response = _call_llm_for_image(
                image_path=image_path,
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                class_names=class_names,
                class_id_map=class_id_map,
                confidence=confidence,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
            )
            next_call_ts = time.perf_counter() + qps_interval
        except DataValidationError as exc:
            llm_failures += 1
            logger.warn(
                "autolabel.llm.request.failed",
                "LLM autolabel failed for image",
                run_id=run_id,
                image=str(image_path),
                error=str(exc),
            )
            continue

        label = LabelRecord(
            schema_version=1,
            image_path=str(image_path.resolve()),
            source="autolabel:llm",
            detections=llm_response.detections,
        )
        label.validate()
        detections_total += len(label.detections)

        label_path = labeled_dir / f"{image_path.stem}.json"
        resolved, action = _resolve_label_record(label_path=label_path, incoming=label, on_conflict=on_conflict)
        if resolved is None:
            skipped += 1
            continue

        _write_json(label_path, resolved.to_dict())
        label_files.append(str(label_path))

        if action == "created":
            created += 1
        elif action == "overwritten":
            overwritten += 1
        elif action == "merged":
            merged += 1

        if visualize and _save_visualization(image_path=image_path, label=resolved, save_path=annotated_dir / image_path.name):
            viz_saved += 1

    stats = {
        "images_total": len(image_paths),
        "labels_created": created,
        "labels_overwritten": overwritten,
        "labels_merged": merged,
        "labels_skipped": skipped,
        "detections_total": detections_total,
        "visualizations_saved": viz_saved,
        "llm_failures": llm_failures,
    }

    return {
        "mode": "llm",
        "input_images_dir": str(images_dir),
        "output_labels_dir": str(labeled_dir),
        "run_outputs_dir": str(run_outputs_dir),
        "stats": stats,
        "label_files": label_files,
        "config": {
            "base_url": base_url,
            "model": model,
            "confidence": confidence,
            "on_conflict": on_conflict,
            "visualize": visualize,
            "max_images": max_images,
            "timeout_sec": timeout_sec,
            "max_retries": max_retries,
            "retry_backoff_sec": retry_backoff_sec,
            "qps_limit": qps_limit,
        },
    }
