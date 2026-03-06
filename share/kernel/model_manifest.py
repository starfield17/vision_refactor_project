"""Model manifest helpers for training and deploy flows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


def _json_hash(payload: Any) -> str:
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(normalized.encode("utf-8")).hexdigest()


def _derive_supported_modes(backend: str, final_infer_model_path: str, export_ok: bool) -> list[str]:
    supported = ["train"]
    path = Path(final_infer_model_path) if final_infer_model_path else None

    if backend == "yolo" and export_ok and path is not None and path.suffix.lower() == ".onnx":
        supported.extend(["autolabel-model", "deploy-edge-local", "deploy-edge-stream", "deploy-remote"])
    return supported


def build_train_model_manifest(cfg: dict[str, Any], run_ctx: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    backend = str(artifacts.get("backend") or cfg["train"]["backend"])
    weights = dict(artifacts.get("weights") or {})
    export = dict(artifacts.get("export") or {})
    class_names = list(cfg["class_map"]["names"])
    class_id_map = dict(cfg["class_map"]["id_map"])
    final_infer_model_path = str(export.get("final_infer_model_path") or weights.get("model_pt") or "")
    export_ok = bool(export.get("export_ok", False))
    model_id = f"{cfg['workspace']['run_name']}-{backend}"

    return {
        "schema_version": 1,
        "model_id": model_id,
        "backend": backend,
        "run_name": str(cfg["workspace"]["run_name"]),
        "run_id": str(run_ctx["run_id"]),
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_shape": {"img_size": int(cfg["train"]["img_size"])} ,
        "class_map": {
            "names": class_names,
            "id_map": class_id_map,
            "hash": _json_hash({"names": class_names, "id_map": class_id_map}),
        },
        "artifacts": {
            "model_pt": str(weights.get("model_pt") or ""),
            "onnx": str(export.get("onnx_path") or ""),
            "onnx_int8": str(export.get("quantized_onnx_path") or ""),
            "onnx_fp16": str(export.get("fp16_onnx_path") or ""),
            "final_infer_model_path": final_infer_model_path,
            "labels_snapshot": str(weights.get("labels_snapshot") or ""),
        },
        "export": {
            "onnx_enabled": bool(cfg["export"]["onnx"]),
            "export_ok": export_ok,
            "quantize_ok": bool(export.get("quantize_ok", False)),
            "quantize_strategy": str(export.get("quantize_strategy") or ""),
            "messages": list(export.get("messages") or []),
        },
        "deployment_ready": export_ok and Path(final_infer_model_path).suffix.lower() == ".onnx",
        "supported_modes": _derive_supported_modes(backend, final_infer_model_path, export_ok),
        "benchmark_summary": None,
    }


def write_model_manifest(model_dir: Path, manifest: dict[str, Any]) -> Path:
    path = model_dir / "model_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def load_model_manifest(model_path: Path) -> dict[str, Any] | None:
    manifest_path = model_path.parent / "model_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def resolve_model_identity(
    model_path: Path,
    *,
    default_backend: str,
    default_model_id: str,
) -> dict[str, str]:
    manifest = load_model_manifest(model_path)
    manifest_path = model_path.parent / "model_manifest.json"

    backend = default_backend
    model_id = default_model_id
    if isinstance(manifest, dict):
        backend = str(manifest.get("backend") or backend)
        model_id = str(manifest.get("model_id") or model_id)

    return {
        "backend": backend,
        "model_id": model_id,
        "artifact_path": str(model_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
    }
