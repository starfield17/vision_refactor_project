"""ONNX export and quantization helpers for train pipeline."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from share.kernel.utils.logging import StructuredLogger


@dataclass(slots=True)
class ExportArtifacts:
    onnx_path: str | None
    quantized_onnx_path: str | None
    fp16_onnx_path: str | None
    final_infer_model_path: str
    export_ok: bool
    quantize_ok: bool
    quantize_strategy: str
    messages: list[str]


def _try_import(module_name: str) -> tuple[bool, str | None]:
    try:
        __import__(module_name)
        return True, None
    except Exception as exc:  # pragma: no cover - import failures are environment-dependent
        return False, f"{module_name} unavailable: {exc}"


def export_yolo_to_onnx(
    trained_pt: Path,
    output_onnx_path: Path,
    opset: int,
    device: str,
    logger: StructuredLogger,
) -> tuple[bool, str | None, list[str]]:
    messages: list[str] = []

    ok_onnx, reason_onnx = _try_import("onnx")
    ok_onnxscript, reason_onnxscript = _try_import("onnxscript")
    if not ok_onnx or not ok_onnxscript:
        if reason_onnx:
            messages.append(reason_onnx)
        if reason_onnxscript:
            messages.append(reason_onnxscript)
        return False, None, messages

    try:
        from ultralytics import YOLO

        model = YOLO(str(trained_pt))
        exported_path = model.export(format="onnx", opset=opset, device=device)
        exported = Path(exported_path)
        output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != output_onnx_path.resolve():
            shutil.copyfile(exported, output_onnx_path)
        messages.append(f"onnx export succeeded: {output_onnx_path}")
        logger.info("train.export.onnx.ok", "ONNX export succeeded", onnx_path=str(output_onnx_path))
        return True, str(output_onnx_path), messages
    except Exception as exc:
        msg = f"onnx export failed: {exc}"
        messages.append(msg)
        logger.warn("train.export.onnx.failed", "ONNX export failed", error=str(exc))
        return False, None, messages


def quantize_with_fallback(
    fp32_onnx_path: Path,
    model_dir: Path,
    quantize_enabled: bool,
    quantize_mode: str,
    logger: StructuredLogger,
) -> tuple[bool, str | None, str | None, str, list[str]]:
    messages: list[str] = []

    if not quantize_enabled:
        messages.append("quantization disabled by config")
        return False, None, None, "disabled", messages

    int8_path = model_dir / "model-int8.onnx"
    fp16_path = model_dir / "model-fp16.onnx"

    if quantize_mode != "dynamic":
        messages.append(
            f"quantize_mode={quantize_mode} is not implemented, expected dynamic"
        )
        logger.warn(
            "train.export.quant.mode.unsupported",
            "Unsupported quantize_mode for current implementation",
            quantize_mode=quantize_mode,
        )
        return False, None, None, "unsupported-mode", messages

    # Strategy 1: ONNXRuntime dynamic/static quantization.
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(fp32_onnx_path),
            model_output=str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        logger.info("train.export.quant.ok", "ONNX dynamic quantization succeeded", int8_path=str(int8_path))
        messages.append(f"int8 quantization succeeded: {int8_path}")
        return True, str(int8_path), None, "onnxruntime-dynamic", messages
    except Exception as exc:
        msg = f"onnxruntime quantization failed: {exc}"
        logger.warn("train.export.quant.failed", "ONNX quantization failed", error=str(exc))
        messages.append(msg)

    # Strategy 2: fp16 fallback if possible.
    try:
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(str(fp32_onnx_path))
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, str(fp16_path))
        logger.info("train.export.fp16.ok", "FP16 fallback succeeded", fp16_path=str(fp16_path))
        messages.append(f"fp16 fallback succeeded: {fp16_path}")
        return False, None, str(fp16_path), "fp16-fallback", messages
    except Exception as exc:
        msg = f"fp16 fallback unavailable: {exc}"
        logger.warn("train.export.fp16.failed", "FP16 fallback failed", error=str(exc))
        messages.append(msg)

    return False, None, None, "fp32-only", messages


def build_export_artifacts(
    trained_pt: Path,
    model_dir: Path,
    cfg_export: dict[str, Any],
    device: str,
    logger: StructuredLogger,
) -> ExportArtifacts:
    onnx_target = model_dir / "model.onnx"

    if not bool(cfg_export.get("onnx", True)):
        return ExportArtifacts(
            onnx_path=None,
            quantized_onnx_path=None,
            fp16_onnx_path=None,
            final_infer_model_path=str(trained_pt),
            export_ok=False,
            quantize_ok=False,
            quantize_strategy="disabled",
            messages=["onnx export disabled by config"],
        )

    export_ok, onnx_path, export_msgs = export_yolo_to_onnx(
        trained_pt=trained_pt,
        output_onnx_path=onnx_target,
        opset=int(cfg_export.get("opset", 17)),
        device=device,
        logger=logger,
    )

    quant_ok = False
    quant_path: str | None = None
    fp16_path: str | None = None
    quant_strategy = "skipped"
    quant_msgs: list[str] = []

    if export_ok and onnx_path is not None:
        quant_ok, quant_path, fp16_path, quant_strategy, quant_msgs = quantize_with_fallback(
            fp32_onnx_path=Path(onnx_path),
            model_dir=model_dir,
            quantize_enabled=bool(cfg_export.get("quantize", True)),
            quantize_mode=str(cfg_export.get("quantize_mode", "dynamic")),
            logger=logger,
        )

    if quant_ok and quant_path:
        final_infer = quant_path
    elif fp16_path:
        final_infer = fp16_path
    elif onnx_path:
        final_infer = onnx_path
    else:
        final_infer = str(trained_pt)

    return ExportArtifacts(
        onnx_path=onnx_path,
        quantized_onnx_path=quant_path,
        fp16_onnx_path=fp16_path,
        final_infer_model_path=final_infer,
        export_ok=export_ok,
        quantize_ok=quant_ok,
        quantize_strategy=quant_strategy,
        messages=[*export_msgs, *quant_msgs],
    )
