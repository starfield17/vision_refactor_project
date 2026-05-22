"""Faster R-CNN ONNX export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_faster_rcnn_to_onnx(
    *,
    model: Any,
    output_onnx_path: Path,
    img_size: int,
    opset: int,
    logger: Any,
) -> tuple[bool, str | None, list[str]]:
    """Export a TorchVision Faster R-CNN model to ONNX."""

    import torch

    messages: list[str] = []
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import onnx
    except Exception as exc:
        msg = f"onnx import failed: {exc}"
        messages.append(msg)
        logger.warn("train.faster_rcnn.export.onnx.failed", "ONNX import failed", error=str(exc))
        return False, None, messages

    try:
        export_device = torch.device("cpu")
        model.to(export_device)
        model.eval()
        dummy = torch.rand(
            1,
            3,
            int(img_size),
            int(img_size),
            dtype=torch.float32,
            device=export_device,
        )
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                str(output_onnx_path),
                opset_version=int(opset),
                input_names=["images"],
                output_names=["boxes", "labels", "scores"],
                dynamic_axes={
                    "images": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "num_detections"},
                    "labels": {0: "num_detections"},
                    "scores": {0: "num_detections"},
                },
                dynamo=False,
            )
        onnx_model = onnx.load(str(output_onnx_path))
        onnx.checker.check_model(onnx_model)
        messages.append(f"faster_rcnn onnx export succeeded: {output_onnx_path}")
        logger.info(
            "train.faster_rcnn.export.onnx.ok",
            "Faster R-CNN ONNX export succeeded",
            onnx_path=str(output_onnx_path),
        )
        return True, str(output_onnx_path), messages
    except Exception as exc:
        msg = f"faster_rcnn onnx export failed: {exc}"
        messages.append(msg)
        logger.warn(
            "train.faster_rcnn.export.onnx.failed",
            "Faster R-CNN ONNX export failed",
            error=str(exc),
        )
        return False, None, messages
