"""Role-specific config helpers for distributed runtime packages."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.config.config_loader import load_toml_config
from common.config.schema import (
    AUTOLABEL_CONFLICTS,
    AUTOLABEL_MODES,
    AUTOLABEL_MODEL_BACKENDS,
    DEFAULT_CONFIG as KERNEL_DEFAULT_CONFIG,
    EDGE_MODES,
    EDGE_SOURCES,
    FASTER_RCNN_VARIANTS,
    LOCATE_ANYTHING_BNB_COMPUTE_DTYPES,
    LOCATE_ANYTHING_BNB_QUANT_TYPES,
    LOCATE_ANYTHING_GENERATION_MODES,
    LOCATE_ANYTHING_QUANTIZATION_MODES,
    QUANTIZE_MODES,
    STATISTICS_STORAGES,
    TRAIN_BACKENDS,
    deep_merge_dict,
    validate_config as validate_kernel_config,
)
from common.types.errors import ConfigError


WORKSPACE_DEFAULT = {
    "root": "../../work-dir",
    "run_name": "exp001",
    "log_file": "log.txt",
    "log_level": "INFO",
}

CLASS_MAP_DEFAULT = {
    "names": ["Kitchen_waste", "Recyclable_waste", "Hazardous_waste", "Other_waste"],
    "id_map": {
        "Kitchen_waste": 0,
        "Recyclable_waste": 1,
        "Hazardous_waste": 2,
        "Other_waste": 3,
    },
}

TRAIN_RUNTIME_DEFAULT = deepcopy(KERNEL_DEFAULT_CONFIG["train"])
EXPORT_DEFAULT = deepcopy(KERNEL_DEFAULT_CONFIG["export"])
AUTOLABEL_RUNTIME_DEFAULT = deepcopy(KERNEL_DEFAULT_CONFIG["autolabel"])
LOCATE_ANYTHING_DEFAULT = deepcopy(KERNEL_DEFAULT_CONFIG["locate_anything"])
EDGE_RUNTIME_DEFAULT = deepcopy(KERNEL_DEFAULT_CONFIG["deploy"]["edge"])
REMOTE_RUNTIME_DEFAULT = deepcopy(KERNEL_DEFAULT_CONFIG["deploy"]["remote"])
STATISTICS_RUNTIME_DEFAULT = deepcopy(KERNEL_DEFAULT_CONFIG["deploy"]["statistics"])


def expect_type(raw: dict[str, Any], key: str, t: type | tuple[type, ...], where: str) -> Any:
    if key not in raw:
        raise ConfigError(f"Missing required field: {where}.{key}")
    value = raw[key]
    if not isinstance(value, t):
        raise ConfigError(
            f"Invalid type for {where}.{key}: expected {t}, got {type(value).__name__}"
        )
    return value


def validate_workspace(cfg: dict[str, Any]) -> None:
    workspace = expect_type(cfg, "workspace", dict, "root")
    expect_type(workspace, "root", str, "workspace")
    expect_type(workspace, "run_name", str, "workspace")
    expect_type(workspace, "log_file", str, "workspace")
    level = expect_type(workspace, "log_level", str, "workspace")
    if level not in {"DEBUG", "INFO", "WARN", "ERROR"}:
        raise ConfigError("workspace.log_level must be one of ['DEBUG', 'ERROR', 'INFO', 'WARN']")


def validate_server(cfg: dict[str, Any], where: str = "server") -> None:
    server = expect_type(cfg, where, dict, "root")
    host = expect_type(server, "host", str, where)
    port = expect_type(server, "port", int, where)
    if not host:
        raise ConfigError(f"{where}.host must not be empty")
    if not (1 <= port <= 65535):
        raise ConfigError(f"{where}.port must be in [1, 65535]")
    expect_type(server, "api_token", str, where)
    expect_type(server, "api_token_env_name", str, where)
    if "advertise_url" in server:
        expect_type(server, "advertise_url", str, where)


def validate_job_store(cfg: dict[str, Any]) -> None:
    job = expect_type(cfg, "job_store", dict, "root")
    path = expect_type(job, "db_path", str, "job_store")
    if not path:
        raise ConfigError("job_store.db_path must not be empty")


def validate_control_plane_ref(cfg: dict[str, Any], required: bool = False) -> None:
    cp = expect_type(cfg, "control_plane", dict, "root")
    url = expect_type(cp, "url", str, "control_plane")
    expect_type(cp, "api_token", str, "control_plane")
    expect_type(cp, "api_token_env_name", str, "control_plane")
    if "heartbeat_interval_sec" in cp:
        interval = expect_type(cp, "heartbeat_interval_sec", int, "control_plane")
        if interval < 0:
            raise ConfigError("control_plane.heartbeat_interval_sec must be >= 0")
    if required and not url:
        raise ConfigError("control_plane.url must not be empty")


def validate_node(cfg: dict[str, Any], role: str) -> None:
    node = expect_type(cfg, "node", dict, "root")
    node_id = expect_type(node, "id", str, "node")
    node_role = expect_type(node, "role", str, "node")
    if not node_id:
        raise ConfigError("node.id must not be empty")
    if node_role != role:
        raise ConfigError(f"node.role must be {role!r}")
    if "endpoint" in node:
        expect_type(node, "endpoint", str, "node")


def validate_class_map(cfg: dict[str, Any]) -> None:
    class_map = expect_type(cfg, "class_map", dict, "root")
    names = expect_type(class_map, "names", list, "class_map")
    id_map = expect_type(class_map, "id_map", dict, "class_map")
    if not names:
        raise ConfigError("class_map.names must not be empty")
    if len(set(names)) != len(names):
        raise ConfigError("class_map.names contains duplicate items")
    expected = {name: idx for idx, name in enumerate(names)}
    if id_map != expected:
        raise ConfigError("class_map.id_map must exactly match names order")


def validate_train_runtime(runtime: dict[str, Any], data: dict[str, Any]) -> None:
    backend = expect_type(runtime, "backend", str, "runtime")
    if backend not in TRAIN_BACKENDS:
        raise ConfigError(f"runtime.backend must be one of {sorted(TRAIN_BACKENDS)}")
    expect_type(runtime, "device", str, "runtime")
    expect_type(runtime, "seed", int, "runtime")
    epochs = expect_type(runtime, "epochs", int, "runtime")
    batch_size = expect_type(runtime, "batch_size", int, "runtime")
    img_size = expect_type(runtime, "img_size", int, "runtime")
    expect_type(runtime, "dry_run", bool, "runtime")
    if epochs <= 0:
        raise ConfigError("runtime.epochs must be > 0")
    if batch_size <= 0:
        raise ConfigError("runtime.batch_size must be > 0")
    if img_size <= 0:
        raise ConfigError("runtime.img_size must be > 0")
    if backend == "yolo" and not str(data.get("yolo_dataset_dir", "")):
        raise ConfigError("data.yolo_dataset_dir must not be empty when runtime.backend=yolo")
    yolo = expect_type(runtime, "yolo", dict, "runtime")
    expect_type(yolo, "weights", str, "runtime.yolo")
    faster = expect_type(runtime, "faster_rcnn", dict, "runtime")
    variant = expect_type(faster, "variant", str, "runtime.faster_rcnn")
    if variant not in FASTER_RCNN_VARIANTS:
        raise ConfigError(f"runtime.faster_rcnn.variant must be one of {sorted(FASTER_RCNN_VARIANTS)}")
    if float(expect_type(faster, "lr", (int, float), "runtime.faster_rcnn")) <= 0:
        raise ConfigError("runtime.faster_rcnn.lr must be > 0")
    momentum = float(expect_type(faster, "momentum", (int, float), "runtime.faster_rcnn"))
    if not (0.0 <= momentum <= 1.0):
        raise ConfigError("runtime.faster_rcnn.momentum must be in [0, 1]")
    if float(expect_type(faster, "weight_decay", (int, float), "runtime.faster_rcnn")) < 0:
        raise ConfigError("runtime.faster_rcnn.weight_decay must be >= 0")
    if expect_type(faster, "num_workers", int, "runtime.faster_rcnn") < 0:
        raise ConfigError("runtime.faster_rcnn.num_workers must be >= 0")
    if expect_type(faster, "max_samples", int, "runtime.faster_rcnn") < 0:
        raise ConfigError("runtime.faster_rcnn.max_samples must be >= 0")


def validate_export(export_cfg: dict[str, Any]) -> None:
    expect_type(export_cfg, "onnx", bool, "export")
    expect_type(export_cfg, "opset", int, "export")
    expect_type(export_cfg, "quantize", bool, "export")
    mode = expect_type(export_cfg, "quantize_mode", str, "export")
    expect_type(export_cfg, "calib_samples", int, "export")
    if mode == "static":
        raise ConfigError("export.quantize_mode=static is not implemented yet; use dynamic")
    if mode not in QUANTIZE_MODES:
        raise ConfigError(f"export.quantize_mode must be one of {sorted(QUANTIZE_MODES)}")


def validate_locate_anything(cfg: dict[str, Any]) -> None:
    expect_type(cfg, "model", str, "locate_anything")
    expect_type(cfg, "device", str, "locate_anything")
    expect_type(cfg, "dtype", str, "locate_anything")
    quant = expect_type(cfg, "quantization", str, "locate_anything")
    if quant not in LOCATE_ANYTHING_QUANTIZATION_MODES:
        raise ConfigError(
            f"locate_anything.quantization must be one of {sorted(LOCATE_ANYTHING_QUANTIZATION_MODES)}"
        )
    compute = expect_type(cfg, "bnb_4bit_compute_dtype", str, "locate_anything")
    if compute not in LOCATE_ANYTHING_BNB_COMPUTE_DTYPES:
        raise ConfigError(
            "locate_anything.bnb_4bit_compute_dtype must be one of "
            f"{sorted(LOCATE_ANYTHING_BNB_COMPUTE_DTYPES)}"
        )
    quant_type = expect_type(cfg, "bnb_4bit_quant_type", str, "locate_anything")
    if quant_type not in LOCATE_ANYTHING_BNB_QUANT_TYPES:
        raise ConfigError(
            f"locate_anything.bnb_4bit_quant_type must be one of {sorted(LOCATE_ANYTHING_BNB_QUANT_TYPES)}"
        )
    expect_type(cfg, "bnb_4bit_use_double_quant", bool, "locate_anything")
    expect_type(cfg, "device_map", str, "locate_anything")
    expect_type(cfg, "attn_implementation", str, "locate_anything")
    generation = expect_type(cfg, "generation_mode", str, "locate_anything")
    if generation not in LOCATE_ANYTHING_GENERATION_MODES:
        raise ConfigError(
            f"locate_anything.generation_mode must be one of {sorted(LOCATE_ANYTHING_GENERATION_MODES)}"
        )
    if expect_type(cfg, "max_new_tokens", int, "locate_anything") <= 0:
        raise ConfigError("locate_anything.max_new_tokens must be > 0")
    expect_type(cfg, "temperature", (int, float), "locate_anything")
    prompt = expect_type(cfg, "prompt_template", str, "locate_anything")
    if "{class_name}" not in prompt:
        raise ConfigError("locate_anything.prompt_template must include {class_name}")
    nms = float(expect_type(cfg, "nms_iou", (int, float), "locate_anything"))
    score = float(expect_type(cfg, "default_score", (int, float), "locate_anything"))
    if not (0.0 <= nms <= 1.0):
        raise ConfigError("locate_anything.nms_iou must be in [0, 1]")
    if not (0.0 <= score <= 1.0):
        raise ConfigError("locate_anything.default_score must be in [0, 1]")
    expect_type(cfg, "verbose", bool, "locate_anything")
    if expect_type(cfg, "max_images", int, "locate_anything") < 0:
        raise ConfigError("locate_anything.max_images must be >= 0")


def validate_autolabel_runtime(runtime: dict[str, Any]) -> None:
    mode = expect_type(runtime, "mode", str, "runtime")
    if mode not in AUTOLABEL_MODES:
        raise ConfigError(f"runtime.mode must be one of {sorted(AUTOLABEL_MODES)}")
    confidence = float(expect_type(runtime, "confidence", (int, float), "runtime"))
    if not (0.0 <= confidence <= 1.0):
        raise ConfigError("runtime.confidence must be in [0, 1]")
    if expect_type(runtime, "batch_size", int, "runtime") <= 0:
        raise ConfigError("runtime.batch_size must be > 0")
    expect_type(runtime, "visualize", bool, "runtime")
    conflict = expect_type(runtime, "on_conflict", str, "runtime")
    if conflict not in AUTOLABEL_CONFLICTS:
        raise ConfigError(f"runtime.on_conflict must be one of {sorted(AUTOLABEL_CONFLICTS)}")
    model_cfg = expect_type(runtime, "model", dict, "runtime")
    expect_type(model_cfg, "onnx_model", str, "runtime.model")
    backend = expect_type(model_cfg, "backend", str, "runtime.model")
    if backend not in AUTOLABEL_MODEL_BACKENDS:
        raise ConfigError(f"runtime.model.backend must be one of {sorted(AUTOLABEL_MODEL_BACKENDS)}")
    if mode == "model" and not model_cfg["onnx_model"]:
        raise ConfigError("runtime.model.onnx_model must not be empty in model mode")
    llm = expect_type(runtime, "llm", dict, "runtime")
    expect_type(llm, "base_url", str, "runtime.llm")
    expect_type(llm, "model", str, "runtime.llm")
    expect_type(llm, "api_key", str, "runtime.llm")
    expect_type(llm, "api_key_env_name", str, "runtime.llm")
    expect_type(llm, "prompt", str, "runtime.llm")
    if float(expect_type(llm, "timeout_sec", (int, float), "runtime.llm")) <= 0:
        raise ConfigError("runtime.llm.timeout_sec must be > 0")
    if expect_type(llm, "max_retries", int, "runtime.llm") < 0:
        raise ConfigError("runtime.llm.max_retries must be >= 0")
    if float(expect_type(llm, "retry_backoff_sec", (int, float), "runtime.llm")) <= 0:
        raise ConfigError("runtime.llm.retry_backoff_sec must be > 0")
    if float(expect_type(llm, "qps_limit", (int, float), "runtime.llm")) <= 0:
        raise ConfigError("runtime.llm.qps_limit must be > 0")
    if expect_type(llm, "max_images", int, "runtime.llm") < 0:
        raise ConfigError("runtime.llm.max_images must be >= 0")


def validate_edge_runtime(runtime: dict[str, Any]) -> None:
    expect_type(runtime, "source_id", str, "runtime")
    mode = expect_type(runtime, "mode", str, "runtime")
    if mode not in EDGE_MODES:
        raise ConfigError(f"runtime.mode must be one of {sorted(EDGE_MODES)}")
    source = expect_type(runtime, "source", str, "runtime")
    if source not in EDGE_SOURCES:
        raise ConfigError(f"runtime.source must be one of {sorted(EDGE_SOURCES)}")
    expect_type(runtime, "camera_id", int, "runtime")
    expect_type(runtime, "video_path", str, "runtime")
    expect_type(runtime, "images_dir", str, "runtime")
    expect_type(runtime, "local_model", str, "runtime")
    expect_type(runtime, "stats_endpoint", str, "runtime")
    expect_type(runtime, "api_key", str, "runtime")
    confidence = float(expect_type(runtime, "confidence", (int, float), "runtime"))
    if not (0.0 <= confidence <= 1.0):
        raise ConfigError("runtime.confidence must be in [0, 1]")
    if float(expect_type(runtime, "fps_limit", (int, float), "runtime")) <= 0:
        raise ConfigError("runtime.fps_limit must be > 0")
    quality = expect_type(runtime, "jpeg_quality", int, "runtime")
    if not (1 <= quality <= 100):
        raise ConfigError("runtime.jpeg_quality must be in [1, 100]")
    if float(expect_type(runtime, "stats_timeout_sec", (int, float), "runtime")) <= 0:
        raise ConfigError("runtime.stats_timeout_sec must be > 0")
    expect_type(runtime, "save_annotated", bool, "runtime")
    if expect_type(runtime, "max_frames", int, "runtime") < 0:
        raise ConfigError("runtime.max_frames must be >= 0")
    expect_type(runtime, "stream_endpoint", str, "runtime")
    if float(expect_type(runtime, "stream_timeout_sec", (int, float), "runtime")) <= 0:
        raise ConfigError("runtime.stream_timeout_sec must be > 0")
    expect_type(runtime, "stream_api_key", str, "runtime")
    llm = expect_type(runtime, "llm", dict, "runtime")
    expect_type(llm, "base_url", str, "runtime.llm")
    expect_type(llm, "model", str, "runtime.llm")
    expect_type(llm, "api_key", str, "runtime.llm")
    expect_type(llm, "api_key_env_name", str, "runtime.llm")
    expect_type(llm, "prompt", str, "runtime.llm")
    if float(expect_type(llm, "timeout_sec", (int, float), "runtime.llm")) <= 0:
        raise ConfigError("runtime.llm.timeout_sec must be > 0")
    if expect_type(llm, "max_retries", int, "runtime.llm") < 0:
        raise ConfigError("runtime.llm.max_retries must be >= 0")
    if float(expect_type(llm, "retry_backoff_sec", (int, float), "runtime.llm")) <= 0:
        raise ConfigError("runtime.llm.retry_backoff_sec must be > 0")
    if float(expect_type(llm, "qps_limit", (int, float), "runtime.llm")) <= 0:
        raise ConfigError("runtime.llm.qps_limit must be > 0")


def validate_remote_runtime(runtime: dict[str, Any]) -> None:
    expect_type(runtime, "source_id", str, "runtime")
    expect_type(runtime, "listen_host", str, "runtime")
    port = expect_type(runtime, "listen_port", int, "runtime")
    if not (1 <= port <= 65535):
        raise ConfigError("runtime.listen_port must be in [1, 65535]")
    expect_type(runtime, "model", str, "runtime")
    confidence = float(expect_type(runtime, "confidence", (int, float), "runtime"))
    if not (0.0 <= confidence <= 1.0):
        raise ConfigError("runtime.confidence must be in [0, 1]")
    expect_type(runtime, "stats_endpoint", str, "runtime")
    expect_type(runtime, "api_key", str, "runtime")
    expect_type(runtime, "ingest_api_key", str, "runtime")
    if float(expect_type(runtime, "stats_timeout_sec", (int, float), "runtime")) <= 0:
        raise ConfigError("runtime.stats_timeout_sec must be > 0")
    expect_type(runtime, "save_annotated", bool, "runtime")
    if expect_type(runtime, "max_payload_mb", int, "runtime") <= 0:
        raise ConfigError("runtime.max_payload_mb must be > 0")


def validate_statistics_runtime(runtime: dict[str, Any]) -> None:
    ui_port = expect_type(runtime, "ui_port", int, "runtime")
    api_port = expect_type(runtime, "api_port", int, "runtime")
    if not (1 <= ui_port <= 65535):
        raise ConfigError("runtime.ui_port must be in [1, 65535]")
    if not (1 <= api_port <= 65535):
        raise ConfigError("runtime.api_port must be in [1, 65535]")
    storage = expect_type(runtime, "storage", str, "runtime")
    if storage not in STATISTICS_STORAGES:
        raise ConfigError(f"runtime.storage must be one of {sorted(STATISTICS_STORAGES)}")
    db_path = expect_type(runtime, "db_path", str, "runtime")
    if not db_path:
        raise ConfigError("runtime.db_path must not be empty")
    expect_type(runtime, "public_host", str, "runtime")
    expect_type(runtime, "api_key", str, "runtime")
    if expect_type(runtime, "rate_limit_per_sec", int, "runtime") < 0:
        raise ConfigError("runtime.rate_limit_per_sec must be >= 0")


def load_role_config(
    config_path: Path,
    default_config: dict[str, Any],
    validate: Any,
    path_fields: tuple[tuple[str, ...], ...],
    overrides: list[str] | None = None,
    workdir_override: str | None = None,
) -> dict[str, Any]:
    return load_toml_config(
        config_path=config_path,
        default_config=default_config,
        validate=validate,
        path_fields=path_fields,
        overrides=overrides,
        workdir_override=workdir_override,
    )


def role_to_kernel_config(
    role_cfg: dict[str, Any],
    role: str,
    service_name: str,
) -> dict[str, Any]:
    """Adapt a role config into the config shape consumed by core."""
    cfg = deepcopy(KERNEL_DEFAULT_CONFIG)
    cfg["workspace"] = deepcopy(role_cfg["workspace"])
    cfg["class_map"] = deepcopy(role_cfg.get("class_map", CLASS_MAP_DEFAULT))
    cfg["data"]["yolo_dataset_dir"] = "./work-dir/datasets/yolo"
    cfg["autolabel"]["model"]["onnx_model"] = "./work-dir/models/placeholder.onnx"

    data = role_cfg.get("data")
    if isinstance(data, dict):
        cfg["data"] = deep_merge_dict(cfg["data"], data)

    if "runtime" in role_cfg:
        if role == "train":
            cfg["train"] = deep_merge_dict(cfg["train"], role_cfg["runtime"])
            cfg["export"] = deep_merge_dict(cfg["export"], role_cfg.get("export", {}))
        elif role == "autolabel":
            cfg["autolabel"] = deep_merge_dict(cfg["autolabel"], role_cfg["runtime"])
            if "train" in role_cfg:
                cfg["train"] = deep_merge_dict(cfg["train"], role_cfg["train"])
            if "locate_anything" in role_cfg:
                cfg["locate_anything"] = deep_merge_dict(
                    cfg["locate_anything"], role_cfg["locate_anything"]
                )
        elif role == "edge":
            cfg["deploy"]["edge"] = deep_merge_dict(cfg["deploy"]["edge"], role_cfg["runtime"])
            if "train" in role_cfg:
                cfg["train"] = deep_merge_dict(cfg["train"], role_cfg["train"])
            if "locate_anything" in role_cfg:
                cfg["locate_anything"] = deep_merge_dict(
                    cfg["locate_anything"], role_cfg["locate_anything"]
                )
        elif role == "remote":
            cfg["deploy"]["remote"] = deep_merge_dict(cfg["deploy"]["remote"], role_cfg["runtime"])
            if "train" in role_cfg:
                cfg["train"] = deep_merge_dict(cfg["train"], role_cfg["train"])
            edge = role_cfg.get("edge")
            if isinstance(edge, dict):
                cfg["deploy"]["edge"] = deep_merge_dict(cfg["deploy"]["edge"], edge)
        elif role == "statistics":
            cfg["deploy"]["statistics"] = deep_merge_dict(
                cfg["deploy"]["statistics"], role_cfg["runtime"]
            )

    return validate_kernel_config(cfg)
