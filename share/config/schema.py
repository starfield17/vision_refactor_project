"""Configuration defaults and validation helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from share.types.errors import ConfigError


LOG_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}
TRAIN_BACKENDS = {"yolo", "faster_rcnn"}
AUTOLABEL_MODES = {"llm", "model"}
QUANTIZE_MODES = {"dynamic"}
AUTOLABEL_CONFLICTS = {"skip", "overwrite", "merge"}
AUTOLABEL_MODEL_BACKENDS = {"yolo", "faster_rcnn"}
FASTER_RCNN_VARIANTS = {"resnet50_fpn", "resnet50_fpn_v2", "mobilenet_v3", "resnet18_fpn"}
EDGE_MODES = {"local", "llm", "stream"}
EDGE_SOURCES = {"camera", "video", "images"}
STATISTICS_STORAGES = {"sqlite"}


DEFAULT_CONFIG: dict[str, Any] = {
    "app": {"schema_version": 1},
    "workspace": {
        "root": "./work-dir",
        "run_name": "exp001",
        "log_file": "log.txt",
        "log_level": "INFO",
    },
    "class_map": {
        "names": [],
        "id_map": {},
    },
    "data": {
        "yolo_dataset_dir": "",
        "labeled_dir": "./work-dir/datasets/labeled",
        "unlabeled_dir": "./work-dir/datasets/unlabeled",
    },
    "train": {
        "backend": "yolo",
        "device": "cpu",
        "seed": 42,
        "epochs": 1,
        "batch_size": 4,
        "img_size": 640,
        "dry_run": True,
        "yolo": {"weights": ""},
        "faster_rcnn": {
            "variant": "mobilenet_v3",
            "lr": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "num_workers": 0,
            "max_samples": 0,
        },
    },
    "export": {
        "onnx": True,
        "opset": 17,
        "quantize": True,
        "quantize_mode": "dynamic",
        "calib_samples": 32,
    },
    "autolabel": {
        "mode": "model",
        "confidence": 0.5,
        "batch_size": 2,
        "visualize": False,
        "on_conflict": "skip",
        "llm": {
            "base_url": "",
            "model": "",
            "api_key_env": "",
            "prompt": "",
            "timeout_sec": 60.0,
            "max_retries": 2,
            "retry_backoff_sec": 1.5,
            "qps_limit": 1.0,
            "max_images": 0,
        },
        "model": {"onnx_model": "", "backend": "yolo"},
    },
    "deploy": {
        "edge": {
            "source_id": "edge-001",
            "mode": "local",
            "source": "images",
            "camera_id": 0,
            "video_path": "",
            "images_dir": "",
            "fps_limit": 10,
            "jpeg_quality": 80,
            "confidence": 0.5,
            "local_model": "",
            "stats_endpoint": "http://127.0.0.1:7797/api/v1/push",
            "api_key": "",
            "stats_timeout_sec": 2.0,
            "save_annotated": True,
            "max_frames": 0,
            "stream_endpoint": "http://127.0.0.1:60051/api/v1/frame",
            "stream_timeout_sec": 5.0,
            "stream_api_key": "",
            "llm": {
                "base_url": "",
                "model": "",
                "api_key_env": "",
                "prompt": "",
                "timeout_sec": 60.0,
                "max_retries": 2,
                "retry_backoff_sec": 1.5,
                "qps_limit": 1.0,
            },
        },
        "remote": {
            "source_id": "remote-001",
            "listen_host": "0.0.0.0",
            "listen_port": 60051,
            "model": "",
            "confidence": 0.5,
            "stats_endpoint": "http://127.0.0.1:7797/api/v1/push",
            "api_key": "",
            "ingest_api_key": "",
            "stats_timeout_sec": 2.0,
            "save_annotated": False,
            "max_payload_mb": 8,
        },
        "statistics": {
            "ui_port": 7796,
            "api_port": 7797,
            "storage": "sqlite",
            "db_path": "./work-dir/stats/stats.db",
            "public_host": "0.0.0.0",
            "api_key": "",
            "rate_limit_per_sec": 0,
        },
    },
    "logging": {"jsonl": True},
    "statistics": {"flush_interval_sec": 5},
}


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expect_type(raw: dict[str, Any], key: str, t: type | tuple[type, ...], where: str) -> Any:
    if key not in raw:
        raise ConfigError(f"Missing required field: {where}.{key}")
    value = raw[key]
    if not isinstance(value, t):
        raise ConfigError(
            f"Invalid type for {where}.{key}: expected {t}, got {type(value).__name__}"
        )
    return value


def _validate_class_map(cfg: dict[str, Any]) -> None:
    class_map = _expect_type(cfg, "class_map", dict, "root")
    names = _expect_type(class_map, "names", list, "class_map")
    id_map = _expect_type(class_map, "id_map", dict, "class_map")

    if not names:
        raise ConfigError("class_map.names must not be empty")
    if len(set(names)) != len(names):
        raise ConfigError("class_map.names contains duplicate items")

    expected = {name: idx for idx, name in enumerate(names)}
    if id_map != expected:
        raise ConfigError(
            "class_map.id_map must exactly match names order, e.g. {a:0,b:1}"
        )


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    _expect_type(cfg, "app", dict, "root")
    _expect_type(cfg["app"], "schema_version", int, "app")
    if cfg["app"]["schema_version"] != 1:
        raise ConfigError("app.schema_version currently must be 1")

    workspace = _expect_type(cfg, "workspace", dict, "root")
    _expect_type(workspace, "root", str, "workspace")
    _expect_type(workspace, "run_name", str, "workspace")
    _expect_type(workspace, "log_file", str, "workspace")
    log_level = _expect_type(workspace, "log_level", str, "workspace")
    if log_level not in LOG_LEVELS:
        raise ConfigError(f"workspace.log_level must be one of {sorted(LOG_LEVELS)}")

    _expect_type(cfg, "data", dict, "root")
    yolo_dataset_dir = _expect_type(cfg["data"], "yolo_dataset_dir", str, "data")
    _expect_type(cfg["data"], "labeled_dir", str, "data")
    _expect_type(cfg["data"], "unlabeled_dir", str, "data")

    train = _expect_type(cfg, "train", dict, "root")
    backend = _expect_type(train, "backend", str, "train")
    if backend not in TRAIN_BACKENDS:
        raise ConfigError(f"train.backend must be one of {sorted(TRAIN_BACKENDS)}")
    _expect_type(train, "device", str, "train")
    _expect_type(train, "seed", int, "train")
    _expect_type(train, "epochs", int, "train")
    _expect_type(train, "batch_size", int, "train")
    _expect_type(train, "img_size", int, "train")
    _expect_type(train, "dry_run", bool, "train")
    if train["epochs"] <= 0:
        raise ConfigError("train.epochs must be > 0")
    if train["batch_size"] <= 0:
        raise ConfigError("train.batch_size must be > 0")
    if train["img_size"] <= 0:
        raise ConfigError("train.img_size must be > 0")

    if backend == "yolo":
        yolo_cfg = _expect_type(train, "yolo", dict, "train")
        _expect_type(yolo_cfg, "weights", str, "train.yolo")
        if not yolo_dataset_dir:
            raise ConfigError("data.yolo_dataset_dir must not be empty when train.backend=yolo")
    elif backend == "faster_rcnn":
        faster_cfg = _expect_type(train, "faster_rcnn", dict, "train")
        variant = _expect_type(faster_cfg, "variant", str, "train.faster_rcnn")
        if variant not in FASTER_RCNN_VARIANTS:
            raise ConfigError(
                f"train.faster_rcnn.variant must be one of {sorted(FASTER_RCNN_VARIANTS)}"
            )
        lr = _expect_type(faster_cfg, "lr", (int, float), "train.faster_rcnn")
        if float(lr) <= 0:
            raise ConfigError("train.faster_rcnn.lr must be > 0")
        momentum = _expect_type(faster_cfg, "momentum", (int, float), "train.faster_rcnn")
        if not (0.0 <= float(momentum) <= 1.0):
            raise ConfigError("train.faster_rcnn.momentum must be in [0, 1]")
        weight_decay = _expect_type(
            faster_cfg, "weight_decay", (int, float), "train.faster_rcnn"
        )
        if float(weight_decay) < 0:
            raise ConfigError("train.faster_rcnn.weight_decay must be >= 0")
        num_workers = _expect_type(faster_cfg, "num_workers", int, "train.faster_rcnn")
        if num_workers < 0:
            raise ConfigError("train.faster_rcnn.num_workers must be >= 0")
        max_samples = _expect_type(faster_cfg, "max_samples", int, "train.faster_rcnn")
        if max_samples < 0:
            raise ConfigError("train.faster_rcnn.max_samples must be >= 0")

    export_cfg = _expect_type(cfg, "export", dict, "root")
    _expect_type(export_cfg, "onnx", bool, "export")
    _expect_type(export_cfg, "opset", int, "export")
    _expect_type(export_cfg, "quantize", bool, "export")
    quantize_mode = _expect_type(export_cfg, "quantize_mode", str, "export")
    _expect_type(export_cfg, "calib_samples", int, "export")
    if quantize_mode == "static":
        raise ConfigError("export.quantize_mode=static is not implemented yet; use dynamic")
    if quantize_mode not in QUANTIZE_MODES:
        raise ConfigError(f"export.quantize_mode must be one of {sorted(QUANTIZE_MODES)}")

    autolabel = _expect_type(cfg, "autolabel", dict, "root")
    mode = _expect_type(autolabel, "mode", str, "autolabel")
    if mode not in AUTOLABEL_MODES:
        raise ConfigError(f"autolabel.mode must be one of {sorted(AUTOLABEL_MODES)}")
    confidence = _expect_type(autolabel, "confidence", (int, float), "autolabel")
    if not (0.0 <= float(confidence) <= 1.0):
        raise ConfigError("autolabel.confidence must be in [0, 1]")
    batch_size = _expect_type(autolabel, "batch_size", int, "autolabel")
    if batch_size <= 0:
        raise ConfigError("autolabel.batch_size must be > 0")
    _expect_type(autolabel, "visualize", bool, "autolabel")
    on_conflict = _expect_type(autolabel, "on_conflict", str, "autolabel")
    if on_conflict not in AUTOLABEL_CONFLICTS:
        raise ConfigError(f"autolabel.on_conflict must be one of {sorted(AUTOLABEL_CONFLICTS)}")

    autolabel_model = _expect_type(autolabel, "model", dict, "autolabel")
    _expect_type(autolabel_model, "onnx_model", str, "autolabel.model")
    model_backend = _expect_type(autolabel_model, "backend", str, "autolabel.model")
    if model_backend not in AUTOLABEL_MODEL_BACKENDS:
        raise ConfigError(
            f"autolabel.model.backend must be one of {sorted(AUTOLABEL_MODEL_BACKENDS)}"
        )
    if mode == "model" and not autolabel_model["onnx_model"]:
        raise ConfigError("autolabel.model.onnx_model must not be empty in model mode")

    autolabel_llm = _expect_type(autolabel, "llm", dict, "autolabel")
    _expect_type(autolabel_llm, "base_url", str, "autolabel.llm")
    _expect_type(autolabel_llm, "model", str, "autolabel.llm")
    _expect_type(autolabel_llm, "api_key_env", str, "autolabel.llm")
    _expect_type(autolabel_llm, "prompt", str, "autolabel.llm")
    timeout_sec = _expect_type(autolabel_llm, "timeout_sec", (int, float), "autolabel.llm")
    if float(timeout_sec) <= 0:
        raise ConfigError("autolabel.llm.timeout_sec must be > 0")
    max_retries = _expect_type(autolabel_llm, "max_retries", int, "autolabel.llm")
    if max_retries < 0:
        raise ConfigError("autolabel.llm.max_retries must be >= 0")
    retry_backoff_sec = _expect_type(
        autolabel_llm, "retry_backoff_sec", (int, float), "autolabel.llm"
    )
    if float(retry_backoff_sec) <= 0:
        raise ConfigError("autolabel.llm.retry_backoff_sec must be > 0")
    qps_limit = _expect_type(autolabel_llm, "qps_limit", (int, float), "autolabel.llm")
    if float(qps_limit) <= 0:
        raise ConfigError("autolabel.llm.qps_limit must be > 0")
    max_images = _expect_type(autolabel_llm, "max_images", int, "autolabel.llm")
    if max_images < 0:
        raise ConfigError("autolabel.llm.max_images must be >= 0")
    if mode == "llm":
        if not autolabel_llm["base_url"] or not autolabel_llm["model"]:
            raise ConfigError("autolabel.llm.base_url/model must not be empty in llm mode")
        if not autolabel_llm["prompt"]:
            raise ConfigError("autolabel.llm.prompt must not be empty in llm mode")

    deploy_cfg = _expect_type(cfg, "deploy", dict, "root")

    edge_cfg = _expect_type(deploy_cfg, "edge", dict, "deploy")
    _expect_type(edge_cfg, "source_id", str, "deploy.edge")
    edge_mode = _expect_type(edge_cfg, "mode", str, "deploy.edge")
    if edge_mode not in EDGE_MODES:
        raise ConfigError(f"deploy.edge.mode must be one of {sorted(EDGE_MODES)}")
    edge_source = _expect_type(edge_cfg, "source", str, "deploy.edge")
    if edge_source not in EDGE_SOURCES:
        raise ConfigError(f"deploy.edge.source must be one of {sorted(EDGE_SOURCES)}")
    _expect_type(edge_cfg, "camera_id", int, "deploy.edge")
    _expect_type(edge_cfg, "video_path", str, "deploy.edge")
    _expect_type(edge_cfg, "images_dir", str, "deploy.edge")
    _expect_type(edge_cfg, "local_model", str, "deploy.edge")
    _expect_type(edge_cfg, "stats_endpoint", str, "deploy.edge")
    _expect_type(edge_cfg, "api_key", str, "deploy.edge")
    edge_confidence = _expect_type(edge_cfg, "confidence", (int, float), "deploy.edge")
    if not (0.0 <= float(edge_confidence) <= 1.0):
        raise ConfigError("deploy.edge.confidence must be in [0, 1]")
    fps_limit = _expect_type(edge_cfg, "fps_limit", (int, float), "deploy.edge")
    if float(fps_limit) <= 0:
        raise ConfigError("deploy.edge.fps_limit must be > 0")
    jpeg_quality = _expect_type(edge_cfg, "jpeg_quality", int, "deploy.edge")
    if not (1 <= jpeg_quality <= 100):
        raise ConfigError("deploy.edge.jpeg_quality must be in [1, 100]")
    timeout_sec = _expect_type(edge_cfg, "stats_timeout_sec", (int, float), "deploy.edge")
    if float(timeout_sec) <= 0:
        raise ConfigError("deploy.edge.stats_timeout_sec must be > 0")
    _expect_type(edge_cfg, "save_annotated", bool, "deploy.edge")
    max_frames = _expect_type(edge_cfg, "max_frames", int, "deploy.edge")
    if max_frames < 0:
        raise ConfigError("deploy.edge.max_frames must be >= 0")
    _expect_type(edge_cfg, "stream_endpoint", str, "deploy.edge")
    stream_timeout_sec = _expect_type(edge_cfg, "stream_timeout_sec", (int, float), "deploy.edge")
    if float(stream_timeout_sec) <= 0:
        raise ConfigError("deploy.edge.stream_timeout_sec must be > 0")
    _expect_type(edge_cfg, "stream_api_key", str, "deploy.edge")
    edge_llm_cfg = _expect_type(edge_cfg, "llm", dict, "deploy.edge")
    _expect_type(edge_llm_cfg, "base_url", str, "deploy.edge.llm")
    _expect_type(edge_llm_cfg, "model", str, "deploy.edge.llm")
    _expect_type(edge_llm_cfg, "api_key_env", str, "deploy.edge.llm")
    _expect_type(edge_llm_cfg, "prompt", str, "deploy.edge.llm")
    edge_llm_timeout = _expect_type(edge_llm_cfg, "timeout_sec", (int, float), "deploy.edge.llm")
    if float(edge_llm_timeout) <= 0:
        raise ConfigError("deploy.edge.llm.timeout_sec must be > 0")
    edge_llm_retries = _expect_type(edge_llm_cfg, "max_retries", int, "deploy.edge.llm")
    if edge_llm_retries < 0:
        raise ConfigError("deploy.edge.llm.max_retries must be >= 0")
    edge_llm_backoff = _expect_type(
        edge_llm_cfg, "retry_backoff_sec", (int, float), "deploy.edge.llm"
    )
    if float(edge_llm_backoff) <= 0:
        raise ConfigError("deploy.edge.llm.retry_backoff_sec must be > 0")
    edge_llm_qps = _expect_type(edge_llm_cfg, "qps_limit", (int, float), "deploy.edge.llm")
    if float(edge_llm_qps) <= 0:
        raise ConfigError("deploy.edge.llm.qps_limit must be > 0")
    if edge_mode == "llm":
        if not edge_llm_cfg["base_url"] or not edge_llm_cfg["model"]:
            raise ConfigError("deploy.edge.llm.base_url/model must not be empty in llm mode")
        if not edge_llm_cfg["prompt"]:
            raise ConfigError("deploy.edge.llm.prompt must not be empty in llm mode")
        if not edge_llm_cfg["api_key_env"]:
            raise ConfigError("deploy.edge.llm.api_key_env must not be empty in llm mode")

    remote_cfg = _expect_type(deploy_cfg, "remote", dict, "deploy")
    _expect_type(remote_cfg, "source_id", str, "deploy.remote")
    _expect_type(remote_cfg, "listen_host", str, "deploy.remote")
    listen_port = _expect_type(remote_cfg, "listen_port", int, "deploy.remote")
    if not (1 <= listen_port <= 65535):
        raise ConfigError("deploy.remote.listen_port must be in [1, 65535]")
    _expect_type(remote_cfg, "model", str, "deploy.remote")
    remote_confidence = _expect_type(remote_cfg, "confidence", (int, float), "deploy.remote")
    if not (0.0 <= float(remote_confidence) <= 1.0):
        raise ConfigError("deploy.remote.confidence must be in [0, 1]")
    _expect_type(remote_cfg, "stats_endpoint", str, "deploy.remote")
    _expect_type(remote_cfg, "api_key", str, "deploy.remote")
    _expect_type(remote_cfg, "ingest_api_key", str, "deploy.remote")
    remote_timeout = _expect_type(remote_cfg, "stats_timeout_sec", (int, float), "deploy.remote")
    if float(remote_timeout) <= 0:
        raise ConfigError("deploy.remote.stats_timeout_sec must be > 0")
    _expect_type(remote_cfg, "save_annotated", bool, "deploy.remote")
    max_payload_mb = _expect_type(remote_cfg, "max_payload_mb", int, "deploy.remote")
    if max_payload_mb <= 0:
        raise ConfigError("deploy.remote.max_payload_mb must be > 0")

    stats_cfg = _expect_type(deploy_cfg, "statistics", dict, "deploy")
    ui_port = _expect_type(stats_cfg, "ui_port", int, "deploy.statistics")
    api_port = _expect_type(stats_cfg, "api_port", int, "deploy.statistics")
    if not (1 <= ui_port <= 65535):
        raise ConfigError("deploy.statistics.ui_port must be in [1, 65535]")
    if not (1 <= api_port <= 65535):
        raise ConfigError("deploy.statistics.api_port must be in [1, 65535]")
    storage = _expect_type(stats_cfg, "storage", str, "deploy.statistics")
    if storage not in STATISTICS_STORAGES:
        raise ConfigError(
            f"deploy.statistics.storage must be one of {sorted(STATISTICS_STORAGES)}"
        )
    db_path = _expect_type(stats_cfg, "db_path", str, "deploy.statistics")
    if not db_path:
        raise ConfigError("deploy.statistics.db_path must not be empty")
    _expect_type(stats_cfg, "public_host", str, "deploy.statistics")
    _expect_type(stats_cfg, "api_key", str, "deploy.statistics")
    rate_limit = _expect_type(stats_cfg, "rate_limit_per_sec", int, "deploy.statistics")
    if rate_limit < 0:
        raise ConfigError("deploy.statistics.rate_limit_per_sec must be >= 0")

    _expect_type(cfg, "logging", dict, "root")
    _expect_type(cfg["logging"], "jsonl", bool, "logging")
    _expect_type(cfg, "statistics", dict, "root")
    flush_interval = _expect_type(cfg["statistics"], "flush_interval_sec", int, "statistics")
    if flush_interval <= 0:
        raise ConfigError("statistics.flush_interval_sec must be > 0")

    _validate_class_map(cfg)
    return cfg
