# share/kernel/

The `kernel/ logic for the Vision Refactor Project.
It is the largest and most important sub-package in `share/`.

The top-level modules (`train/`, `autolabel/`, `deploy/`) are thin CLI wrappers —
they parse arguments, build a registry, then delegate everything to the kernel.

---

## Table of Contents

1. [kernel.py — VisionKernel](#kernelpy--visionkernel)
2. [registry.py — KernelRegistry](#registrypy--kernelregistry)
3. [trainer/](#trainer)
4. [infer/](#infer)
5. [autolabel/](#autolabel)
6. [deploy/](#deploy)
7. [export/](#export)
8. [transport/](#transport)
9. [statistics/](#statistics)
10. [llm/](#llm)
11. [media/](#media)
12. [utils/](#utils)

---

## `kernel.py` — `VisionKernel`

The kernel orchestrates every pipeline run.  It is instantiated by each CLI with a
validated config dict, a logger, and a populated registry.

### `RunContext`

```python
@dataclass(slots=True)
class RunContext:
    run_id:  str    # "<run_name>-<YYYYMMDD-HHMMSS>"
    mode:    str    # "train" | "autolabel" | "deploy-edge" | "deploy-remote"
    workdir: Path
    run_dir: Path   # work-dir/runs/<run_id>/
```

### `RunResult`

```python
@dataclass(slots=True)
class RunResult:
    run_context: RunContext
    status:      str    # "ok" | "failed"
    backend:     str    # backend/mode name
    elapsed_ms:  float
    artifacts:   dict[str, Any]
    error:       str | None
```

### Methods

| Method | Triggered by |
|--------|-------------|
| `run_train()` | `train/cli.py` |
| `run_autolabel()` | `autolabel/cli.py` |
| `run_deploy_edge()` | `deploy/edge/cli.py` |
| `run_deploy_remote()` | `deploy/remote/cli.py` |
| `run_infer()` | Alias for `run_deploy_edge()` |

All methods follow the same pattern:
1. `_ensure_workdir_layout()` — create required subdirectories
2. `_make_run_context()` — generate unique run ID, create run directory
3. Lookup backend from registry
4. Call backend function inside `try/except` (always writes artifacts even on failure)
5. Write `metrics.json` and `artifacts.json`
6. Return `RunResult`

---

## `registry.py` — `KernelRegistry`

A lightweight plugin registry mapping string names to callable backends.

```python
class KernelRegistry:
    def register_trainer(name: str, fn: TrainerFn)     → None
    def get_trainer(name: str)                          → TrainerFn
    def register_autolabeler(name: str, fn: AutolabelFn) → None
    def get_autolabeler(name: str)                      → AutolabelFn
    def register_deployer(name: str, fn: DeployerFn)   → None
    def get_deployer(name: str)                         → DeployerFn
```

All callable types share the[str, Any]], dict[str, Any]]
# (cfg, run_ctx) → artifacts_dict
```

`run_ctx` always contains: `run_id` (str), `run_dir` (str), `logger` (StructuredLogger).

---

## `trainer/`

Contains training runner functions registered with the kernel.

### `trainer/yolo.py` — `run_yolo_train(cfg, run_ctx)`

Trains a YOLO model using the `ultralytics` library.

**Behaviour:**
- Reads `train.yolo.weights` as the pretrained checkpoint.
- Reads `data.yolo_dataset_dir` for the YOLO dataset (must contain `dataset.yaml`).
- Trains for `train.epochs` epochs on `train.device`.
- If `train.dry_run = true`, skips actual training and returns dummy artifacts.
- After training, calls `onnx_export.export_model()` to produce `.onnx` and `-int8.onnx`.

**Returns:**

```python
{
    "model_path":      "/abs/.../model.onnx",
    "model_int8_path": "/abs/.../model-int8.onnx",
    "epochs_trained":  50,
    "best_map50":      0.872,
}
```

### `trainer/faster_rcnn.py` — `run_faster_rcnn_train(cfg, run_ctx)`

Trains a Faster-RCNN model using `torchvision.models.detection`.

**Variants** (set via `train.faster_rcnn.variant`):

| Variant | Backbone |
|---------|---------|
| `mobilenet_v3` | MobileNetV3-Large FPN (default, fastest) |
| `resnet50_fpn` | ResNet-50 FPN |
| `resnet50_fpn_v2` | ResNet-50 FPN v2 |
| `resnet18_fpn` | ResNet-18 FPN (smallest) |

**Key config fields used:**
- `train.faster_rcnn.lr` — learning rate (SGD)
- `train.faster_rcnn.momentum` — SGD momentum
- `train.faster_rcnn.weight_decay` — L2 regularization
- `train.faster_rcnn.num_workers` — DataLoader workers
- `train.faster_rcnn.max_samples` — cap training set size (0 = all)

---

## `infer/`

ONNX inference adapters used by autolabel and edge deploy.

### `infer/local_yolo.py`

ONNX Runtime inference for YOLO-format models.  Handles pre-processing (resize, normalize),
inference, and NMS post-processing.  Returns a list of `Detection` objects.

### `infer/faster_rcnn.py`

ONNX Runtime inference for Faster-RCNN-format models.  Handles the different output format
(boxes, labels, scores) and maps class IDs to names using `cfg["class_map"]`.

---

## `autolabel/`

### `autolabel/model_autolabel.py` — `run_model_autolabel(cfg, run_ctx)`

Scans `data.unlabeled_dir` for images, runs ONNX inference using the configured model,
and writes YOLO-format `.txt` label files to `data.labeled_dir`.

**Conflict handling** (controlled by `autolabel.on_conflict`):
- `skip` — if a `.txt` label file already exists for an image, skip it entirely.
- `overwrite` — replace the existing `.txt` file with new predictions.
- `merge` — read the existing `.txt` file, append new detections that don't overlap
  significantly with existing ones (using IoU threshold), and write back.

If `autolabel.visualize = true`, saves annotated preview images alongside the label files.

### `autolabel/llm_autolabel.py` — `run_llm_autolabel(cfg, run_ctx)`

Sends images to an OpenAI-compatible vision API (e.g., GPT-4o, Claude) and parses
bounding box predictions from the response.

**Rate limiting:** The `qps_limit` field limits queries per second per the LLM API.
Uses a token-bucket algorithm to avoid API throttling errors.

**Retry logic:** On transient failures (HTTP 429, 500, 502, 503), retries up to
`autolabel.llm.max_retries` times with exponential backoff starting at
`autolabel.llm.retry_backoff_sec`.

---

## `deploy/`

### `deploy/edge_local.py` — `run_edge_local_deploy(cfg, run_ctx)`

Runs the full detection pipeline locally on the edge device:
1. Opens the configured source (`camera`, `video`, or `images`) via `frame_source.py`.
2. For each frame, runs ONNX inference (via `infer/`).
3. Creates a `StatsEvent` and pushes it to the statistics API via `transport/stats_http.py`.
4. If `save_annotated = true`, renders bounding boxes on the frame and saves to `outputs/`.
5. Stops after `max_frames` frames (0 = run forever / until source exhausted).

### `deploy/edge_stream.py` — `run_edge_stream_deploy(cfg, run_ctx)`

Stream mode: instead of running inference locally, sends each JPEG-encoded frame to the
remote inference server (`deploy/remote/`) via `transport/frame_http.py`.

1. Encodes the frame as JPEG with `jpeg_quality` compression.
2. Base64-encodes the JPEG and POSTs it to `deploy.edge.stream_endpoint`.
3. Receives back a list of detections.
4. Pushes stats as in local mode.

### `deploy/edge_llm.py` — `run_edge_llm_deploy(cfg, run_ctx)`

LLM mode: sends frames to an OpenAI-compatible vision LLM API for inference.
Follows the same frame-source and stats-push flow as local mode, but calls the LLM client
for predictions.

### `deploy/remote_server.py` — `run_remote_deploy(cfg, run_ctx)`

Starts an HTTP server (using `FastAPI` + `uvicorn`) that listens for incoming frame payloads
from edge devices in stream mode.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/frame` | Accept base64 JPEG, run ONNX inference, return detections |

The `/api/v1/frame` endpoint:
1. Validates `X-API-Key` header against `deploy.remote.ingest_api_key`.
2. Decodes the base64 JPEG frame.
3. Runs ONNX inference.
4. Creates and pushes a `StatsEvent`.
5. Returns a JSON list of detection dicts.

### `deploy/edge_common.py`

Shared helpers for edge deploy modes: result formatting, annotated frame saving.

---

## `export/`

### `export/onnx_export.py` — `export_model(model_path, output_dir, cfg)`

Converts a trained model (`.pt` for YOLO, PyTorch state dict for Faster-RCNN) to ONNX,
then optionally applies INT-8 dynamic quantization.

**Export steps:**
1. Export full-precision `.onnx` at `export.opset` (default: 17).
2. If `export.quantize = true`, apply `onnxruntime.quantization.quantize_dynamic()`
   with `weight_type=QuantType.QInt8` and calibration samples = `export.calib_samples`.
3. Save as `model-int8.onnx`.

**Note:** Only `quantize_mode = "dynamic"` is currently implemented.  Static quantization
would require a calibration dataset and is explicitly blocked by the validator.

---

## `transport/`

Pure-stdlib HTTP transport helpers (no `requests` dependency).

### `transport/stats_http.py`

```python
def push_stats_event(event: StatsEvent, endpoint: str, api_key: str, timeout_sec: float) → None
```

Serializes `event` to JSON and POSTs to `endpoint`.  Sets `X-API-Key` header if `api_key`
is non-empty.  Raises `TransportError` on non-200 HTTP status or network failure.

### `transport/frame_http.py`

```python
def encode_jpeg_base64(frame_bgr, jpeg_quality: int) → str
def decode_jpeg_base64(encoded: str) → frame_bgr
def post_json(endpoint: str, payload: dict, timeout_sec: float, api_key: str) → dict
```

Used by edge stream mode and the remote server to exchange frames and detection results.
Requires `opencv-python` for JPEG encode/decode.

---

## `statistics/`

### `statistics/sqlite_store.py`

SQLite-backed persistence for `StatsEvent` records.

| Function | Description |
|----------|-------------|
| `init_stats_db(db_path)` | Create DB and `stats_events` table if not present |
| `insert_stats_event(db_path, event)` | Serialize and insert one event |
| `get_recent_events(db_path, limit)` | Return up to `limit` most recent events as dicts |
| `get_overview(db_path)` | Return aggregate: `events_total`, `detections_total`, `avg_latency_ms`, `source_count`, `last_event_ts_utc` |
| `get_class_totals(db_path, limit)` | Return `{class_name: total_count}` over last `limit` events |

`counts_by_class` is stored as a JSON string and deserialized on read.

---

## `llm/`

### `llm/client.py` — `LLMClient`

An OpenAI-compatible vision LLM API client.

**Constructor parameters:**

```python
LLMClient(
    base_url:          str,
    model:             str,
    api_key:           str,       # read from env var named by api_key_env
    timeout_sec:       float,
    max_retries:       int,
    retry_backoff_sec: float,
    qps_limit:         float,     # queries per second
)
```

**Key method:**

```python
def send_image(self, image_b64: str, prompt: str) → str
```

Sends a base64-encoded image + prompt to the vision LLM and returns the text response.
Applies rate limiting (QPS) and retry with exponential backoff internally.

---

## `media/`

### `media/frame_source.py` — `FrameSource`

An iterator/generator that yields OpenCV BGR frames from a configured source.

| Source type | Config | Description |
|------------|--------|-------------|
| `camera` | `deploy.edge.source = "camera"` | OpenCV VideoCapture from device index `camera_id` |
| `video` | `deploy.edge.source = "video"` | OpenCV VideoCapture from `video_path` |
| `images` | `deploy.edge.source = "images"` | Sorted glob of `.jpg`, `.jpeg`, `.png` from `images_dir` |

FPS limiting is applied via a sleep-based throttle using `fps_limit`.
If `max_frames > 0`, the generator stops after that many frames.

### `media/preview.py`

Renders detected bounding boxes and class labels onto a BGR frame using OpenCV's drawing API.
Returns an annotated copy of the frame (does not modify in-place).

---

## `utils/`

### `utils/logging.py` — `StructuredLogger`

Thread-safe JSONL logger.

```python
logger = StructuredLogger(log_path=Path("work-dir/log.txt"), level="INFO")

logger.debug("event.name", "Message", extra_field=value)
logger.info("event.name", "Message", extra_field=value)
logger.warn("event.name", "Message", extra_field=value)
logger.error("event.name", "Message", extra_field=value)
```

Output format (one JSON per line):

```json
{
  "ts_utc":     "2025-01-01T12:00:00.000000+00:00",
  "level":      "INFO",
  "event":      "train.start",
  "message":    "Training started",
  "run_id":     "exp001-20250101-120000",
  "backend":    "yolo"
}
```

- Log level filtering: messages below `min_level` are discarded without file I/O.
- Thread safety: all file writes are protected by `threading.Lock`.
- Parent directory creation: `log_path.parent.mkdir(parents=True, exist_ok=True)` on init.

