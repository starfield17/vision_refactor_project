# Vision Refactor Project

A fully modular, self-contained computer-vision platform covering the complete ML lifecycle:
**Train → AutoLabel → Deploy (Edge / Remote) → Statistics**.

It is designed to be independent of any legacy codebase, with clean separation between pipeline
modules, a single unified configuration system, and structured JSONL logging throughout.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Modules](#modules)
   - [train](#train)
   - [autolabel](#autolabel)
   - [deploy/edge](#deployedge)
   - [deploy/remote](#deployremote)
   - [deploy/statistics](#deploystatistics)
8. [Common CLI Flags](#common-cli-flags)
9. [Run Outputs & Artifacts](#run-outputs--artifacts)
10. [Logging](#logging)
11. [Class Map](#class-map)
12. [Scripts](#scripts)
13. [Gitignore Conventions](#gitignore-conventions)

---

## Project Overview

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `train` | Train a detection model (YOLO or Faster-RCNN), export to ONNX |
| 2 | _(export)_ | Embedded in training: INT-8 dynamic quantization via ONNX Runtime |
| 3 | `autolabel` | Auto-annotate an unlabeled image folder using a local model or an LLM API |
| 4 | `deploy/statistics` | Ingest and visualize per-frame detection telemetry via REST API + Streamlit UI |
| 5 | `deploy/edge` | Run inference on camera / video / image folder, push stats to the statistics service |
| 5 | `deploy/remote` | Host a REST inference server that edge devices can stream frames to |

All modules share a single TOML config file, a common `share/` library, and the `work-dir/`
runtime directory.  No module may import from any other top-level module — all shared logic
lives in `share/`.

---

## Repository Layout

```
vision-refactor-project/
├── README.md                  ← this file (user guide)
├── README_DEV.md              ← developer guide (architecture, constraints, CI)
├── pyproject.toml             ← package metadata & pip dependencies
│
├── train/                     ← CLI entry-point: model training
│   ├── __init__.py
│   ├── cli.py                 ← python -m train.cli
│   └── web.py                 ← python -m train.web (Streamlit, 7794)
│
├── autolabel/                 ← CLI entry-point: automatic labeling
│   ├── __init__.py
│   ├── cli.py                 ← python -m autolabel.cli
│   └── web.py                 ← python -m autolabel.web (Streamlit, 7795)
│
├── deploy/                    ← CLI entry-points: deployment services
│   ├── edge/
│   │   ├── cli.py             ← python -m deploy.edge.cli
│   ├── remote/
│   │   ├── cli.py             ← python -m deploy.remote.cli
│   └── statistics/
│       ├── api.py             ← python -m deploy.statistics.api   (FastAPI, port 7797)
│       └── ui.py              ← python -m deploy.statistics.ui    (Streamlit, port 7796)
│
├── share/                     ← Shared library (imported by all modules above)
│   ├── config/
│   │   ├── config_loader.py   ← load, merge, validate, resolve paths, serialize TOML
│   │   └── schema.py          ← DEFAULT_CONFIG + validate_config()
│   ├── kernel/                ← Core pipeline implementations
│   │   ├── kernel.py          ← VisionKernel: orchestrates run contexts & pipelines
│   │   ├── registry.py        ← KernelRegistry: plugin registry for trainers/deployers
│   │   ├── trainer/           ← yolo.py, faster_rcnn.py
│   │   ├── infer/             ← local_yolo.py, faster_rcnn.py
│   │   ├── autolabel/         ← model_autolabel.py, llm_autolabel.py
│   │   ├── deploy/            ← edge_local.py, edge_stream.py, edge_llm.py, remote_server.py
│   │   ├── export/            ← onnx_export.py
│   │   ├── transport/         ← stats_http.py, frame_http.py
│   │   ├── statistics/        ← sqlite_store.py
│   │   ├── llm/               ← client.py  (OpenAI-compatible API wrapper)
│   │   ├── media/             ← frame_source.py, preview.py
│   │   └── utils/             ← logging.py (StructuredLogger)
│   └── types/
│       ├── detection.py       ← Detection dataclass + validation
│       ├── label.py           ← LabelRecord dataclass + validation
│       ├── stats.py           ← StatsEvent dataclass + validation
│       └── errors.py          ← Project-wide exception hierarchy
│
├── scripts/                   ← Shell utility & operations scripts
│   ├── README.md
│   ├── start_stats.sh         ← One-shot start statistics API + UI
│   ├── stop_stats.sh
│   ├── status_stats.sh
│   ├── restart_stats.sh
│   ├── add_to_systemd.sh
│   ├── add_to_systemd_bin.sh
│   ├── get_frame.sh           ← Grab a single frame from a source
│   ├── rescan.sh
│   ├── proxy.sh
│   └── change_pip_conda_source.sh
│
└── work-dir/                  ← Runtime data directory (NOT committed to Git)
    ├── README.md
    ├── config.example.toml    ← Template — copy to config.toml before running
    ├── datasets/
    ├── models/
    ├── runs/
    ├── outputs/
    ├── stats/
    └── tmp/
```

---

## Requirements

- **Python ≥ 3.11**
- Packages (installed via `pip install -e .`):
  - `ultralytics >= 8.4.0` — YOLO training & inference
  - `opencv-python >= 4.9.0` — frame capture and JPEG encode/decode
  - `onnx >= 1.16.0` — model export
  - `onnxruntime >= 1.18.0` — ONNX inference + INT-8 quantization
  - `onnxconverter-common >= 1.14.0` — ONNX conversion helpers
  - `fastapi >= 0.115.0` — statistics REST API
  - `uvicorn >= 0.30.0` — ASGI server for FastAPI
  - `streamlit >= 1.40.0` — statistics UI
- Optional (auto-detected at runtime):
  - `plotly` — enhanced charts in statistics UI
  - `torchvision` — required for Faster-RCNN training/inference

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/starfield17/vision_refactor_project.git
cd vision-refactor-project

# 2. Prepare your local configuration
cp work-dir/config.example.toml work-dir/config.toml
# Then edit work-dir/config.toml to match your environment
```

---

## Quick Start

### Step 1 — Edit your config

Open `work-dir/config.toml` and at minimum configure:

```toml
[class_map]
names = ["cat", "dog"]
id_map = { cat = 0, dog = 1 }

[workspace]
run_name = "my-first-run"

[train]
backend = "yolo"
device = "cuda"       # or "cpu"
epochs = 50

[train.yolo]
weights = "./weights/yolov8n.pt"

[data]
yolo_dataset_dir = "./work-dir/datasets/yolo"
```

### Step 2 — Train

```bash
python -m train.cli \
  --config ./work-dir/config.toml \
  --workdir ./work-dir
```

Output is written to `work-dir/runs/<run-id>/`.

### Step 3 — AutoLabel unlabeled images

```bash
# Using the trained model
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=model \
  --set autolabel.model.onnx_model=./work-dir/models/exp001/model-int8.onnx
```

### Optional — Web UIs for Train/AutoLabel

```bash
# Train web: http://localhost:7794
python -m train.web --config ./work-dir/config.toml --workdir ./work-dir

# AutoLabel web: http://localhost:7795
python -m autolabel.web --config ./work-dir/config.toml --workdir ./work-dir
```

Both web pages provide a `Device` selector (`cpu` or `gpu`), where `gpu` maps to `train.device=cuda:0`.

### Step 4 — Start the statistics service

```bash
# Terminal A: API (receives telemetry)
python -m deploy.statistics.api --config ./work-dir/config.toml

# Terminal B: UI (http://localhost:7796)
python -m deploy.statistics.ui --config ./work-dir/config.toml
```

Or use the convenience script:

```bash
bash scripts/start_stats.sh --config ./work-dir/config.toml
```

### Step 5 — Deploy on edge

```bash
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=local \
  --set deploy.edge.source=images \
  --set deploy.edge.images_dir=./work-dir/datasets/smoke/images
```

---

## Configuration

All modules share a single `work-dir/config.toml`.  The file format is **TOML**.

### Key Sections

| Section | Purpose |
|---------|---------|
| `[app]` | Schema version (must be `1`) |
| `[workspace]` | Working directory, run name, log file, log level |
| `[class_map]` | Object class names and integer IDs |
| `[data]` | Dataset directory paths |
| `[train]` | Training backend, device, hyperparameters |
| `[train.yolo]` | YOLO-specific: pretrained weights path |
| `[train.faster_rcnn]` | Faster-RCNN-specific: variant, lr, momentum, etc. |
| `[export]` | ONNX export settings, quantization |
| `[autolabel]` | Autolabel mode (model/llm), confidence, conflict handling |
| `[autolabel.llm]` | LLM API endpoint, model, prompt, rate limiting |
| `[autolabel.model]` | ONNX model path for model-based autolabeling |
| `[deploy.edge]` | Edge inference: source type, model, FPS, stats endpoint |
| `[deploy.edge.llm]` | LLM settings when edge mode = `llm` |
| `[deploy.remote]` | Remote server: listen host/port, model, ingest API key |
| `[deploy.statistics]` | API/UI ports, SQLite DB path, API key, rate limit |
| `[logging]` | Enable/disable JSONL output |
| `[statistics]` | DB flush interval |

### Relative Path Resolution

All path fields (e.g., `train.yolo.weights`, `deploy.edge.local_model`) are resolved
**relative to the directory containing `config.toml`**.  If `config.toml` lives inside
`work-dir/`, paths are resolved from the repository root for historical compatibility.

### Runtime Overrides

Any config key can be overridden at the command line without editing the file:

```bash
python -m train.cli --config ./work-dir/config.toml \
  --set train.epochs=100 \
  --set train.device=cuda \
  --set workspace.run_name=experiment42
```

Dotted key paths are supported for any nesting depth (`section.subsection.key=value`).

---

## Modules

### `train`

**Entry point:** `python -m train.cli`

Trains an object-detection model and exports it to ONNX (with optional INT-8 quantization).

**Backends:**

| Backend | Config key | Notes |
|---------|-----------|-------|
| `yolo` | `train.backend = "yolo"` | Requires `ultralytics`, `data.yolo_dataset_dir` |
| `faster_rcnn` | `train.backend = "faster_rcnn"` | Requires `torchvision` |

**Key config fields:**

```toml
[train]
backend    = "yolo"         # "yolo" | "faster_rcnn"
device     = "cuda"         # "cpu" | "cuda" | "mps"
seed       = 42
epochs     = 50
batch_size = 16
img_size   = 640
dry_run    = false          # If true, skips actual training (useful for testing config)

[train.yolo]
weights = "./weights/yolov8n.pt"   # Pretrained checkpoint

[export]
onnx          = true
opset         = 17
quantize      = true
quantize_mode = "dynamic"   # Only "dynamic" is currently supported
calib_samples = 32
```

**Artifacts produced:**

| Path | Content |
|------|---------|
| `work-dir/runs/<run-id>/metrics.json` | Run ID, status, elapsed ms, errors |
| `work-dir/runs/<run-id>/artifacts.json` | Backend, model paths, status |
| `work-dir/runs/<run-id>/config.resolved.toml` | Snapshot of resolved config used |
| `work-dir/config.resolved.toml` | Symlink/copy of most recent resolved config |

---

### `autolabel`

**Entry point:** `python -m autolabel.cli`

Scans unlabeled images and generates YOLO-format label files automatically.

**Modes:**

| Mode | Config key | Description |
|------|-----------|-------------|
| `model` | `autolabel.mode = "model"` | Local ONNX model inference |
| `llm` | `autolabel.mode = "llm"` | OpenAI-compatible vision LLM API |

**Key config fields:**

```toml
[autolabel]
mode        = "model"   # "model" | "llm"
confidence  = 0.5       # Detection confidence threshold (0–1)
batch_size  = 2         # Images per batch
visualize   = true      # Save annotated preview images
on_conflict = "skip"    # "skip" | "overwrite" | "merge"

[autolabel.model]
onnx_model = "./work-dir/models/exp001/model-int8.onnx"
backend    = "yolo"

[autolabel.llm]
base_url      = "https://api.openai.com/v1"
model         = "gpt-4o"
api_key_env   = "OPENAI_API_KEY"      # name of env var holding the key
prompt        = "List all objects..."
timeout_sec   = 60.0
max_retries   = 2
retry_backoff_sec = 1.5
qps_limit     = 1.0                   # queries per second
max_images    = 0                     # 0 = no limit
```

`on_conflict` controls behaviour when a label file already exists for an image:
- `skip` — do not overwrite existing labels
- `overwrite` — replace existing labels
- `merge` — merge new detections into existing label file

---

### `deploy/edge`

**Entry point:** `python -m deploy.edge.cli`

Runs detection on a local image source (camera, video file, or image folder) and pushes
telemetry to the statistics API.

**Modes:**

| Mode | Description |
|------|-------------|
| `local` | ONNX inference runs on the edge device itself |
| `stream` | Raw JPEG frames are streamed to a remote inference server |
| `llm` | Frames are sent to an OpenAI-compatible vision LLM API |

**Key config fields:**

```toml
[deploy.edge]
source_id      = "edge-001"       # Unique identifier for this edge device
mode           = "local"          # "local" | "stream" | "llm"
source         = "images"         # "camera" | "video" | "images"
camera_id      = 0
video_path     = ""
images_dir     = "./work-dir/datasets/smoke/images"
fps_limit      = 10               # Max frames per second
jpeg_quality   = 80               # JPEG quality (1–100) for stream mode
confidence     = 0.5
local_model    = "./work-dir/models/exp001/model-int8.onnx"
stats_endpoint = "http://127.0.0.1:7797/api/v1/push"
api_key        = ""               # Optional API key for statistics endpoint
stats_timeout_sec = 2.0
save_annotated = true             # Save annotated frames to work-dir/outputs/
max_frames     = 0                # 0 = process all frames
stream_endpoint   = "http://127.0.0.1:60051/api/v1/frame"
stream_timeout_sec = 5.0
stream_api_key    = ""
```

---

### `deploy/remote`

**Entry point:** `python -m deploy.remote.cli`

Hosts an HTTP inference server.  Edge devices running in `stream` mode POST base64-encoded
JPEG frames to this server, which performs ONNX inference and returns detection results.

**Key config fields:**

```toml
[deploy.remote]
source_id      = "remote-001"
listen_host    = "0.0.0.0"
listen_port    = 60051
model          = "./work-dir/models/exp001/model.onnx"
confidence     = 0.5
stats_endpoint = "http://127.0.0.1:7797/api/v1/push"
api_key        = ""
ingest_api_key = ""              # Key required from edge clients
stats_timeout_sec = 2.0
save_annotated = false
max_payload_mb = 8               # Max accepted request size in MB
```

---

### `deploy/statistics`

**API entry point:** `python -m deploy.statistics.api`
**UI entry point:** `python -m deploy.statistics.ui`

A two-service observability stack:

- **API** (`FastAPI`, default port `7797`): receives `StatsEvent` JSON payloads from edge/remote
  deployers via `POST /api/v1/push`.  Supports per-source rate limiting and optional API key
  authentication.  Storage backend: SQLite.
- **UI** (`Streamlit`, default port `7796`): real-time analytics dashboard with latency trend,
  detection count trend, class distribution donut chart, per-source breakdown, and raw event log.
  Supports source filtering, minimum-detection threshold filter, and auto-refresh.

**Key config fields:**

```toml
[deploy.statistics]
ui_port          = 7796
api_port         = 7797
storage          = "sqlite"
db_path          = "./work-dir/stats/stats.db"
public_host      = "0.0.0.0"
api_key          = ""            # If non-empty, X-API-Key header required on POST
rate_limit_per_sec = 0           # 0 = unlimited; per source_id

[statistics]
flush_interval_sec = 5
```

**API Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check; returns storage info |
| `POST` | `/api/v1/push` | Ingest a `StatsEvent` payload |

**StatsEvent JSON schema:**

```json
{
  "schema_version": 1,
  "source_id": "edge-001",
  "ts_utc": "2025-01-01T12:00:00+00:00",
  "total_detections": 3,
  "counts_by_class": { "Kitchen_waste": 2, "Recyclable_waste": 1 },
  "latency_ms": 45.2
}
```

---

## Common CLI Flags

All CLIs (`train.cli`, `autolabel.cli`, `deploy.edge.cli`, `deploy.remote.cli`,
`deploy.statistics.api`, `deploy.statistics.ui`) share these flags:

| Flag | Required | Description |
|------|----------|-------------|
| `--config PATH` | Yes | Path to `config.toml` |
| `--workdir PATH` | No | Override `workspace.root` from config |
| `--set KEY=VALUE` | No | Override any config key (repeatable) |

---

## Run Outputs & Artifacts

Every pipeline invocation (train, autolabel, deploy edge/remote) creates a unique run directory:

```
work-dir/runs/<run-name>-<YYYYMMDD-HHMMSS>/
├── metrics.json          # run_id, mode, status, elapsed_ms, error
├── artifacts.json        # backend/mode-specific details and output paths
└── config.resolved.toml  # Exact config snapshot (with all paths resolved)
```

A copy of `config.resolved.toml` is also written to `work-dir/config.resolved.toml` for
quick inspection of the last run.

Exit codes:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Pipeline failed (see `error=` in output) |
| `2` | Config validation error |
| `3` | Runtime import error (missing optional dependency) |

---

## Logging

All modules write structured **JSONL** logs to `work-dir/log.txt` (path set by
`workspace.log_file`).

Each line is a JSON object:

```json
{
  "ts_utc": "2025-01-01T12:00:00.123456+00:00",
  "level": "INFO",
  "event": "train.start",
  "message": "Train pipeline started",
  "run_id": "exp001-20250101-120000",
  "backend": "yolo"
}
```

Log level is controlled by `workspace.log_level` (one of `DEBUG`, `INFO`, `WARN`, `ERROR`).
JSONL output can be disabled by setting `logging.jsonl = false`.

---

## Class Map

Every config must define a non-empty class map with consecutively numbered IDs starting at 0:

```toml
[class_map]
names  = ["Kitchen_waste", "Recyclable_waste", "Hazardous_waste", "Other_waste"]
id_map = { Kitchen_waste = 0, Recyclable_waste = 1, Hazardous_waste = 2, Other_waste = 3 }
```

`id_map` must exactly match the order of `names` — validation will reject mismatches
or duplicate names.

---

## Scripts

Convenience shell scripts live in `scripts/`. See `scripts/README.md` for details.

Quick reference:

```bash
bash scripts/start_stats.sh   # Start statistics API + UI, with health check
bash scripts/stop_stats.sh    # Stop statistics API + UI
bash scripts/status_stats.sh  # Show process status and health
bash scripts/restart_stats.sh # stop + start
```

---

## Gitignore Conventions

The following are **never committed**:

- `work-dir/config.toml` — your local config
- `work-dir/runs/`, `work-dir/models/`, `work-dir/outputs/`, `work-dir/stats/`, `work-dir/tmp/`
  — all runtime artifacts
- `codex.md` — internal development notes

The following **are committed**:

- `work-dir/config.example.toml` — config template
- `work-dir/README.md` — runtime directory documentation
- `work-dir/.gitkeep` — keeps the directory in Git
