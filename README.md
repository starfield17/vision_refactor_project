# Vision Refactor Project

A fully modular, self-contained computer-vision platform covering the complete ML lifecycle:
**Train → AutoLabel → Deploy (Edge / Remote) → Statistics**.

It is designed to be independent of any legacy codebase, with clean separation between pipeline
modules, a single unified configuration system, structured JSONL logging, and service-backed
frontends. CLI, React, and PySide6 frontends call long-running backend APIs instead of running
pipelines directly.

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
13. [For AI Agents](#for-ai-agents)
14. [Gitignore Conventions](#gitignore-conventions)

---

## Project Overview

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `train` | Train a detection model (YOLO or Faster-RCNN), export to ONNX |
| 2 | _(export)_ | Embedded in training: ONNX export for YOLO and Faster-RCNN; YOLO supports optional INT-8 dynamic quantization |
| 3 | `autolabel` | Auto-annotate an unlabeled image folder using a local model or an LLM API |
| 4 | `deploy/statistics` | Ingest/query per-frame telemetry through the deploy-statistics backend and React dashboard |
| 5 | `deploy/edge` | Run inference on camera / video / image folder, push stats to the statistics service |
| 5 | `deploy/remote` | Host a REST inference server that edge devices can stream frames to |

All modules share a single TOML config file, a common `share/` library, and the `work-dir/`
runtime directory. Runtime work is coordinated by two backend services:
`services.train_autolabel` and `services.deploy_statistics`. No module may import from any
other top-level module — all shared logic lives in `share/`.

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
│   └── web.py                 ← compatibility notice; React app lives in web/train_autolabel
│   └── launch_gui.py          ← python -m train.launch_gui
│
├── autolabel/                 ← CLI entry-point: automatic labeling
│   ├── __init__.py
│   ├── cli.py                 ← python -m autolabel.cli
│   └── web.py                 ← compatibility notice; React app lives in web/train_autolabel
│   └── launch_gui.py          ← python -m autolabel.launch_gui
│
├── deploy/                    ← CLI entry-points: deployment services
│   ├── edge/
│   │   ├── cli.py             ← python -m deploy.edge.cli
│   ├── remote/
│   │   ├── cli.py             ← python -m deploy.remote.cli
│   └── statistics/
│       ├── api.py             ← compatibility launcher for services.deploy_statistics.api
│       └── ui.py              ← compatibility notice; React app lives in web/deploy_statistics
│
├── services/                  ← FastAPI backend daemons for train/autolabel and deploy/statistics
├── web/                       ← React frontends: train_autolabel and deploy_statistics
├── share/                     ← Shared library (imported by all modules above)
│   ├── config/
│   │   ├── config_loader.py   ← load, merge, validate, resolve paths, serialize TOML
│   │   └── schema.py          ← DEFAULT_CONFIG + validate_config()
│   ├── application/           ← API clients, job store/runner, and service helpers
│   ├── desktop/               ← PySide6 desktop GUI
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
│   ├── README_scripts.md
│   ├── start_stats.sh         ← Start deploy/statistics backend + React UI
│   ├── stop_stats.sh
│   ├── status_stats.sh
│   ├── restart_stats.sh
│   ├── start_train_autolabel.sh ← Start train/autolabel backend + React UI
│   ├── stop_train_autolabel.sh
│   ├── add_to_systemd.sh
│   ├── add_to_systemd_bin.sh
│   ├── get_frame.sh           ← Grab a single frame from a source
│   ├── rescan.sh
│   ├── proxy.sh
│   └── change_pip_conda_source.sh
│
└── work-dir/                  ← Runtime data directory (NOT committed to Git)
    ├── README_work-dir.md
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
  - `fastapi >= 0.115.0` — backend service APIs
  - `uvicorn >= 0.30.0` — ASGI server for FastAPI
- Node.js + npm — React web frontends under `web/`
- `PySide6 >= 6.8.0` — local desktop GUI for Train/AutoLabel
- Optional (auto-detected at runtime):
  - `torchvision` — required for Faster-RCNN training/inference
  - `onnxruntime-gpu` — enables CUDAExecutionProvider for ONNX inference when the matching CUDA runtime libraries are installed

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/starfield17/vision_refactor_project.git
cd vision-refactor-project

# 2. Install runtime dependencies
pip install -r train/requirements.txt      # train.cli / train.web / train.launch_gui
pip install -r autolabel/requirements.txt  # autolabel.cli / autolabel.web / autolabel.launch_gui
pip install -r deploy/requirements.txt     # deploy.edge / deploy.remote / deploy.statistics

# 3. Prepare your local configuration
cp work-dir/config.example.toml work-dir/config.toml
# Then edit work-dir/config.toml to match your environment

# 4. Install frontend dependencies when using the React web apps
npm --prefix web/train_autolabel install
npm --prefix web/deploy_statistics install
```

The root `pyproject.toml` remains the shared package metadata. These module-local
`requirements.txt` files are convenience runtime dependency sets for running the
repo directly from source, and they keep the shared version floors from
`pyproject.toml` where those packages are already declared.

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

### Step 2 — Start Backend Services

```bash
# Terminal A: train/autolabel backend, default http://127.0.0.1:7793
python -m services.train_autolabel.api --config ./work-dir/config.toml

# Terminal B: deploy/statistics backend, default http://127.0.0.1:7797
python -m services.deploy_statistics.api --config ./work-dir/config.toml
```

### Step 3 — Train

```bash
python -m train.cli \
  --config ./work-dir/config.toml \
  --workdir ./work-dir
```

The CLI submits a backend job and waits by default. Output is written to
`work-dir/runs/<run-id>/`.

### Step 4 — AutoLabel unlabeled images

```bash
# Using the trained model
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=model \
  --set autolabel.model.onnx_model=./work-dir/models/exp001/model-int8.onnx
```

### Optional — React Web UIs

```bash
# Train + Autolabel web: http://localhost:7794
npm --prefix web/train_autolabel run dev

# Deploy + Statistics web: http://localhost:7796
npm --prefix web/deploy_statistics run dev
```

The React apps call the backend service APIs. Configure `VITE_TRAIN_AUTOLABEL_API_URL`,
`VITE_TRAIN_AUTOLABEL_API_TOKEN`, `VITE_DEPLOY_STATISTICS_API_URL`, or
`VITE_DEPLOY_STATISTICS_API_TOKEN` when the defaults are not suitable.

**Health check notes:**

- Backend API root path (`http://127.0.0.1:7793/` or `http://127.0.0.1:7797/`) may return
  `404` — this is normal. Use `/health` to check liveness.
- `scripts/start_train_autolabel.sh` and `scripts/start_stats.sh` handle startup, health
  polling, and PID-file management for both the API and the React dev server. They source
  `work-dir/secrets/llm.env` if it exists.

### LLM Key Configuration

When using LLM features (autolabel LLM mode or edge LLM mode), keep API keys out of tracked
config. Use the `api_key_env_name` field to point to an environment variable:

```toml
[autolabel.llm]
api_key_env_name = "VISION_LLM_API_KEY"
```

Then set the key at runtime:

```bash
export VISION_LLM_API_KEY='your-key'
```

For persistent local runs, prefer a gitignored env file:

```bash
mkdir -p work-dir/secrets
printf 'VISION_LLM_API_KEY=your-key\n' > work-dir/secrets/llm.env
chmod 600 work-dir/secrets/llm.env
source work-dir/secrets/llm.env
```

`work-dir/secrets/llm.env` is gitignored. Never paste real keys into tracked docs or config.

### Optional — Local Desktop GUI for Train/AutoLabel

```bash
# Open the shared desktop app with the Train page selected
python -m train.launch_gui

# Open the same desktop app with the AutoLabel page selected
python -m autolabel.launch_gui
```

The desktop GUI is a local PySide6 application. It submits and monitors backend jobs
through the same HTTP APIs used by CLI and React.

### Step 5 — Start the deploy/statistics service and dashboard

```bash
# Starts services.deploy_statistics.api and the React dashboard dev server
bash scripts/start_stats.sh --config ./work-dir/config.toml
```

### Step 6 — Deploy on edge

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
| `[deploy.statistics]` | Legacy UI port, SQLite stats DB path, stats ingest API key, rate limit |
| `[services.train_autolabel]` | Train/autolabel backend host, port, token, job DB |
| `[services.deploy_statistics]` | Deploy/statistics backend host, port, token, job DB |
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

For `train.cli` and `autolabel.cli`, you can also persist CLI changes into
`config.toml`:

```bash
python -m train.cli \
  --config ./work-dir/config.toml \
  --epochs 50 \
  --backend yolo \
  --save-config --config-only
```

`--config-only` applies updates and exits; omit it to save and run in one command.

---

## Modules

### `train`

**Entry point:** `python -m train.cli`

Trains an object-detection model and exports it to ONNX. YOLO can also produce an optional
INT-8 dynamic-quantized ONNX model. Faster-RCNN exports a full-precision ONNX model through
`share/kernel/export/faster_rcnn_onnx_export.py`; quantization is currently skipped for that
backend.

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
backend    = "yolo"   # "yolo" | "faster_rcnn"

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

LLM credentials should be provided through `autolabel.llm.api_key_env_name` or a local
environment file such as `work-dir/secrets/llm.env`. Keep API keys out of tracked config.

---

### `deploy/edge`

**Entry point:** `python -m deploy.edge.cli`

Runs detection on a local image source (camera, video file, or image folder) and pushes
telemetry to the statistics API.

**Modes:**

| Mode | Description |
|------|-------------|
| `local` | Backend-aware inference runs on the edge device itself |
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
JPEG frames to this server, which performs backend-aware model inference and returns
detection results.

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

**Backend entry point:** `python -m services.deploy_statistics.api`
**Compatibility API entry point:** `python -m deploy.statistics.api`
**React UI:** `npm --prefix web/deploy_statistics run dev`

The deploy/statistics backend is a single FastAPI service. It receives telemetry, serves
dashboard queries, manages deploy edge jobs, and can start/stop the remote frame inference
runtime used by stream mode. The React dashboard calls this API; it does not read SQLite directly.

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
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/push` | Ingest a `StatsEvent` payload |
| `GET` | `/api/v1/statistics/dashboard` | Dashboard view model |
| `GET` | `/api/v1/statistics/events` | Recent events |
| `POST` | `/api/v1/deploy/edge/jobs` | Submit edge deploy job |
| `POST` | `/api/v1/deploy/remote/start` | Start remote frame runtime |
| `POST` | `/api/v1/deploy/remote/stop` | Stop remote frame runtime |
| `POST` | `/api/v1/frame` | Remote frame inference endpoint |

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
`deploy.statistics.api`, `deploy.statistics.ui`) share these local config flags. User-facing
CLIs also accept `--api-url`, `--api-token`, `--no-wait`, `--poll-sec`, and
`--wait-timeout-sec` where job submission is supported:

| Flag | Required | Description |
|------|----------|-------------|
| `--config PATH` | Yes | Path to `config.toml` |
| `--workdir PATH` | No | Override `workspace.root` from config |
| `--set KEY=VALUE` | No | Override any config key (repeatable) |

Train and AutoLabel CLIs additionally support granular section flags (for example
`--epochs`, `--backend`, `--mode`, `--llm-base-url`) plus:

| Flag | Required | Description |
|------|----------|-------------|
| `--save-config` | No | Persist current granular flags and `--set` values to `config.toml` |
| `--config-only` | No | Only update `config.toml`, do not execute the pipeline (requires `--save-config`) |

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

Convenience shell scripts live in `scripts/`. See `scripts/README_scripts.md` for details.

Quick reference:

```bash
bash scripts/start_stats.sh   # Start statistics API + UI, with health check
bash scripts/stop_stats.sh    # Stop statistics API + UI
bash scripts/status_stats.sh  # Show process status and health
bash scripts/restart_stats.sh # stop + start
bash scripts/start_train_autolabel.sh # Start train/autolabel API + UI
bash scripts/stop_train_autolabel.sh  # Stop train/autolabel API + UI

python scripts/prepare_voc_detection_dataset.py --workdir ./work-dir \
  --max-train 1500 --max-val 500 --max-unlabeled 30 --max-deploy 30
```

`prepare_voc_detection_dataset.py` downloads VOC2007 and creates YOLO training data,
Faster-RCNN `LabelRecord` JSON labels, unlabeled images for AutoLabel, and image-folder
inputs for deploy smoke runs under `work-dir/datasets/`.

---

---

## For AI Agents

> Target Audience: Codex / Claude Code / Other Code Agents
>
> Goal: Help users **quickly complete train / autolabel / deploy / statistics
> configuration and execution**, not architectural development.
>
> Language Convention: Prioritize using Chinese to communicate with users; commands,
> paths, and configuration keys use original English.

### Agent Working Method

When users request "help me run/configure/deploy/train/autolabel", prioritize this
workflow:

1. **Don't modify code first** — check if only configuration or commands need changes.
2. Prioritize reading:
   - `work-dir/config.toml`
   - `README.md`
   - `scripts/README_scripts.md`
3. If the task is execution-related, prioritize confirming:
   - Is Python environment available
   - Are dependencies installed
   - Do configuration paths exist
   - Are model paths / data paths / ports correct
4. Unless the user explicitly wants to develop new features, prioritize using:
   - `--set KEY=VALUE`
   - `--save-config`
   - Editing `work-dir/config.toml`
5. If the user's goal is "quick verification", prioritize:
   - `images` as input source
   - `max_frames=1` or few frames
   - Existing models rather than retraining
6. If the user's goal is "real deployment", prioritize checking:
   - Is Statistics started
   - Are `stats_endpoint` / `stream_endpoint` reachable
   - Does API key come from environment variables
7. If the user's goal is using WebUI, prioritize starting the API and UI scripts:
   - `scripts/start_train_autolabel.sh`
   - `scripts/start_stats.sh`
   then open the UI port, not the backend root path.

**Don't do these by default:**

- Don't refactor code by default
- Don't add features by default
- Don't run long training sessions by default
- Don't overwrite user's existing configuration by default, unless explicitly agreed

### Key Directory Quick Reference

- Configuration file: `work-dir/config.toml`
- Configuration template: `work-dir/config.example.toml`
- Logs: `work-dir/log.txt`
- Training artifacts: `work-dir/models/<run_name>/`
- Run records: `work-dir/runs/<run_id>/`
- Statistics database: `work-dir/stats/stats.db`
- Output images/annotated images: `work-dir/outputs/`
- Service PID/log files: `work-dir/tmp/*.pid`, `work-dir/tmp/*.log`
- Train + AutoLabel WebUI: `web/train_autolabel`
- Deploy + Statistics WebUI: `web/deploy_statistics`

Key items to check after training:

- `model.pt`
- `model.onnx`
- `model-int8.onnx` (YOLO quantized export when enabled)
- `model_manifest.json`

### Pre-Run Checklist

Before executing any train / autolabel / deploy, verify these items:

**Environment:**

```bash
python --version                    # requires Python ≥ 3.11
python -m pip show vision-refactor-project
```

If the project is not installed:

```bash
pip install -e .
```

**Configuration file:**

```bash
# Confirm existence
ls work-dir/config.toml

# If missing, copy from template
cp work-dir/config.example.toml work-dir/config.toml
```

**Key config items to verify:**

- `workspace.run_name`
- `class_map.names` / `class_map.id_map`
- `train.backend`
- `data.yolo_dataset_dir`
- `autolabel.model.onnx_model`
- `deploy.edge.local_model`
- `deploy.remote.model`
- `deploy.edge.stats_endpoint`
- `deploy.edge.stream_endpoint`
- `deploy.statistics.db_path`

**LLM key fields:**

Three fields exist; `api_key_env_name` is the recommended one:

- `api_key_env_name` — name of env var holding the key (preferred)
- `api_key_env` — legacy compatibility
- `api_key` — plaintext (not recommended)

### Backend Services and WebUI Startup

The project follows a frontend/backend split. Backends are long-running FastAPI services;
CLI, React WebUI, and PySide6 frontends call those services over HTTP.

**Train + AutoLabel WebUI:**

Recommended startup:

```bash
PYTHON_BIN=/home/hazel/miniconda3/envs/Lab/bin/python \
  bash scripts/start_train_autolabel.sh --config ./work-dir/config.toml
```

Stop:

```bash
PYTHON_BIN=/home/hazel/miniconda3/envs/Lab/bin/python \
  bash scripts/stop_train_autolabel.sh --config ./work-dir/config.toml
```

Default addresses:

| Service | URL |
|---------|-----|
| UI | `http://127.0.0.1:7794` |
| API health | `http://127.0.0.1:7793/health` |

API root `http://127.0.0.1:7793/` returns `404` — use `/health` for liveness checks.

The startup script:
- Starts `services.train_autolabel.api`
- Starts Vite from `web/train_autolabel`
- Sources `work-dir/secrets/llm.env` if it exists
- Writes PID files under `work-dir/tmp/`
- Reports `external` when the port is already served by a manually started process

Manual fallback:

```bash
# Terminal A: backend API
source work-dir/secrets/llm.env 2>/dev/null || true
python -m services.train_autolabel.api --config ./work-dir/config.toml

# Terminal B: React WebUI
npm --prefix web/train_autolabel run dev
```

**Deploy + Statistics WebUI:**

Recommended startup:

```bash
bash scripts/start_stats.sh --config ./work-dir/config.toml
```

Status / stop:

```bash
bash scripts/status_stats.sh --config ./work-dir/config.toml
bash scripts/stop_stats.sh --config ./work-dir/config.toml
```

Default addresses:

| Service | URL |
|---------|-----|
| UI | `http://127.0.0.1:7796` |
| API health | `http://127.0.0.1:7797/health` |

### Quick Operation Guides

#### Train

Priority backends:
1. `yolo` — supports training, ONNX, INT-8 quantization, full deploy
2. `faster_rcnn` — supports training, FP32 ONNX, model autolabel, edge/stream/remote
   deploy; quantization is intentionally skipped

Minimum YOLO command:

```bash
python -m train.cli \
  --config ./work-dir/config.toml \
  --set workspace.run_name=my-yolo-run \
  --set train.backend=yolo \
  --set data.yolo_dataset_dir=../coco128 \
  --set train.yolo.weights=../yolo26n.pt \
  --set train.epochs=1 \
  --set train.batch_size=4 \
  --set train.img_size=320
```

For quick verification, add `--set train.dry_run=true`.

Faster R-CNN example:

```bash
python -m train.cli \
  --config ./work-dir/config.toml \
  --set workspace.run_name=my-frcnn-run \
  --set train.backend=faster_rcnn \
  --set train.faster_rcnn.variant=mobilenet_v3 \
  --set train.epochs=1 \
  --set train.batch_size=2 \
  --set export.onnx=true
```

Post-training checklist:
- `work-dir/models/<run_name>/model_manifest.json`
- `work-dir/models/<run_name>/model-int8.onnx` (YOLO quantized)
- `work-dir/models/<run_name>/model.onnx` (YOLO FP32 or Faster R-CNN FP32)
- `work-dir/runs/<run_id>/artifacts.json`
- `work-dir/runs/<run_id>/metrics.json`

To connect training results to downstream tasks, update these config keys to point at
the artifact directory containing `model_manifest.json`:
- `autolabel.model.onnx_model`
- `deploy.edge.local_model`
- `deploy.remote.model`

#### AutoLabel

**Model mode** (local ONNX inference):

```bash
# YOLO
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=model \
  --set autolabel.model.backend=yolo \
  --set autolabel.model.onnx_model=./work-dir/models/my-yolo-run/model-int8.onnx \
  --set data.unlabeled_dir=../coco128/images/train2017 \
  --set data.labeled_dir=./work-dir/datasets/labeled

# Faster R-CNN
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=model \
  --set autolabel.model.backend=faster_rcnn \
  --set autolabel.model.onnx_model=./work-dir/models/my-frcnn-run/model.onnx
```

**LLM mode** (vision LLM API):

```bash
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=llm \
  --set data.unlabeled_dir=../coco128/images/train2017 \
  --set autolabel.llm.max_images=5
```

**Conflict strategy** (`autolabel.on_conflict`):

| Value | Behavior |
|-------|----------|
| `skip` | Skip if label already exists (recommended for first run) |
| `overwrite` | Replace existing labels (recommended for explicit refresh) |
| `merge` | Merge new detections into existing label file |

#### Deploy

**Edge Local** — run inference locally, push stats to Statistics:

```bash
# YOLO
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=local \
  --set deploy.edge.source=images \
  --set deploy.edge.images_dir=../coco128/images/train2017 \
  --set deploy.edge.local_model=./work-dir/models/my-yolo-run/model-int8.onnx \
  --set deploy.edge.max_frames=1

# Faster R-CNN
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=local \
  --set deploy.edge.source=images \
  --set deploy.edge.local_model=./work-dir/models/my-frcnn-run/model.onnx \
  --set deploy.edge.max_frames=1
```

**Edge Stream + Remote** — edge sends frames, remote runs inference:

```bash
# Terminal A: start remote server
python -m deploy.remote.cli \
  --config ./work-dir/config.toml \
  --set deploy.remote.model=./work-dir/models/my-yolo-run/model.onnx

# Terminal B: start edge stream
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=stream \
  --set deploy.edge.source=images \
  --set deploy.edge.images_dir=../coco128/images/train2017 \
  --set deploy.edge.stream_endpoint=http://127.0.0.1:60051/api/v1/frame \
  --set deploy.edge.max_frames=1
```

The remote `/api/v1/frame` response includes `model_id` and `backend` — use these to
confirm which model is serving.

**Edge LLM** — edge device calls vision LLM directly:

```bash
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=llm \
  --set deploy.edge.source=images \
  --set deploy.edge.images_dir=../coco128/images/train2017 \
  --set deploy.edge.max_frames=1
```

### Common Task Templates

**"Help me quickly run local deployment":**
1. Check `work-dir/config.toml`
2. Start Statistics
3. Check `deploy.edge.local_model` exists
4. Run `deploy.edge.cli` with `images + max_frames=1`
5. Verify: command output, `work-dir/runs/<run_id>/artifacts.json`, Statistics UI

**"Help me connect this model to autolabel":**
1. Find `model_manifest.json` in the model directory
2. Use `final_infer_model_path` from the manifest (YOLO: `model-int8.onnx`, Faster R-CNN:
   `model.onnx`)
3. Write path to `autolabel.model.onnx_model`
4. Run `autolabel.mode=model` on a small sample to verify

**"Help me do remote inference deployment":**
1. Start Statistics
2. Start `deploy.remote.cli`
3. Use `deploy.edge.mode=stream` with 1-frame smoke test
4. Check remote returns `model_id`/`backend`
5. Confirm events visible in Statistics UI

### Troubleshooting

**Configuration errors** — check in order:
- CLI stderr
- `work-dir/log.txt`
- `ConfigError` messages

Common causes: `class_map.id_map` inconsistent with `class_map.names`, ONNX model path
doesn't exist, wrong `data.yolo_dataset_dir`, missing `api_key_env_name` for LLM.

**Deploy produces no results** — check:
- Is `deploy.edge.max_frames` too small
- Does `deploy.edge.source` match the corresponding path
- Is `deploy.edge.confidence` too high
- Are Remote/Statistics started

**Statistics shows no data** — check:
- `deploy.edge.stats_endpoint` / `deploy.remote.stats_endpoint`
- `scripts/status_stats.sh`
- `work-dir/stats/stats.db`

**LLM call failure** — check:
- Are environment variables exported
- Is `base_url` an OpenAI-compatible endpoint root path
- Is `prompt` empty
- Is `qps_limit` too high

### Project Facts

- **The main pipeline owner is the backend service layer.** CLI, React WebUI, and PySide6
  are API clients.
- **Configuration source is `work-dir/config.toml`.** Don't hardcode parameters.
- **For the fastest deploy path, default to YOLO.** Faster R-CNN supports FP32 ONNX deploy
  but not quantized deploy.
- **Train + AutoLabel UI (7794) and API (7793) are two processes.**
- **Statistics UI (7796) and API (7797) are two processes.**
- **Backend API root path may return 404.** Use `/health` and UI ports for browser pages.
- **Model directory after training contains `model_manifest.json`.** Use it to identify
  model identity.
- **LLM keys via environment variable name are preferred.** Don't write real keys into
  repository configuration.

### Response Style & Priority Order

When users ask "how to run", responses should include:
1. Configuration keys that need modification
2. Commands that can be directly executed
3. Where to look after a successful run
4. What to check first on failure

Priority order for "quickly complete task":
1. Adjust configuration
2. Provide accurate CLI commands
3. Do a smoke test
4. Help user interpret output and logs
5. Only modify code when explicitly needed

If the user hasn't requested developing new features, don't escalate the task into a
development task.

---

## Gitignore Conventions

The following are **never committed**:

- `work-dir/config.toml` — your local config
- `work-dir/runs/`, `work-dir/models/`, `work-dir/outputs/`, `work-dir/stats/`, `work-dir/tmp/`
  — all runtime artifacts
- model/data archives and weights such as `*.pt`, `*.onnx`, `*.zip`, `*.tar`, `*.tar.gz`
- `codex.md` — internal development notes

The following **are committed**:

- `work-dir/config.example.toml` — config template
- `work-dir/README_work-dir.md` — runtime directory documentation
- `work-dir/.gitkeep` — keeps the directory in Git
