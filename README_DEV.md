# Developer README

This document is for contributors and maintainers.  It covers the architecture, constraints,
module internals, configuration system design, and development workflow.

For user-facing usage instructions, see `README.md`.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Dependency Rules](#module-dependency-rules)
3. [Directory Map & Responsibilities](#directory-map--responsibilities)
4. [Configuration System](#configuration-system)
5. [Kernel & Registry Pattern](#kernel--registry-pattern)
6. [Data Contracts (share/types)](#data-contracts-sharetypes)
7. [Logging (StructuredLogger)](#logging-structuredlogger)
8. [Transport Layer](#transport-layer)
9. [Statistics Store](#statistics-store)
10. [Adding a New Backend](#adding-a-new-backend)
11. [Local Development Workflow](#local-development-workflow)
12. [Commit Guidelines](#commit-guidelines)
13. [Testing Guidance](#testing-guidance)
14. [Known Limitations / TODOs](#known-limitations--todos)

---

## Architecture Overview

```
┌──────────┐  ┌───────────┐  ┌─────────────────────┐  ┌────────────────────┐
│  train/  │  │autolabel/ │  │    deploy/edge/      │  │  deploy/remote/    │
│  cli.py  │  │  cli.py   │  │      cli.py          │  │     cli.py         │
└────┬─────┘  └─────┬─────┘  └──────────┬───────────┘  └────────┬───────────┘
     │               │                   │                        │
     └───────────────┴───────────────────┴────────────────────────┘
                                   │  all import from
                             ┌─────▼──────┐
                             │  share/    │
                             │  kernel/   │  ← all pipeline logic lives here
                             │  config/   │  ← TOML load/validate/resolve
                             │  types/    │  ← typed data contracts
                             └────────────┘
                                   │
                              work-dir/      ← runtime data (never committed)
                              scripts/       ← shell operations tools
```

The top-level modules (`train`, `autolabel`, `deploy`) are **thin CLI wrappers**.  They:

1. Parse CLI arguments
2. Call `load_config()` to get a validated, fully-resolved config dict
3. Build a `KernelRegistry` and register concrete backend functions
4. Instantiate `VisionKernel` and call the appropriate `run_*` method
5. Save the resolved config snapshot and print the run summary

All pipeline logic, model loading, inference, data I/O, and network transport live in `share/`.

---

## Module Dependency Rules

**Hard rule: no cross-imports between top-level modules.**

```
✅ train/    → share/
✅ autolabel/ → share/
✅ deploy/   → share/
❌ train/    → autolabel/
❌ deploy/   → train/
❌ autolabel/ → deploy/
```

This constraint is enforced by convention; a lint check or import-linter rule is recommended
for CI enforcement (see [Testing Guidance](#testing-guidance)).

`share/` is the only module allowed to be imported by all others.  Within `share/`, modules
may freely depend on each other.

---

## Directory Map & Responsibilities

### `share/config/`

| File | Responsibility |
|------|---------------|
| `schema.py` | Defines `DEFAULT_CONFIG` (nested dict), validation constants (enum sets), and `validate_config()` which raises `ConfigError` for any schema violation. |
| `config_loader.py` | `load_config()`: loads TOML → deep-merges with defaults → applies `--set` overrides → resolves relative paths → validates. Also provides `save_resolved_config()` and `to_toml()` for writing snapshots. |

**Path resolution:** The `PATH_FIELDS` tuple in `config_loader.py` lists every key that
holds a filesystem path.  Relative paths are resolved against the directory containing
`config.toml`, with a special case: if `config.toml` is inside a directory named `work-dir/`,
paths resolve from the parent (repo root) for backward compatibility.

**Overrides:** `--set key=value` calls `parse_override_value()` which auto-converts
`true`/`false` → `bool`, numeric strings → `int`/`float`, and everything else → `str`.

### `share/types/`

Typed data contracts shared across all pipeline stages:

| File | Class | Purpose |
|------|-------|---------|
| `detection.py` | `Detection` | A single bounding box result (class_id, class_name, score, bbox_xyxy) |
| `label.py` | `LabelRecord` | An annotated image (path, source, list of detections) |
| `stats.py` | `StatsEvent` | Telemetry event (source_id, timestamp, detection counts, latency) |
| `errors.py` | Various | Exception hierarchy rooted at `VisionRefactorError` |

All types implement `validate()`, `to_dict()`, and `from_dict()`.  `validate()` is
always called before serialization.  `from_dict()` always calls `validate()` after
deserialization.  This ensures no invalid data can silently propagate through the pipeline.

### `share/kernel/`

The core business logic.  Each sub-package maps to one pipeline stage:

| Package | Description |
|---------|-------------|
| `trainer/` | `run_yolo_train()`, `run_faster_rcnn_train()` — registered with the kernel as backends |
| `infer/` | ONNX inference adapters for YOLO and Faster-RCNN |
| `autolabel/` | `run_model_autolabel()`, `run_llm_autolabel()` |
| `deploy/` | `run_edge_local_deploy()`, `run_edge_stream_deploy()`, `run_edge_llm_deploy()`, `run_remote_deploy()` |
| `export/` | `onnx_export.py` — ONNX export + INT-8 dynamic quantization |
| `transport/` | `stats_http.py` (push stats), `frame_http.py` (stream frames) — pure stdlib urllib |
| `statistics/` | `sqlite_store.py` — SQLite schema init, insert, query |
| `llm/` | `client.py` — OpenAI-compatible vision API wrapper with retry and QPS limiting |
| `media/` | `frame_source.py` — camera/video/image folder generator; `preview.py` — annotated frame rendering |
| `utils/` | `logging.py` — `StructuredLogger` |

### `share/kernel/kernel.py` — `VisionKernel`

The kernel is not the business logic; it is the **orchestrator**.  For every pipeline run:

1. Calls `_ensure_workdir_layout()` to create the required subdirectories.
2. Calls `_make_run_context()` to generate a unique `run_id` and create the run directory.
3. Looks up the registered backend function from the registry.
4. Calls the backend function, catching all exceptions to ensure run artifacts are always
   written, regardless of failure.
5. Writes `metrics.json` and `artifacts.json` to the run directory.
6. Returns a `RunResult` dataclass.

### `share/kernel/registry.py` — `KernelRegistry`

A simple plugin registry that maps string keys to callables.  There are three registries:
trainers, autolabelers, deployers.  All callables share the same signature:

```python
def my_backend(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    ...
```

Where `run_ctx` contains `run_id`, `run_dir`, and `logger`.

---

## Configuration System

### Validation Rules (enforced in `validate_config`)

Every field is type-checked and range-validated.  Key constraints:

- `app.schema_version` must be exactly `1`
- `workspace.log_level` must be one of `{DEBUG, INFO, WARN, ERROR}`
- `train.backend` must be one of `{yolo, faster_rcnn}`
- `train.epochs`, `train.batch_size`, `train.img_size` must all be `> 0`
- `train.faster_rcnn.lr` must be `> 0`; `momentum` must be in `[0, 1]`
- `autolabel.confidence` must be in `[0, 1]`
- `autolabel.on_conflict` must be one of `{skip, overwrite, merge}`
- `deploy.edge.mode` must be one of `{local, stream, llm}`
- `deploy.edge.source` must be one of `{camera, video, images}`
- `deploy.edge.jpeg_quality` must be in `[1, 100]`
- `deploy.remote.listen_port` must be in `[1, 65535]`
- `deploy.statistics.storage` must be `sqlite` (only backend currently implemented)
- `class_map.names` must be non-empty, no duplicates; `id_map` must match `names` order exactly
- LLM mode fields (`base_url`, `model`, `prompt`) must be non-empty when `mode = "llm"`

### Adding a New Config Field

1. Add the field with its default value to `DEFAULT_CONFIG` in `share/config/schema.py`.
2. Add validation logic in `validate_config()` in the same file.
3. If it's a path field, add it to `PATH_FIELDS` in `share/config/config_loader.py`.
4. Update `work-dir/config.example.toml` with a documented example.
5. Update the relevant README section.

---

## Kernel & Registry Pattern

### Why this pattern?

The registry decouples the CLI entry point from the backend implementation.  The CLI only
knows which backend name to request; the registry maps that name to the actual function.
This makes it straightforward to add new backends without changing existing code.

### Extending the registry

To add a new training backend called `"detr"`:

```python
# In share/kernel/trainer/detr.py:
def run_detr_train(cfg: dict, run_ctx: dict) -> dict:
    ...
    return {"model_path": str(output_path), "status": "ok"}

# In train/cli.py, in main():
registry.register_trainer("detr", run_detr_train)
```

Update `TRAIN_BACKENDS` in `share/config/schema.py` and add validation for any new
`[train.detr]` config sub-section.

---

## Data Contracts (`share/types`)

### `Detection`

```python
@dataclass(slots=True)
class Detection:
    schema_version: int       # Must be 1
    class_id:       int       # >= 0
    class_name:     str       # non-empty
    score:          float     # [0.0, 1.0]
    bbox_xyxy:      list[float]  # exactly 4 values: [x1, y1, x2, y2]
```

### `LabelRecord`

```python
@dataclass(slots=True)
class LabelRecord:
    schema_version: int         # Must be 1
    image_path:     str         # non-empty
    source:         str         # e.g. "model", "llm", "manual"
    detections:     list[Detection]
```

### `StatsEvent`

```python
@dataclass(slots=True)
class StatsEvent:
    schema_version:   int             # Must be 1
    source_id:        str             # non-empty
    ts_utc:           str             # ISO-8601 UTC timestamp
    total_detections: int             # >= 0
    counts_by_class:  dict[str, int]  # class_name → count (all values >= 0)
    latency_ms:       float           # >= 0
```

### Exception Hierarchy

```
VisionRefactorError
├── ConfigError         — config load/validation failure
├── DataValidationError — type/data contract violation
├── ModelExportError    — ONNX export failure
└── TransportError      — HTTP/network failure
```

---

## Logging (`StructuredLogger`)

Located at `share/kernel/utils/logging.py`.

Each `StructuredLogger` instance writes one JSON object per line to a file, thread-safely.

```python
logger = StructuredLogger(log_path=Path("work-dir/log.txt"), level="INFO")
logger.info("train.start", "Training started", run_id="exp001", backend="yolo")
```

Output line:
```json
{"ts_utc": "2025-01-01T12:00:00.000000+00:00", "level": "INFO", "event": "train.start",
 "message": "Training started", "run_id": "exp001", "backend": "yolo"}
```

Level filtering: messages below `min_level` are silently dropped without I/O.
Locking: uses `threading.Lock` — safe for multi-threaded servers.

---

## Transport Layer

### `stats_http.py` — `push_stats_event()`

Serializes a `StatsEvent`, POSTs to the statistics API endpoint via stdlib `urllib`.
Raises `TransportError` on non-200 HTTP responses or network errors.
Optionally includes `X-API-Key` header.

### `frame_http.py` — `post_json()`, `encode_jpeg_base64()`, `decode_jpeg_base64()`

Used in `deploy/edge` (stream mode) to forward raw JPEG frames to the remote inference server.

- `encode_jpeg_base64(frame_bgr, jpeg_quality)` — OpenCV JPEG encode → base64 string
- `decode_jpeg_base64(encoded)` — base64 string → OpenCV BGR frame
- `post_json(endpoint, payload, timeout_sec, api_key)` — HTTP POST, raises `TransportError`

Both functions depend on `opencv-python`.  Import errors are wrapped as `DataValidationError`.

---

## Statistics Store

`share/kernel/statistics/sqlite_store.py`

Schema (auto-created by `init_stats_db(db_path)`):

```sql
CREATE TABLE IF NOT EXISTS stats_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id        TEXT    NOT NULL,
    ts_utc           TEXT    NOT NULL,
    total_detections INTEGER NOT NULL,
    counts_by_class  TEXT    NOT NULL,  -- JSON
    latency_ms       REAL    NOT NULL
);
```

Key functions:

| Function | Description |
|----------|-------------|
| `init_stats_db(db_path)` | Creates DB and table if they don't exist |
| `insert_stats_event(db_path, event)` | Inserts one `StatsEvent` |
| `get_recent_events(db_path, limit)` | Returns most recent N events (newest first) |
| `get_overview(db_path)` | Returns aggregate stats: total events, total detections, avg latency, source count, last timestamp |
| `get_class_totals(db_path, limit)` | Returns per-class detection totals over the last N events |

---

## Adding a New Backend

General checklist:

1. **Create implementation file** in the appropriate `share/kernel/` sub-package.
2. **Define the runner function** with signature `(cfg, run_ctx) -> dict`.
3. **Add the backend name** to the relevant validation set in `share/config/schema.py`
   (e.g., `TRAIN_BACKENDS`).
4. **Add config sub-section** (if needed) to `DEFAULT_CONFIG` and `validate_config()`.
5. **Register the function** in the relevant CLI's `main()`.
6. **Update `work-dir/config.example.toml`** with commented example fields.
7. **Update READMEs** (this file and `README.md`).
8. **Write a smoke test** (see [Testing Guidance](#testing-guidance)).

---

## Local Development Workflow

```bash
# Install in editable mode
python -m pip install -e .

# Syntax-check a file without running it
python -m py_compile share/config/config_loader.py

# Syntax-check the entire share/ package
find share/ -name "*.py" | xargs python -m py_compile

# Run a quick dry-run train to validate your config and the pipeline
python -m train.cli \
  --config ./work-dir/config.toml \
  --set train.dry_run=true \
  --set train.epochs=1

# Verify statistics API starts cleanly
python -m deploy.statistics.api --config ./work-dir/config.toml &
curl http://localhost:7797/health
```

---

## Commit Guidelines

- **One logical change per commit** — small, reviewable diffs.
- **Before committing**, verify:
  - No model weight files (`*.pt`, `*.onnx`) are staged.
  - No dataset files, logs, or databases are staged.
  - `work-dir/` runtime artifacts are excluded.
  - No `codex.md` or `config.toml` is staged.
- **Commit message format:** `<scope>: <short description>`
  - Scopes: `train`, `autolabel`, `deploy`, `share`, `config`, `scripts`, `docs`, `ci`
  - Example: `share: add StatsEvent.from_dict validation for empty source_id`

---

## Testing Guidance

Currently no automated test suite is present.  Recommended additions:

### Smoke Tests (pytest)

Create `tests/` at the repo root.

```
tests/
├── test_config.py      # validate_config() with valid/invalid inputs
├── test_types.py       # Detection/LabelRecord/StatsEvent round-trip
├── test_train_dry.py   # run train.cli with dry_run=true
├── test_autolabel.py   # run autolabel.cli with model mode on fixture images
└── test_stats_api.py   # start the FastAPI app, push an event, query health
```

### CI Pipeline (GitHub Actions example)

```yaml
jobs:
  lint-and-test:
    steps:
      - run: pip install -e .[dev]
      - run: find share/ -name "*.py" | xargs python -m py_compile
      - run: ruff check .
      - run: pytest tests/ -v
```

### Import-Linter (enforce cross-module constraint)

Add `import-linter` to dev dependencies and configure `.importlinter`:

```ini
[importlinter]
root_package = train

[importlinter:contract:no-cross-module]
name = Top-level modules must not cross-import
type = forbidden
source_modules = train, autolabel, deploy
forbidden_modules = train, autolabel, deploy
```

---

## Known Limitations / TODOs

| Area | Limitation / TODO |
|------|------------------|
| Export | `exportquantize_mode = "static"` is not implemented; only `"dynamic"` is supported |
| Statistics storage | Only `sqlite` backend is implemented; `postgres` is a natural next step |
| Statistics flush | `statistics.flush_interval_sec` is defined in config but the flush behaviour depends on the SQLite backend implementation |
| CI | No CI pipeline exists yet; add GitHub Actions lint + py_compile + smoke test |
| Tests | No pytest suite; add smoke regression tests for each pipeline |
| Remote server | No TLS support; assumes trusted internal network or a terminating proxy |
| Rate limiter | The statistics API rate limiter is an in-memory window per source; it resets on restart |
| LLM autolabel | Response parsing is LLM/prompt dependent; a structured output schema would improve reliability |

