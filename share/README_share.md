# share/

The `share/` package is the **shared library** of the Vision Refactor Project.
It contains all reusable logic that is imported by the top-level pipeline modules
(`train/`, `autolabel/`, `deploy/`).

**No top-level module may import from another top-level module.**
All shared logic must live here.

---

## Sub-packages

### `share/config/`

Configuration loading, validation, path resolution, and TOML serialization.

| Module | Purpose |
|--------|---------|
| `schema.py` | `DEFAULT_CONFIG`, validation enums, `validate_config()` |
| `config_loader.py` | `load_config()`, `apply_overrides()`, `resolve_path_fields()`, `save_resolved_config()`, `to_toml()` |

**Key public API:**

```python
from share.config.config_loader import load_config, save_resolved_config

cfg = load_config(
    config_path=Path("work-dir/config.toml"),
    overrides=["train.epochs=50", "train.device=cuda"],
    workdir_override=None,
)
```

`load_config()` performs this pipeline:
1. Read TOML file
2. Deep-merge with `DEFAULT_CONFIG` (caller's keys win)
3. Apply `--set` overrides (dotted key syntax, auto-typed values)
4. Optionally override `workspace.root` via `workdir_override`
5. Resolve all path fields to absolute paths
6. Validate with `validate_config()` → raises `ConfigError` on any violation
7. Return the final config dict

---

### `share/types/`

Typed data contracts (dataclasses) shared between all pipeline stages.

| Module | Class | Description |
|--------|-------|-------------|
| `detection.py` | `Detection` | One bounding-box result with class info and score |
| `label.py` | `LabelRecord` | One annotated image with a list of detections |
| `stats.py` | `StatsEvent` | One telemetry event with detection counts and latency |
| `errors.py` | `VisionRefactorError` and subclasses | Exception hierarchy |

All types validate on `.to_dict()` and `.from_dict()`.

---

### `share/kernel/`

Core pipeline implementations.  See `share/kernel/README.md` for full details.

| Sub-package | Contents |
|-------------|---------|
| `trainer/` | YOLO and Faster-RCNN training runners |
| `infer/` | ONNX inference adapters |
| `autolabel/` | Model-based and LLM-based autolabeling runners |
| `deploy/` | Edge (local/stream/llm) and remote server deploy runners |
| `export/` | ONNX export and INT-8 quantization |
| `transport/` | HTTP helpers for stats push and frame streaming |
| `statistics/` | SQLite-backed statistics store |
| `llm/` | OpenAI-compatible vision LLM client |
| `media/` | Frame source generator and annotated preview rendering |
| `utils/` | Structured JSONL logger |
| `kernel.py` | `VisionKernel` orchestrator |
| `registry.py` | `KernelRegistry` plugin registry |

---

## Dependency Policy

- `share/` may be imported by any module.
- `share/` sub-packages may import from each other freely.
- `share/` must **not** import from `train/`, `autolabel/`, or `deploy/`.
- All third-party imports inside `share/kernel/` that are optional (e.g., `cv2`, `torch`,
  `ultralytics`, `fastapi`, `streamlit`) are guarded with `try/except` and raise
  `ConfigError` or `DataValidationError` with a clear message if the dependency is missing.

