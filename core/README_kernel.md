# core/

`core/` contains the vision runtime logic used by the distributed roles. It does not
own HTTP orchestration, node registration, or Control Plane state.

## Main Modules

```text
kernel.py       VisionKernel orchestration for train/autolabel/deploy runs
registry.py     Runtime backend registry
trainer/        YOLO and Faster R-CNN training
autolabel/      model, LLM, and LocateAnything auto-labeling
deploy/         edge local, edge LLM, edge stream, and remote server runtime
infer/          local and ONNX inferencer factories
export/         ONNX export and quantization helpers
statistics/     event storage and dashboard aggregation
transport/      frame/statistics HTTP helpers
llm/            OpenAI-compatible LLM client utilities
media/          frame sources and preview generation
utils/          logging and small runtime helpers
```

## Config Boundary

Role packages load role-local configs. `common.config.role_schema.role_to_kernel_config()`
adapts those role configs into the full config shape expected by `VisionKernel`.

New Control Plane or worker APIs should use role config keys such as `runtime.*` and
should keep orchestration concerns outside `core/`.

## Run Result Contract

`VisionKernel` returns a `RunResult` with:

```text
run_context.run_id
run_context.run_dir
status
backend
elapsed_ms
artifacts
error
```

Each run writes `metrics.json`, `artifacts.json`, and a resolved config snapshot under
`work-dir/runs/<run-id>/`.

## LLM API Keys

LLM runtimes accept either direct `api_key` or `api_key_env_name`. Prefer environment
variables for deployed services.
