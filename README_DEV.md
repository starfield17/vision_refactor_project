# Developer Notes

## Design Direction

The active codebase is a distributed vision system:

```text
Control Plane -> registered nodes -> role workers -> core vision kernel/statistics
```

The Control Plane owns orchestration state. Role packages own execution state. The
kernel remains framework/runtime logic and does not know about HTTP services.

## Package Boundaries

```text
common/
  application/   HTTP helpers, job runner/store, node registration, service helpers
  config/        role config loading and role-to-kernel config adapter
  types/         shared data/error types

core/
  trainer/       YOLO/Faster R-CNN training
  autolabel/     model, LLM, LocateAnything labeling
  deploy/        edge local/LLM/stream/remote runtime
  statistics/    SQLite event store and dashboard aggregation

control_plane/
  api.py         node registry, job dispatch, status/log aggregation
  store.py       SQLite control-plane nodes/jobs
  web/           Vite React operations UI

*_worker/ and *_agent/
  config/        role-owned TOML schema and example
  service.py     HTTP service for Control Plane integration
  worker.py      subprocess job entry when jobs are long-running
  cli.py         local direct-run entry where applicable
```

## Config Rules

Each role uses its own TOML file. Do not add a new global config. Prefer these keys:

```text
workspace.*
node.*
server.*
job_store.*
control_plane.*
runtime.*
```

The `common.config.role_schema.role_to_kernel_config()` adapter exists only because
`core` consumes the full kernel config shape. New APIs and tests should use role config
keys and should not expose old package names or old service names.

## Control Plane Job Flow

1. Worker starts and calls `POST /api/v1/nodes/heartbeat`.
2. Control Plane persists node identity, endpoint, role, status, and dispatch token.
3. Client submits `POST /api/v1/jobs` with `kind` and `payload`.
4. Control Plane maps `kind` to a role and forwards to that worker:
   - `train` -> `train_worker`
   - `autolabel` -> `autolabel_worker`
   - `edge_run` -> `edge`
5. Worker creates a local `JobStore` record and launches its subprocess runner.
6. Control Plane stores the upstream job id and refreshes status/logs through worker APIs.

## Adding a New Role

1. Create `<role>/config/schema.py` and `<role>/config/config.example.toml`.
2. Add `node`, `server`, `job_store`, and `control_plane` sections if it participates in orchestration.
3. Add a service with:
   - `GET /health`
   - `GET /api/v1/status`
   - `POST /api/v1/nodes/register`
   - `POST /api/v1/jobs` when it runs jobs
   - job list/detail/logs/cancel endpoints for long-running jobs
4. Register role package discovery in `pyproject.toml`.
5. Add focused tests under `tests/`.
6. Add Podman compose wiring if it should be deployable.

## Testing

Use the configured Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q \
  common core control_plane train_worker autolabel_worker edge_agent remote_worker stats_service
```

For package and frontend checks:

```bash
python -m pip wheel . --no-deps -w /tmp/vision_wheel_test
npm --prefix control_plane/web install
npm --prefix control_plane/web run build
bash -n deployments/install.sh
```

The Podman Web image is defined in `control_plane/web/Containerfile` and is wired into
the `all-in-one` profile as `control-plane-web`.

## Local Archive

Pre-distributed code lives under `archive/legacy_pre_distributed/` for local reference
only. `archive/**` is ignored by git. Do not import from or test archived code.
