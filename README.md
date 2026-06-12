# Distributed Vision System

This repository is organized as a distributed visual system. Each runtime role owns its
own config, API surface, and optional web entry instead of sharing one global config.

## Active Layout

```text
common/             Shared API, config, job, and type helpers
core/               Vision kernel, training, inference, deploy, statistics logic
control_plane/      Control Plane API, SQLite state, and Web UI
train_worker/       Train worker API, CLI, worker process, role config
autolabel_worker/   Auto-label worker API, CLI, worker process, role config
edge_agent/         Edge agent API/CLI/worker for local camera/video/image runs
remote_worker/      Remote inference API for GPU/offloaded frame inference
stats_service/      Statistics ingest and dashboard API
deployments/        Podman compose profiles and install script
archive/            Local-only archived pre-distributed code, ignored by git
```

Old top-level `train/`, `autolabel/`, `deploy/`, `services/`, `share/`, and `web/`
code has been moved to `archive/legacy_pre_distributed/` and is intentionally ignored.

## Python Environment

Use an activated Python environment with the project dependencies installed:

```bash
python -m pytest -q
```

## Role Configs

Each role config is independent:

```text
control_plane/config/config.example.toml
train_worker/config/config.example.toml
autolabel_worker/config/config.example.toml
edge_agent/config/config.example.toml
remote_worker/config/config.example.toml
stats_service/config/config.example.toml
```

Important sections:

```text
[workspace]       Role-local workspace and logging
[node]            Node identity registered with the Control Plane
[server]          HTTP bind address, port, auth token, advertised URL
[job_store]       SQLite job state for roles that run jobs
[control_plane]   Control Plane URL/token for node registration
[runtime]         Role-specific train/autolabel/edge/remote/statistics runtime
```

The worker CLIs and APIs accept overrides with role-local keys, for example:

```bash
python -m train_worker.cli \
  --config train_worker/config/config.example.toml \
  --set runtime.dry_run=true \
  --json-summary
```

## Control Plane

Start the Control Plane API:

```bash
python -m control_plane.api \
  --config control_plane/config/config.example.toml
```

Start workers:

```bash
python -m train_worker.service \
  --config train_worker/config/config.example.toml

python -m autolabel_worker.service \
  --config autolabel_worker/config/config.example.toml

python -m edge_agent.service \
  --config edge_agent/config/config.example.toml

python -m stats_service.api \
  --config stats_service/config/config.example.toml
```

Workers register with the Control Plane on startup when `[control_plane].url` is set.
They can also be registered manually:

```bash
curl -X POST http://127.0.0.1:7811/api/v1/nodes/register
```

Control Plane endpoints:

```text
GET  /health
GET  /api/v1/nodes
GET  /api/v1/nodes/{node_id}
POST /api/v1/nodes/heartbeat
GET  /api/v1/workers/status
POST /api/v1/jobs
GET  /api/v1/jobs?refresh=true
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/logs
POST /api/v1/jobs/{job_id}/cancel
GET  /api/v1/models
```

Submit a train job through the Control Plane:

```bash
curl -X POST http://127.0.0.1:7800/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"train","payload":{"dry_run":true}}'
```

Submit an auto-label job:

```bash
curl -X POST http://127.0.0.1:7800/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"autolabel","payload":{"mode":"model"}}'
```

Submit an edge run:

```bash
curl -X POST http://127.0.0.1:7800/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"edge_run","payload":{"mode":"local","source":"images","max_frames":20}}'
```

## Web UI

The active web UI lives under the Control Plane:

```bash
npm --prefix control_plane/web install
npm --prefix control_plane/web run dev
```

Set `VITE_CONTROL_PLANE_API_URL` when the API is not on `http://127.0.0.1:7800`.

The `all-in-one` Podman profile also serves the built Web UI. Defaults:

```text
Control Plane API: http://127.0.0.1:7800
Control Plane Web: http://127.0.0.1:7801
```

## Podman Deployment

The project targets Podman, not Docker. Profiles:

```bash
bash deployments/install.sh all-in-one
bash deployments/install.sh edge
bash deployments/install.sh remote-gpu
```

The install script creates a profile `.env` from `.env.example`, creates required
`work-dir` subdirectories, and starts `podman compose` or `podman-compose`.

`all-in-one` starts:

```text
control-plane
control-plane-web
statistics
train-worker
autolabel-worker
edge-agent
```

## Verification

```bash
python -m pip install -e ".[dev]"
bash scripts/check.sh

# Apply Ruff fixes and formatting before running the same checks:
bash scripts/check.sh --fix
```
