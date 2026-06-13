# work-dir/

`work-dir/` is the local runtime data directory for the distributed vision system.
Runtime artifacts are ignored by git. Only this README, `.gitkeep`, and the pointer
`config.example.toml` are tracked.

## Contents

```text
work-dir/
├── artifacts/     Control Plane and role artifacts
├── datasets/      Training, auto-label, and edge input data
├── models/        Model outputs and model registry manifests
├── outputs/       Annotated images and exported outputs
├── runs/          Per-run metrics, artifacts, and resolved config snapshots
├── state/         SQLite job/control-plane databases
├── stats/         Statistics SQLite database
└── tmp/           Temporary files and service logs
```

## Configs

The active system uses role-owned configs:

```text
control_plane/config/config.example.toml
train/config/config.example.toml
autolabel/config/config.example.toml
edge_agent/config/config.example.toml
remote_worker/config/config.example.toml
stats_service/config/config.example.toml
```

Use `--workdir /path/to/work-dir` on role CLIs/services to redirect runtime output.

## Typical Checks

```bash
find work-dir/state -maxdepth 1 -type f
find work-dir/runs -maxdepth 2 -type f | head
find work-dir/models -maxdepth 3 -type f | head
```
