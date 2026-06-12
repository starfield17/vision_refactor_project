# scripts/

Shell utility and operations scripts for the Vision Refactor Project.

These scripts originated from the `common/` directory of the legacy repository and were
selectively migrated during the refactor.  New scripts specific to the refactored
statistics service have been added.

**All scripts are Bash and assume a Linux/macOS environment.**

---

## Table of Contents

1. [Statistics Service Scripts](#statistics-service-scripts)
   - [start_stats.sh](#start_statssh)
   - [stop_stats.sh](#stop_statssh)
   - [status_stats.sh](#status_statssh)
   - [restart_stats.sh](#restart_statssh)
2. [Train / AutoLabel Service Scripts](#train--autolabel-service-scripts)
   - [start_train_autolabel.sh](#start_train_autolabelsh)
   - [stop_train_autolabel.sh](#stop_train_autolabelsh)
3. [System Integration Scripts](#system-integration-scripts)
   - [add_to_systemd.sh](#add_to_systemdsh)
   - [add_to_systemd_bin.sh](#add_to_systemd_binsh)
4. [Utility Scripts](#utility-scripts)
   - [prepare_voc_detection_dataset.py](#prepare_voc_detection_datasetpy)
   - [get_frame.sh](#get_framesh)
   - [rescan.sh](#rescanssh)
   - [proxy.sh](#proxysh)
   - [change_pip_conda_source.sh](#change_pip_conda_sourcesh)
5. [What Was NOT Migrated](#what-was-not-migrated)

---

## Statistics Service Scripts

These four scripts manage the two-process deploy/statistics stack:
- **Backend** — `services.deploy_statistics.api` FastAPI server on port `7797`
  by default. It receives telemetry, serves dashboard queries, manages deploy jobs,
  and can host the remote frame inference runtime.
- **UI** — React/Vite dashboard on port `7796` by default.

Both processes are tracked via PID files written to `work-dir/`.

### `start_stats.sh`

Starts the deploy/statistics backend and React dashboard in the background, then performs
a health check against the backend `/health` endpoint.

**Usage:**

```bash
bash scripts/start_stats.sh [--config PATH] [--workdir PATH]
```

**What it does:**

1. Reads `--config` (default: `./work-dir/config.toml`) to determine ports and paths.
2. Launches `python -m services.deploy_statistics.api` in the background, saving its PID to
   `work-dir/tmp/statistics_api.pid`.
3. Launches `npm --prefix web/deploy_statistics run dev` in the background, saving its PID to
   `work-dir/tmp/statistics_ui.pid`.
4. Polls `http://localhost:<services.deploy_statistics.port>/health` until the backend responds or a timeout
   (default: 15 seconds) is reached.
5. Prints a success or failure summary with the PIDs and URLs.

**Health check output example:**

```
[start_stats] Starting deploy/statistics backend on port 7797...
[start_stats] Starting React dashboard on port 7796...
[start_stats] Waiting for API health check...
[start_stats] ✓ API is up (pid 12345)
[start_stats] ✓ UI is up (pid 12346)
[start_stats] Dashboard: http://localhost:7796
```

**Common failure modes:**
- Port already in use → check `status_stats.sh`
- Missing `config.toml` → run `cp work-dir/config.example.toml work-dir/config.toml` first
- Python package not installed → run `pip install -e .`

---

### `stop_stats.sh`

Stops the deploy/statistics backend and React UI by sending `SIGTERM` to the PIDs recorded
in the PID files.

**Usage:**

```bash
bash scripts/stop_stats.sh [--workdir PATH]
```

**What it does:**

1. Reads `work-dir/tmp/statistics_api.pid` and `work-dir/tmp/statistics_ui.pid`.
2. Sends `SIGTERM` to each process.
3. Waits up to 5 seconds for each process to exit; if it doesn't, sends `SIGKILL`.
4. Removes the PID files.

**Notes:**
- If a PID file doesn't exist, the script skips that service gracefully.
- If the process is already gone, the script reports it as "already stopped".

---

### `status_stats.sh`

Reports the current running state of the deploy/statistics backend and React UI.

**Usage:**

```bash
bash scripts/status_stats.sh [--workdir PATH]
```

**Output example:**

```
[status_stats] API (pid 12345): RUNNING
[status_stats] API health: OK {"ok": true, "service": "deploy_statistics", ...}
[status_stats] UI  (pid 12346): RUNNING
[status_stats] UI  accessible at http://localhost:7796
```

Or, if services are not running:

```
[status_stats] API: NOT RUNNING (no pid file)
[status_stats] UI:  NOT RUNNING (no pid file)
```

---

### `restart_stats.sh`

Convenience wrapper: calls `stop_stats.sh` followed by `start_stats.sh`.

**Usage:**

```bash
bash scripts/restart_stats.sh [--config PATH] [--workdir PATH]
```

---

## Train / AutoLabel Service Scripts

These scripts manage the two-process train/autolabel WebUI stack:
- **Backend** — `services.train_autolabel.api` FastAPI server on port `7793` by default.
- **UI** — React/Vite app under `web/train_autolabel` on port `7794` by default.

The backend API does not serve `/`; use `/health` for health checks and the UI port for
the browser page.

### `start_train_autolabel.sh`

Starts the train/autolabel backend and React WebUI in the background, then checks both
the backend `/health` endpoint and the UI port.

**Usage:**

```bash
bash scripts/start_train_autolabel.sh [--config PATH] [--workdir PATH] [--ui-port PORT] \
  [--env-file PATH] [--python-bin PATH]
```

**What it does:**

1. Reads `--config` (default: `./work-dir/config.toml`) to resolve `workspace.root` and
   `[services.train_autolabel]`.
2. Sources `work-dir/secrets/llm.env` if present, or a custom `--env-file`, so LLM
   AutoLabel credentials are available to the backend without tracking them in Git.
3. Launches `python -m services.train_autolabel.api` and writes
   `work-dir/tmp/train_autolabel_api.pid`.
4. Launches the Vite dev server from `web/train_autolabel` and writes
   `work-dir/tmp/train_autolabel_ui.pid`.
5. Exports `VITE_TRAIN_AUTOLABEL_API_URL` and `VITE_TRAIN_AUTOLABEL_API_TOKEN` for the UI.

If the API or UI port is already serving, the script reports that process as `external`
instead of starting a duplicate process.

### `stop_train_autolabel.sh`

Stops train/autolabel processes started by `start_train_autolabel.sh`.

**Usage:**

```bash
bash scripts/stop_train_autolabel.sh [--config PATH] [--workdir PATH] [--python-bin PATH]
```

The script only stops PIDs recorded in `work-dir/tmp/train_autolabel_api.pid` and
`work-dir/tmp/train_autolabel_ui.pid`. If the service was started manually before the
start script was used, stop that manual process directly.

---

## System Integration Scripts

### `add_to_systemd.sh`

Generates and installs a `systemd` unit file for the statistics API and UI as user services.
This script still targets the compatibility `deploy.statistics.api` launcher; new
installations should prefer `services.deploy_statistics.api`.

**Usage:**

```bash
bash scripts/add_to_systemd.sh --config PATH [--workdir PATH] [--user]
```

**What it does:**

1. Generates two `.service` unit files:
   - `vision-stats-api.service`
   - `vision-stats-ui.service`
2. Writes them to `~/.config/systemd/user/` (with `--user`) or `/etc/systemd/system/`.
3. Calls `systemctl daemon-reload` and optionally `systemctl enable`.

**Generated unit file example (API):**

```ini
[Unit]
Description=Vision Refactor Statistics API
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 -m services.deploy_statistics.api --config /path/to/config.toml
WorkingDirectory=/path/to/repo
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

---

### `add_to_systemd_bin.sh`

Variant of `add_to_systemd.sh` that uses a compiled or virtualenv-specific Python binary
path instead of the system Python.  Useful when the project is installed inside a `venv`.

**Usage:**

```bash
bash scripts/add_to_systemd_bin.sh --python-bin /path/to/venv/bin/python \
  --config PATH [--workdir PATH]
```

---

## Utility Scripts

### `prepare_voc_detection_dataset.py`

Downloads Pascal VOC2007 and converts it into the project runtime layouts:

- YOLO dataset: `work-dir/datasets/voc2007_yolo`
- Faster-RCNN / AutoLabel `LabelRecord` JSON data: `work-dir/datasets/voc2007_labeled`
- Unlabeled image folder for LLM AutoLabel: `work-dir/datasets/voc2007_unlabeled/images`
- Image-folder deploy source: `work-dir/datasets/voc2007_deploy_images`

**Usage:**

```bash
python scripts/prepare_voc_detection_dataset.py --workdir ./work-dir \
  --max-train 1500 --max-val 500 --max-unlabeled 30 --max-deploy 30
```

The downloaded VOC tar archives and converted data stay under `work-dir/` and are ignored
by Git.

### `get_frame.sh`

Grabs a single frame from a camera, video file, or RTSP stream and saves it to disk.
Useful for debugging the edge capture pipeline or testing image quality.

**Usage:**

```bash
bash scripts/get_frame.sh [--source camera|video|rtsp] [--id 0] [--output frame.jpg]
```

**Examples:**

```bash
# Grab from the default camera
bash scripts/get_frame.sh --source camera --id 0 --output /tmp/test.jpg

# Grab a frame from a video file
bash scripts/get_frame.sh --source video --input ./recording.mp4 --output /tmp/frame.jpg
```

Internally uses `python -c "import cv2; ..."` or `ffmpeg` (whichever is available).

---

### `rescan.sh`

Re-scans a directory for new or changed image files and optionally re-runs autolabel
on unprocessed images.

**Usage:**

```bash
bash scripts/rescan.sh --config PATH [--dir PATH]
```

This script is used for incremental labeling pipelines where new unlabeled images are
deposited into a folder continuously.

---

### `proxy.sh`

Starts a simple HTTP reverse proxy (using `socat` or `nginx`) to expose the statistics
API and UI under a single port or with TLS termination.

**Usage:**

```bash
bash scripts/proxy.sh --api-port 7797 --ui-port 7796 --public-port 8080
```

Intended for development setups that need to expose the two services through a single
public-facing URL.  For production, prefer a proper `nginx` or `caddy` configuration.

---

### `change_pip_conda_source.sh`

Switches `pip` and/or `conda` to use a mirror registry (e.g., a corporate PyPI mirror or
a Chinese mirror for faster downloads).

**Usage:**

```bash
# Switch to Tsinghua mirror
bash scripts/change_pip_conda_source.sh --source tsinghua

# Switch back to official PyPI
bash scripts/change_pip_conda_source.sh --source official
```

**Supported sources:**

| Name | pip index URL |
|------|--------------|
| `official` | `https://pypi.org/simple` |
| `tsinghua` | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| `aliyun` | `https://mirrors.aliyun.com/pypi/simple` |

---

## LocateAnything Smoke Tests

After starting the Train/AutoLabel service, run a capped AutoLabel smoke test:

```bash
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=locate_anything \
  --set autolabel.visualize=true \
  --set locate_anything.device=cuda \
  --set locate_anything.max_images=20
```

After starting Statistics, run a capped deploy smoke test:

```bash
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=locate_anything \
  --set deploy.edge.source=images \
  --set deploy.edge.max_frames=1 \
  --set locate_anything.device=cuda
```

Use these smoke tests before removing `max_images` or increasing `max_frames`, because
LocateAnything uses a much heavier VLM path than local ONNX inference.

