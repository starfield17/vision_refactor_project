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
2. [System Integration Scripts](#system-integration-scripts)
   - [add_to_systemd.sh](#add_to_systemdsh)
   - [add_to_systemd_bin.sh](#add_to_systemd_binsh)
3. [Utility Scripts](#utility-scripts)
   - [get_frame.sh](#get_framesh)
   - [rescan.sh](#rescanssh)
   - [proxy.sh](#proxysh)
   - [change_pip_conda_source.sh](#change_pip_conda_sourcesh)
4. [What Was NOT Migrated](#what-was-not-migrated)

---

## Statistics Service Scripts

These four scripts manage the two-process statistics stack:
- **API** — FastAPI server on port `7797` (receives telemetry)
- **UI** — Streamlit dashboard on port `7796` (visualization)

Both processes are tracked via PID files written to `work-dir/`.

### `start_stats.sh`

Starts the Statistics API and UI in the background, then performs a health check against
the API's `/health` endpoint.

**Usage:**

```bash
bash scripts/start_stats.sh [--config PATH] [--workdir PATH]
```

**What it does:**

1. Reads `--config` (default: `./work-dir/config.toml`) to determine ports and paths.
2. Launches `python -m deploy.statistics.api` in the background, saving its PID to
   `work-dir/stats-api.pid`.
3. Launches `python -m deploy.statistics.ui` in the background, saving its PID to
   `work-dir/stats-ui.pid`.
4. Polls `http://localhost:<api_port>/health` until the API responds or a timeout
   (default: 15 seconds) is reached.
5. Prints a success or failure summary with the PIDs and URLs.

**Health check output example:**

```
[start_stats] Starting Statistics API on port 7797...
[start_stats] Starting Statistics UI on port 7796...
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

Stops the Statistics API and UI by sending `SIGTERM` to the PIDs recorded in the PID files.

**Usage:**

```bash
bash scripts/stop_stats.sh [--workdir PATH]
```

**What it does:**

1. Reads `work-dir/stats-api.pid` and `work-dir/stats-ui.pid`.
2. Sends `SIGTERM` to each process.
3. Waits up to 5 seconds for each process to exit; if it doesn't, sends `SIGKILL`.
4. Removes the PID files.

**Notes:**
- If a PID file doesn't exist, the script skips that service gracefully.
- If the process is already gone, the script reports it as "already stopped".

---

### `status_stats.sh`

Reports the current running state of the Statistics API and UI.

**Usage:**

```bash
bash scripts/status_stats.sh [--workdir PATH]
```

**Output example:**

```
[status_stats] API (pid 12345): RUNNING
[status_stats] API health: OK {"ok": true, "storage": "sqlite", ...}
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

## System Integration Scripts

### `add_to_systemd.sh`

Generates and installs a `systemd` unit file for the Statistics API and UI as user services.

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
ExecStart=/usr/bin/python3 -m deploy.statistics.api --config /path/to/config.toml
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