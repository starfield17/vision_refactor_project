# Manual for Agent

> Target Audience: Codex / Claude Code / Other Code Agents
> 
> Goal: Help users **quickly complete train / autolabel / deploy / statistics configuration and execution**, not architectural development.
> 
> Language Convention: Prioritize using Chinese to communicate with users; commands, paths, and configuration keys use original English.

---

## 1. One-Sentence Project Description

This is a modular vision project with the main workflow:

`Train -> AutoLabel -> Deploy (Edge / Remote) -> Statistics`

All entry points are Python modules:

- `python -m train.cli`
- `python -m autolabel.cli`
- `python -m deploy.edge.cli`
- `python -m deploy.remote.cli`
- `python -m deploy.statistics.api`
- `python -m deploy.statistics.ui`

Core project features:

- Single configuration file: `work-dir/config.toml`
- All runtime artifacts in: `work-dir/`
- Training artifacts generate `model_manifest.json`, which can automatically identify `model_id/backend` during deployment
- Statistics uses SQLite by default
- Structured logs in `work-dir/log.txt`

---

## 2. Your Default Working Method as an Agent

When users request "help me run/configure/deploy/train/autolabel", prioritize the following workflow:

1. **Don't modify code first**, check if only configuration or commands need changes.
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
   to complete tasks.
5. If the user's goal is "quick verification", prioritize choosing:
   - `images` as input source
   - `max_frames=1` or few frames
   - Existing models rather than retraining
6. If the user's goal is "real deployment", prioritize checking:
   - Is Statistics started
   - Are `stats_endpoint` / `stream_endpoint` reachable
   - Does API key come from environment variables

**Don't do these things by default:**

- Don't refactor code by default
- Don't add features by default
- Don't run long training sessions by default
- Don't overwrite user's existing configuration by default, unless explicitly agreed

---

## 3. Key Directory Quick Reference

- Configuration file: `work-dir/config.toml`
- Configuration template: `work-dir/config.example.toml`
- Logs: `work-dir/log.txt`
- Training artifacts: `work-dir/models/<run_name>/`
- Run records: `work-dir/runs/<run_id>/`
- Statistics database: `work-dir/stats/stats.db`
- Output images/annotated images: `work-dir/outputs/`

Key items to check after training:

- `model.pt`
- `model.onnx`
- `model-int8.onnx`
- `model-fp16.onnx` (if fallback succeeds)
- `model_manifest.json`

---

## 4. Minimum Pre-Run Checklist

Before executing any train / autolabel / deploy, check these items first:

### 4.1 Environment

Run in repository root:

```bash
python --version
python -m pip show vision-refactor-project
```

If project is not installed, typically use:

```bash
pip install -e .
```

### 4.2 Configuration File

Confirm existence:

```bash
work-dir/config.toml
```

If it doesn't exist:

```bash
cp work-dir/config.example.toml work-dir/config.toml
```

### 4.3 Key Configuration Items

Check at least these keys:

- `workspace.run_name`
- `class_map.names`
- `class_map.id_map`
- `train.backend`
- `data.yolo_dataset_dir`
- `autolabel.model.onnx_model`
- `deploy.edge.local_model`
- `deploy.remote.model`
- `deploy.edge.stats_endpoint`
- `deploy.edge.stream_endpoint`
- `deploy.statistics.db_path`

### 4.4 LLM Key Configuration

Prioritize using:

- `autolabel.llm.api_key_env_name`
- `deploy.edge.llm.api_key_env_name`

For example:

```toml
[autolabel.llm]
api_key = ""
api_key_env_name = "VISION_LLM_API_KEY"
api_key_env = ""
```

Then in shell:

```bash
export VISION_LLM_API_KEY='your-real-key'
```

Explanation:

- `api_key`: Allows plaintext, but not recommended
- `api_key_env_name`: Recommended, indicates "which environment variable to read"
- `api_key_env`: Legacy compatibility field, old tasks may still use it

---

## 5. Train Quick Operation Guide

## 5.1 Applicable Scenarios

When users say:

- "Help me train a YOLO model"
- "Help me export ONNX"
- "Help me do quantization"

You should prioritize using `train.cli`.

## 5.2 Preferred Backend

Priority recommendations:

1. `yolo`: Most complete, supports training, ONNX, quantization, deployment
2. `faster_rcnn`: Can train, but currently **not suitable as complete deploy main path**, because ONNX/quantization pipeline is not fully integrated

So if the user's goal is deployment, don't default to `faster_rcnn`.

## 5.3 Minimum Training Command

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

If just quickly verifying the workflow, recommend additionally using:

```bash
--set train.dry_run=true
```

## 5.4 Post-Training Checklist

After training completes, prioritize checking:

- `work-dir/models/<run_name>/model_manifest.json`
- `work-dir/models/<run_name>/model-int8.onnx`
- `work-dir/runs/<run_id>/artifacts.json`
- `work-dir/runs/<run_id>/metrics.json`

If connecting training results to subsequent deployment, prioritize writing these paths back to configuration:

- `autolabel.model.onnx_model`
- `deploy.edge.local_model`
- `deploy.remote.model`

Recommend using final artifacts in the directory containing `model_manifest.json`.

---

## 6. AutoLabel Quick Operation Guide

## 6.1 Two Modes

### Model Mode

Suitable for: Local ONNX model available, want to quickly auto-label images.

Key configuration:

- `autolabel.mode = "model"`
- `autolabel.model.onnx_model`
- `autolabel.model.backend`
- `data.unlabeled_dir`
- `data.labeled_dir`

Minimum command:

```bash
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=model \
  --set autolabel.model.backend=yolo \
  --set autolabel.model.onnx_model=./work-dir/models/my-yolo-run/model-int8.onnx \
  --set data.unlabeled_dir=../coco128/images/train2017 \
  --set data.labeled_dir=./work-dir/datasets/labeled
```

### LLM Mode

Suitable for: No suitable model, but want to quickly use vision LLM for initial labeling.

Key configuration:

- `autolabel.mode = "llm"`
- `autolabel.llm.base_url`
- `autolabel.llm.model`
- `autolabel.llm.api_key_env_name`
- `autolabel.llm.prompt`

Minimum command:

```bash
python -m autolabel.cli \
  --config ./work-dir/config.toml \
  --set autolabel.mode=llm \
  --set data.unlabeled_dir=../coco128/images/train2017 \
  --set autolabel.llm.max_images=5
```

## 6.2 Conflict Strategy

Focus on: `autolabel.on_conflict`

- `skip`: Skip if label already exists
- `overwrite`: Directly overwrite
- `merge`: Try to merge

If user has no special requirements, prioritize recommending:

- First run: `skip`
- Explicit refresh: `overwrite`

---

## 7. Statistics Quick Operation Guide

If the user's goal is deploy, typically should start Statistics first.

## 7.1 Recommended Startup Method

Prioritize using script:

```bash
bash scripts/start_stats.sh --config ./work-dir/config.toml
```

Check status:

```bash
bash scripts/status_stats.sh --workdir ./work-dir
```

Stop:

```bash
bash scripts/stop_stats.sh --workdir ./work-dir
```

## 7.2 Access Addresses

Default:

- UI: `http://127.0.0.1:7796`
- API: `http://127.0.0.1:7797`

If user modified ports, refer to `work-dir/config.toml`:

- `deploy.statistics.ui_port`
- `deploy.statistics.api_port`

---

## 8. Deploy Quick Operation Guide

## 8.1 Edge Local

Suitable for: Run ONNX inference directly on local machine, then push statistics to Statistics.

Key configuration:

- `deploy.edge.mode = "local"`
- `deploy.edge.source = "images" | "video" | "camera"`
- `deploy.edge.local_model`
- `deploy.edge.stats_endpoint`

Minimum verification command:

```bash
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=local \
  --set deploy.edge.source=images \
  --set deploy.edge.images_dir=../coco128/images/train2017 \
  --set deploy.edge.local_model=./work-dir/models/my-yolo-run/model-int8.onnx \
  --set deploy.edge.max_frames=1
```

## 8.2 Edge Stream + Remote

Suitable for: Edge sends images, remote performs unified inference.

### Start Remote First

```bash
python -m deploy.remote.cli \
  --config ./work-dir/config.toml \
  --set deploy.remote.model=./work-dir/models/my-yolo-run/model.onnx
```

### Then Start Edge Stream

```bash
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=stream \
  --set deploy.edge.source=images \
  --set deploy.edge.images_dir=../coco128/images/train2017 \
  --set deploy.edge.stream_endpoint=http://127.0.0.1:60051/api/v1/frame \
  --set deploy.edge.max_frames=1
```

### Protocol Key Points in This Mode

Remote's `/api/v1/frame` now returns:

- `request_id`
- `backend`
- `model_id`
- `detections`
- `metadata`

So if users want to troubleshoot "which model is actually working", prioritize checking:

- `model_id` in return payload
- `work-dir/models/<run_name>/model_manifest.json`
- `work-dir/runs/<run_id>/stats.jsonl`

## 8.3 Edge LLM

Suitable for: Edge device directly calls vision LLM for inference.

Minimum command:

```bash
python -m deploy.edge.cli \
  --config ./work-dir/config.toml \
  --set deploy.edge.mode=llm \
  --set deploy.edge.source=images \
  --set deploy.edge.images_dir=../coco128/images/train2017 \
  --set deploy.edge.max_frames=1
```

Prerequisites:

- `deploy.edge.llm.base_url`
- `deploy.edge.llm.model`
- `deploy.edge.llm.api_key_env_name`
- `deploy.edge.llm.prompt`

All configured.

---

## 9. Agent Most Common Task Templates

## 9.1 "Help me quickly run local deployment"

Recommended workflow:

1. Check `work-dir/config.toml`
2. Start Statistics
3. Check if `deploy.edge.local_model` exists
4. Run `deploy.edge.cli` with `images + max_frames=1`
5. Check:
   - Command output
   - `work-dir/runs/<run_id>/artifacts.json`
   - Statistics UI

## 9.2 "Help me connect this model to autolabel"

Recommended workflow:

1. Find `model_manifest.json` in model directory
2. Prioritize choosing `final_infer_model_path` or `model-int8.onnx`
3. Write path to `autolabel.model.onnx_model`
4. Run with `autolabel.mode=model` on small sample for verification

## 9.3 "Help me do remote inference deployment"

Recommended workflow:

1. Start Statistics
2. Start `deploy.remote.cli`
3. Use `deploy.edge.mode=stream` to send 1 frame smoke test
4. Check Remote returns `model_id/backend`
5. Confirm events visible in Statistics

---

## 10. Common Troubleshooting Rules

## 10.1 Configuration Errors

Prioritize checking:

- CLI stderr
- `work-dir/log.txt`
- `ConfigError`

Common causes:

- `class_map.id_map` inconsistent with `class_map.names`
- ONNX model path doesn't exist
- Wrong `data.yolo_dataset_dir`
- LLM missing `api_key_env_name`

## 10.2 Deploy Has No Results

Prioritize checking:

- Is `deploy.edge.max_frames` too small
- Does `deploy.edge.source` match corresponding path
- Is `deploy.edge.confidence` too high
- Are Remote/Statistics started

## 10.3 Statistics Shows No Data

Prioritize checking:

- `deploy.edge.stats_endpoint`
- `deploy.remote.stats_endpoint`
- `scripts/status_stats.sh`
- `work-dir/stats/stats.db`

## 10.4 LLM Call Failure

Prioritize checking:

- Are environment variables exported
- Is `base_url` an OpenAI-compatible endpoint root path
- Is `prompt` empty
- Is `qps_limit` too high

---

## 11. Project Facts You Should Not Misjudge

- **The main operation entry point of this project is not Web, but CLI.** Web is only auxiliary.
- **Configuration source is `work-dir/config.toml`.** Don't hardcode parameters everywhere.
- **If user's goal is deploy, prioritize recommending YOLO, not Faster R-CNN.**
- **Statistics UI and API are two processes.** UI 7796, API 7797 (default).
- **Model directory after training now contains `model_manifest.json`.** Prioritize using it to identify model identity.
- **LLM key recommended via environment variable name.** Don't encourage writing real keys into repository configuration.

---

## 12. Recommended Response Style

When users ask "how to run", your response should prioritize including:

1. Configuration keys that need modification
2. Commands that can be directly executed
3. Where to look after successful run
4. If it fails, what to check first

Concise example:

- First confirm `deploy.edge.local_model` in `work-dir/config.toml` points to an existing ONNX file
- Then run: `python -m deploy.edge.cli --config ./work-dir/config.toml --set deploy.edge.mode=local --set deploy.edge.max_frames=1`
- After success check `work-dir/runs/<run_id>/artifacts.json` and `http://127.0.0.1:7796`

---

## 13. If User Wants to "Quickly Complete Task", Your Priority Order

Priority from high to low:

1. Adjust configuration
2. Provide accurate CLI commands
3. Do smoke test
4. Help user interpret output and logs
5. Only modify code when explicitly needed

If user hasn't requested developing new features, please don't escalate the task into a development task.