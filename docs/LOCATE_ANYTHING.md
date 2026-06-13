# LocateAnything Runtime Notes

LocateAnything is available through role-local configs:

```text
autolabel/config/config.example.toml  -> [locate_anything]
edge_agent/config/config.example.toml -> [locate_anything]
```

## Auto-label Mode

Run LocateAnything auto-labeling locally:

```bash
python -m autolabel.main --cli \
  --config autolabel/config/config.example.toml \
  --set runtime.mode=locate_anything \
  --set runtime.visualize=true \
  --set locate_anything.device=cuda \
  --set locate_anything.generation_mode=hybrid \
  --set locate_anything.max_images=20 \
  --json-summary
```

## Edge Mode

Run LocateAnything on an edge agent directly:

```bash
python -m edge_agent.cli \
  --config edge_agent/config/config.example.toml \
  --set runtime.mode=locate_anything \
  --set runtime.source=images \
  --set runtime.images_dir=./work-dir/datasets/smoke/images \
  --set runtime.max_frames=20 \
  --set runtime.save_annotated=true \
  --set locate_anything.device=cuda \
  --set locate_anything.generation_mode=hybrid \
  run
```

Submit an edge run through the Control Plane:

```bash
curl -X POST http://127.0.0.1:7800/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "edge_run",
    "payload": {
      "mode": "locate_anything",
      "source": "images",
      "images_dir": "./work-dir/datasets/smoke/images",
      "max_frames": 20,
      "save_annotated": true,
      "locate_anything_device": "cuda",
      "locate_anything_generation_mode": "hybrid"
    }
  }'
```

## Useful Config Fields

```toml
[locate_anything]
model = "nvidia/LocateAnything-3B"
device = "auto"
dtype = "auto"
quantization = "none"
generation_mode = "hybrid"
max_new_tokens = 8192
temperature = 0.0
prompt_template = "Locate all the instances that match the following description: {class_name}."
nms_iou = 0.65
default_score = 1.0
max_images = 0
```

Use `max_images` for auto-label jobs and `runtime.max_frames` for edge runs to limit
smoke tests.
