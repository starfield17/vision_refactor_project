# LocateAnything Integration

This project treats LocateAnything as a **grounding backend**, not as a replacement for the
existing YOLO/Faster-RCNN ONNX path. It is best suited for two workflows:

1. **AutoLabel bootstrap** — use natural-language class prompts to produce initial boxes, then
   review/correct labels and train a smaller production detector.
2. **GPU-backed deploy** — use open-vocabulary localization online when class definitions change
   often and the deployment node has enough GPU memory/latency budget.

For fixed classes, high FPS, edge-only deployment, keep using `deploy.edge.mode="local"` with a
trained ONNX model.

## Configuration

LocateAnything settings are shared by AutoLabel and edge deploy:

```toml
[locate_anything]
model = "nvidia/LocateAnything-3B"
device = "auto"                 # "auto" | "cuda" | "cuda:0" | "cpu"
dtype = "auto"                  # "auto" | "float16" | "bfloat16" | "float32"
generation_mode = "hybrid"      # "fast" | "slow" | "hybrid"
max_new_tokens = 8192
temperature = 0.0
prompt_template = "Locate all the instances that match the following description: {class_name}."
nms_iou = 0.65
default_score = 1.0
verbose = false
max_images = 0                   # 0 = no limit
```

The `prompt_template` must include `{class_name}`. Each item in `class_map.names` is rendered
through this template and queried independently.

## AutoLabel

```bash
python -m autolabel.cli   --config ./work-dir/config.toml   --set autolabel.mode=locate_anything   --set autolabel.visualize=true   --set locate_anything.device=cuda   --set locate_anything.generation_mode=hybrid   --set locate_anything.max_images=20
```

Outputs follow the existing AutoLabel layout:

- YOLO-format label files under `data.labeled_dir`.
- Optional annotated previews under `work-dir/outputs/<run_id>/annotated_frames`.
- Raw LocateAnything responses under `work-dir/outputs/<run_id>/locate_anything_raw`.

## Edge deploy

```bash
python -m deploy.edge.cli   --config ./work-dir/config.toml   --set deploy.edge.mode=locate_anything   --set deploy.edge.source=images   --set deploy.edge.images_dir=./work-dir/datasets/smoke/images   --set deploy.edge.max_frames=20   --set deploy.edge.save_annotated=true   --set locate_anything.device=cuda   --set locate_anything.generation_mode=hybrid
```

Statistics events use:

```text
backend = "locate_anything"
model_id = "locate_anything:<model>"
```

## UI support

- React Train & AutoLabel UI includes an **AutoLabel / LocateAnything** preset.
- React Deploy & Statistics UI includes an **Edge / LocateAnything** preset.
- The PySide6 AutoLabel page includes a LocateAnything configuration group and can persist
  `[locate_anything]` settings into `work-dir/config.toml`.

## Operational guidance

- Start with `locate_anything.max_images=20` or another small cap for smoke testing.
- Use `generation_mode="hybrid"` by default.
- Use `generation_mode="fast"` for exploratory throughput tests.
- Use `generation_mode="slow"` when debugging formatting or dense-object failures.
- Keep `temperature=0.0` for reproducible label generation.
- Treat `default_score` as a compatibility score; LocateAnything does not behave like a
  calibrated YOLO confidence head.

## Recommended production pattern

1. Run LocateAnything AutoLabel on a representative image subset.
2. Inspect raw responses and annotated previews.
3. Correct labels manually or with a review tool.
4. Train YOLO/Faster-RCNN on reviewed labels.
5. Deploy ONNX locally for fixed-class high-FPS paths; reserve LocateAnything deploy for
   open-vocabulary or frequently changing classes.
