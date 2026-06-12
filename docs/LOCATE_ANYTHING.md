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
quantization = "none"           # "none" | "bnb_4bit"
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"     # "nf4" | "fp4"
bnb_4bit_use_double_quant = true
device_map = ""                 # use "auto" for bnb_4bit
attn_implementation = ""        # use "sdpa" on RTX 4060 / non-Hopper GPUs
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

## Local deployment

LocateAnything uses Hugging Face `transformers` remote code and downloads the
`nvidia/LocateAnything-3B` model into the local Hugging Face cache on first use. The
model repository ships full safetensors shards, not separate Q4 weights, so Q4 is applied
at load time through bitsandbytes.

Install the runtime dependencies in the Python environment used to run this project:

```bash
python -m pip install --upgrade \
  'bitsandbytes' \
  'transformers==4.57.1' \
  'accelerate==1.5.2' \
  'tokenizers==0.22.0'
```

If Hugging Face/Xet downloads are unstable, disable Xet and prefetch only runtime files:

```bash
HF_HUB_DISABLE_XET=1 hf download \
  nvidia/LocateAnything-3B \
  --include '*.json' '*.py' '*.txt' '*.model' 'merges.txt' 'vocab.json' \
  'added_tokens.json' 'special_tokens_map.json' 'tokenizer_config.json' \
  'model-*.safetensors'
```

Use a one-image, one-class smoke before running a full class map:

```bash
HF_HUB_DISABLE_XET=1 python - <<'PY'
from pathlib import Path
from share.application.autolabel_service import run_autolabel

result = run_autolabel(Path("work-dir/config.toml"), overrides=[
    "workspace.run_name=locateanything_q4_smoke",
    "class_map.names=['person']",
    "class_map.id_map={'person': 0}",
    "autolabel.mode=locate_anything",
    "autolabel.confidence=0.3",
    "autolabel.visualize=true",
    "autolabel.on_conflict=overwrite",
    "locate_anything.quantization=bnb_4bit",
    "locate_anything.bnb_4bit_compute_dtype=float16",
    "locate_anything.bnb_4bit_quant_type=nf4",
    "locate_anything.bnb_4bit_use_double_quant=true",
    "locate_anything.device=cuda:0",
    "locate_anything.dtype=float16",
    "locate_anything.device_map=auto",
    "locate_anything.attn_implementation=sdpa",
    "locate_anything.generation_mode=fast",
    "locate_anything.max_images=1",
    "locate_anything.max_new_tokens=512",
])
print(result)
PY
```

Inspect the returned `run_id`, then check:

- Raw responses under `work-dir/outputs/<run_id>/locate_anything_raw`.
- Optional annotated previews under `work-dir/outputs/<run_id>/annotated_frames`.
- Label JSON files under the configured `data.labeled_dir`.

For small GPUs, keep `max_images=1`, `max_new_tokens=512`, and one or a few class names
for the first smoke. Use `attn_implementation="sdpa"` on non-Hopper/Blackwell GPUs;
Magi Attention is for Hopper/Blackwell-class systems.

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
- On 8 GB GPUs, start with `locate_anything.quantization="bnb_4bit"`,
  `locate_anything.device_map="auto"`, one image, and one class.
- Use `generation_mode="hybrid"` by default.
- Use `generation_mode="fast"` for exploratory throughput tests.
- Use `generation_mode="slow"` when debugging formatting or dense-object failures.
- Keep `temperature=0.0` for reproducible label generation.
- Treat `default_score` as a compatibility score; LocateAnything does not behave like a
  calibrated YOLO confidence head.
- The NVIDIA model license permits academic and non-profit research use only; review
  the Hugging Face model card before production or commercial use.

## Troubleshooting

- `bitsandbytes` import or CUDA errors: confirm the environment can import `bitsandbytes`,
  `torch.cuda.is_available()` is true, and the installed bitsandbytes wheel supports the
  local GPU/CUDA runtime.
- Hugging Face SSL/Xet failures: retry with `HF_HUB_DISABLE_XET=1` and the include-filtered
  `hf download` command above.
- CUDA OOM: close other GPU processes, keep `bnb_4bit`, reduce `max_new_tokens`, and run
  one image/class at a time.
- Malformed or empty outputs: first try `generation_mode="hybrid"`; inspect raw answers
  under `locate_anything_raw` before changing prompts.

## Recommended production pattern

1. Run LocateAnything AutoLabel on a representative image subset.
2. Inspect raw responses and annotated previews.
3. Correct labels manually or with a review tool.
4. Train YOLO/Faster-RCNN on reviewed labels.
5. Deploy ONNX locally for fixed-class high-FPS paths; reserve LocateAnything deploy for
   open-vocabulary or frequently changing classes.
