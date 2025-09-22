# Screen Understanding System

Input a UI screenshot → get:
- an annotated PNG with clickable elements labeled, and
- a minimal JSON describing each element’s likely action(s) and a page summary.

Optionally, a decision agent (GPT‑4‑OSS style LLM) picks exactly one element+action for a given task.

## Install

Prereqs: Python 3.9, Conda. GPU recommended.

```
conda env create -f environment.yml
conda activate screen-understanding
# If needed for PaddleOCR
# GPU: pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
# CPU: pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Place YOLO weights at `weights/icon_detect/model.pt` (or set `OMNIPARSER_WEIGHTS`/`omniparser_weights_path`).

## Run (one‑liners)

Linux/macOS (bash):
```
export IMAGE_PATH=examples/united_sample.png
export DW="Select LAX as departure city"   # optional task for decision
export DECIDER_MODEL=openai/gpt-oss-20b    # optional, default is this
python pipeline.py
```

Windows PowerShell:
```
$env:IMAGE_PATH = "examples/united_sample.png"
$env:DW = "Select LAX as departure city"
$env:DECIDER_MODEL = "openai/gpt-oss-20b"
python pipeline.py
```

Outputs are saved to `pipeline_outputs/`:
- `<image>_screen_understanding_output.json`
- `<image>_element_actions.png`
- `<image>_decision.json` (if `DW`/`DW_TASK` is set)

## What It Does

- OCR + UI element detection (OmniParser/YOLO) to find on‑screen elements.
- VLM (default: Qwen2‑VL) to classify likely actions per element and summarize the screen.
- Decision agent (default model: `openai/gpt-oss-20b`) uses the summary + per‑element actions to choose one element/action that best satisfies your task.

That’s it—install, set input image and optional task with env vars, run once, and inspect the JSON/PNG outputs.

## Notes

- Models download from Hugging Face on first use. Optional cache:
```
export HF_HOME=./models/huggingface
```
- You can change the VLM in code via `ProcessingConfig.vlm_model_name`.
- `DW` and `DW_TASK` are both accepted for the task variable.

## License

AGPL-3.0-only. See `LICENSE`.

Third‑party components retain their own licenses:
- Detector (Ultralytics/YOLO icon_detect): AGPL‑3.0 or commercial license
- VLM (e.g., Qwen/Qwen2.5‑VL): Qwen Model License (Alibaba)
- Decision model (e.g., openai/gpt‑oss‑20b): see model card/license
- OCR (PaddleOCR/PaddlePaddle): Apache‑2.0
