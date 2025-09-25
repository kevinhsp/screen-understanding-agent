# Digital World Agent: Robust Screen Understanding & Decision Pipeline

This project is a prototype Digital World Agent. It takes a raw UI screenshot and produces structured representations of the screen, reasoning traces, and task-oriented decisions. The goal is to bridge perception and reasoning for robust UI automation.

---

## Features
- Screen parsing: detect UI elements (YOLO/OmniParser) + OCR
- Action semantics: VLM (default: Qwen2-VL) infers likely actions and summarizes the page
- Decision agent: an LLM (default: GPT-OSS-20B) selects one best element+action for a task with a transparent thinking trace
- Outputs:
  - Annotated screenshot (`element_actions.png`)
  - Structured JSON with elements + actions (`_output.json`)
  - Decision reasoning trace (`_decision.json`)
Custom prompting & algorithmic heuristics: Specially designed task-specific prompts and logic rules significantly improve reliability, beyond naïve model calls.
---

## Demo

- Task: `find flights from boston to la on 2025/10/5 and back on 2025/10/8`
- Session folder: `pipeline_outputs/sessions/session_20250924_151105`
- Preview (Step 1 actions):

![Demo Step 1 Actions](pipeline_outputs/sessions/session_20250924_151105/step_1/step_1_thinking_actions.png)

- Thinking excerpt (`step_1_thinking.json`):

```json
{
  "plan": "Set departure and destination, choose round-trip, enter travel dates, and search for flights.",
  "steps": [
    {"element_id": "element_6", "actions": ["click","type"], "details": "Enter departure location as Boston"},
    {"element_id": "element_5", "actions": ["click","type"], "details": "Enter destination as Los Angeles"},
    {"element_id": "element_11", "actions": ["click"], "details": "Select round-trip travel option"},
    {"element_id": "element_0", "actions": ["click","type"], "details": "Open dates picker and enter departure date 2025/10/5"},
    {"element_id": "element_0", "actions": ["type"], "details": "Enter return date 2025/10/8"},
    {"element_id": "element_2", "actions": ["click"], "details": "Initiate flight search"}
  ]
}
```

- More outputs (JSON, images) are available in `pipeline_outputs/sessions/session_20250924_151105`.

- Note: The outputs are not just raw model results. Each step involves custom prompting strategies and additional algorithms to refine element selection, enforce task constraints, and ensure robustness.
---

## Installation
Prereqs: Python 3.9, Conda. GPU strongly recommended.

```bash
conda env create -f environment.yml
conda activate screen-understanding
```

Place YOLO weights at:
```
weights/icon_detect/model.pt
```

---

## Quickstart (Sessions)

- All platforms:
  - Run: `python session_cli.py`
  - Enter your task when prompted. The tool creates `pipeline_outputs/sessions/session_YYYYMMDD_HHMMSS/`.
  - Drop screenshots named `1.png`, `2.png`, ... into that session folder and press Enter to advance.

- Optional env vars:
  - `DECIDER_MODEL`: decision model id (default `openai/gpt-oss-20b`).

- Linux/macOS (bash):
```bash
export DECIDER_MODEL=openai/gpt-oss-20b
python session_cli.py
```

- Windows PowerShell:
```powershell
$env:DECIDER_MODEL = "openai/gpt-oss-20b"
python session_cli.py
```

- Alternative: one-off image
  - To run the pipeline once on a single image:
```bash
export IMAGE_PATH=examples/united_sample.png
python pipeline.py
```

---

## Outputs
Primary (session) outputs are saved under `pipeline_outputs/sessions/session_YYYYMMDD_HHMMSS/step_N/`:
- `step_N_input.png`
- `step_N_screen_understanding.json`
- `step_N_element_actions.png` (if available)
- `step_N_thinking.json`
- `step_N_thinking_actions.png`

Session root also contains:
- `task.txt`
- `actions_history.json`
- `thinking_latest.txt`

Alternative (one-off image via `pipeline.py`) writes under `pipeline_outputs/`:
- `<image>_screen_understanding_output.json`
- `<image>_element_actions.png`
- `<image>_decision.json` (when `DW`/`DW_TASK` is set)

Example `step_N_decision.json`:
```json
{
  "task": "Select LAX as departure city",
  "decision": {
    "thoughts": "Prefer the From field labeled LAX.",
    "element_id": "element_12",
    "action": "click",
    "model": "openai/gpt-oss-20b"
  }
}
```

---


## Notes
- First run may download models from Hugging Face. Optional cache path:
  ```bash
  export HF_HOME=./models/huggingface
  ```
- Change the VLM in code via `ProcessingConfig.vlm_model_name`.
- `DW` and `DW_TASK` are interchangeable environment variables.

---

## Why This Matters
This pipeline is a first step towards robust Digital World AI agents:
- Perception + Reasoning + (Optional) Action Selection
- Transparent reasoning traces (inspectable CoT)
- Extensible to real automation (Playwright / Appium adapters)

It is designed to be:
- Research-friendly (swap models, log outputs)
- Explainable (stores reasoning traces)
- Agent-ready (outputs can plug into executors for automation)

---

## Limitations & Future Work
- Current element detection (OmniParser) has limited coverage on diverse UIs
- VLM struggles with fine-grained text-element alignment
- LLM reasoning requires fine-tuning for robustness
- Next step: explore reinforcement learning and better multimodal alignment

---

## License
MIT. See `LICENSE`.

Third-party components retain their own licenses:
- Detector weights/config under `weights/icon_detect/`: see `weights/icon_detect/LICENSE`
- VLM (e.g., `Qwen/Qwen2.5-VL`): Qwen Model License (Alibaba)
- Decision model (e.g., `openai/gpt-oss-20b`): see model card/license
- OCR (PaddleOCR/PaddlePaddle): Apache-2.0

