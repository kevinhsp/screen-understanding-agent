#  Digital World Agent: Robust Screen Understanding & Decision Pipeline

This project is a prototype Digital World Agent. It takes a raw UI screenshot and produces structured representations of the screen, reasoning traces, and task鈥憃riented decisions. The goal: bridge perception and reasoning for robust UI automation.

---

##  Features
- Screen parsing: detect UI elements (YOLO/OmniParser) + OCR
- Action semantics: VLM (default: Qwen2鈥慥L) infers likely actions and summarizes the page
- Decision agent (optional): an LLM (default: GPT鈥慜SS鈥?0B) selects one best element+action for a task with a transparent thinking trace
- Outputs:
  - Annotated screenshot (`element_actions.png`)
  - Structured JSON with elements + actions (`_output.json`)
  - Decision reasoning trace (`_decision.json`)

---

##  Demo

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

---
##  Installation
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

##  Quickstart (Sessions)

- All platforms:
  - Run: `python session_cli.py`
  - Enter your task when prompted. The tool creates `pipeline_outputs/sessions/session_YYYYMMDD_HHMMSS/`.
  - Drop screenshots named `1.png`, `2.png`, ... into that session folder and press Enter to advance.

- Optional env vars:
  - `DECIDER_MODEL`: decision model id (default `openai/gpt-oss-20b`).
  - `THINKING_ONLY`: if `1` (default) only planning outputs are saved; set `0` to also save candidates and the final decision.

- Linux/macOS (bash):
```bash
export DECIDER_MODEL=openai/gpt-oss-20b
export THINKING_ONLY=0   # set 0 to save candidates + decision
python session_cli.py
```

- Windows PowerShell:
```powershell
$env:DECIDER_MODEL = "openai/gpt-oss-20b"
$env:THINKING_ONLY = "0"   # set 0 to save candidates + decision
python session_cli.py
```

- Alternative: one鈥憃ff image
  - To run the pipeline once on a single image:
```bash
export IMAGE_PATH=examples/united_sample.png
python pipeline.py
```

---

##  Outputs
Primary (session) outputs are saved under `pipeline_outputs/sessions/session_YYYYMMDD_HHMMSS/step_N/`:
- `step_N_input.png`
- `step_N_screen_understanding.json`
- `step_N_element_actions.png` (if available)
- `step_N_thinking.json`
- `step_N_thinking_actions.png`
- `step_N_candidates.json` (when `THINKING_ONLY=0`)
- `step_N_decision.json` (when `THINKING_ONLY=0`)

Session root also contains:
- `task.txt`
- `actions_history.json`
- `thinking_latest.txt`

Alternative (one鈥憃ff image via `pipeline.py`) writes under `pipeline_outputs/`:
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

##  Interactive Session (Terminal)
Run a step鈥慴y鈥憇tep session entirely in the terminal, without manual renaming.

- Entry: `session_cli.py`
- Flow:
  1) Enter the overall task (English).
  2) A session folder is created under `pipeline_outputs/sessions/session_YYYYMMDD_HHMMSS/` and the task is saved to `task.txt`.
  3) For step N, drop `N.png` into the session folder (`1.png`, `2.png`, ...).
  4) The tool processes the screenshot and saves:
     - `step_N_screen_understanding.json`
     - `step_N_element_actions.png` (if available)
     - `step_N_thinking.json` and `step_N_thinking_actions.png`
     - `step_N_candidates.json` and `step_N_decision.json` (when `THINKING_ONLY=0`)
  5) Press Enter to continue; type `q` to quit any time.

- Usage (all platforms):
  - `python session_cli.py`
  - Follow the on鈥憇creen prompts.

- Notes:
  - Env options: `DECIDER_MODEL` (default: `openai/gpt-oss-20b`), `THINKING_ONLY` (default: `1`).
  - Resume later by adding `3.png`, `4.png`, etc. into the same session folder and running the tool again.
  - Uses the same OCR + element detection + VLM + decision agent stack as `pipeline.py`.

---

##  How It Works
1. UI element detection: YOLO/OmniParser finds visual components
2. VLM classification: Qwen2鈥慥L assigns likely actions (click, type, toggle, ...)
3. Screen summary: the VLM generates a concise global description
4. Decision agent: an LLM integrates task + summary + per鈥慹lement actions and outputs a single action with reasoning

---

##  Notes
- First run may download models from Hugging Face. Optional cache path:
  ```bash
  export HF_HOME=./models/huggingface
  ```
- Change the VLM in code via `ProcessingConfig.vlm_model_name`.
- `DW` and `DW_TASK` are interchangeable environment variables.

---

##  Why This Matters
This pipeline is a first step towards robust Digital World AI agents:
- Perception + Reasoning + (Optional) Action Selection
- Transparent reasoning traces (inspectable CoT)
- Extensible to real automation (Playwright / Appium adapters)

It is designed to be:
- Research鈥慺riendly (swap models, log outputs)
- Explainable (stores reasoning traces)
- Agent鈥憆eady (outputs can plug into executors for automation)

---

## Limitations & Future Work
- Current element detection (OmniParser) has limited coverage on diverse UIs
- VLM struggles with fine鈥慻rained text鈥慹lement alignment
- LLM reasoning requires fine鈥憈uning for robustness
- Next step: explore reinforcement learning and better multimodal alignment

---

##  License
MIT. See `LICENSE`.

Third鈥憄arty components retain their own licenses:
- Detector weights/config under `weights/icon_detect/`: see `weights/icon_detect/LICENSE`
- VLM (e.g., `Qwen/Qwen2.5-VL`): Qwen Model License (Alibaba)
- Decision model (e.g., `openai/gpt-oss-20b`): see model card/license
- OCR (PaddleOCR/PaddlePaddle): Apache鈥?.0


