#  Digital World Agent: Robust Screen Understanding & Decision Pipeline

This project is a **prototype Digital World Agent**. It takes a **raw UI screenshot** and produces structured representations of the screen, reasoning traces, and even task-oriented decisions. The goal: **bridge perception and reasoning for robust UI automation**.

---

##  Features
- **Screen Parsing** → Detect UI elements (YOLO / OmniParser) + OCR.
- **Action Semantics** → VLM (default: Qwen2-VL) infers each element’s likely action(s) and provides a page summary.
- **Decision Agent (optional)** → An LLM (default: GPT-OSS-20B) reasons over the elements and picks *one best element+action* for a given task, outputting a transparent “thinking trace.”
- **Outputs** →
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
**Prereqs**: Python ≥3.9, Conda. GPU strongly recommended.  

```bash
conda env create -f environment.yml
conda activate screen-understanding
```

Place YOLO weights at:  
```
weights/icon_detect/model.pt
```

---

##  Quickstart (One-liners)

**Linux/macOS (bash):**
```bash
export IMAGE_PATH=examples/united_sample.png
export DW="Select LAX as departure city"   # optional task
export DECIDER_MODEL=openai/gpt-oss-20b    # optional, default is this
python pipeline.py
```

**Windows PowerShell:**
```powershell
$env:IMAGE_PATH = "examples/united_sample.png"
$env:DW = "Select LAX as departure city"
$env:DECIDER_MODEL = "openai/gpt-oss-20b"
python pipeline.py
```

---


##  Outputs
Saved under `pipeline_outputs/`:
- `<image>_screen_understanding_output.json`  
- `<image>_element_actions.png`  
- `<image>_decision.json` (if `DW`/`DW_TASK` is set)  

**Example decision trace:**
```json
{
  "task": "Select LAX as departure city",
  "thinking": "The task is to select LAX. The detected dropdown shows airports. The element labeled 'From: LAX' is the best match.",
  "action": {"id": 12, "role": "dropdown", "text": "LAX"}
}
```

---

##  Interactive Session (Terminal)
Run a step-by-step session entirely in the terminal, without manual renaming each time.

- Entry: `session_cli.py`
- Flow:
  1) Enter the overall task (English).
  2) A session folder is created under `pipeline_outputs/sessions/session_YYYYMMDD_HHMMSS/` and the task is saved to `task.txt`.
  3) For step N, drop `N.png` into the session folder (`1.png`, `2.png`, ...).
  4) The script processes the screenshot and saves:
     - `step_N_screen_understanding.json`
     - `step_N_element_actions.png` (if annotation is available)
     - `step_N_decision.json`
  5) Press Enter to continue; type `q` to quit any time.

- Usage (all platforms):
  - `python session_cli.py`
  - Follow the on‑screen prompts.

- Notes:
  - Override the decision model with `DECIDER_MODEL` (default: `openai/gpt-oss-20b`).
  - You can resume later by adding `3.png`, `4.png`, etc. into the same session folder and running the tool again.
  - Uses the same OCR → element detection → VLM pipeline and the same decision agent as `pipeline.py`.

---

##  How It Works
1. **UI Element Detection** → YOLO / OmniParser finds visual components.  
2. **VLM Classification** → Qwen2-VL assigns semantic actions (click, type, toggle…).  
3. **Screen Summary** → VLM generates a concise global description.  
4. **Decision Agent** → LLM integrates task + summary + per-element actions, and outputs a **single action with reasoning**.  

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
This pipeline is a **first step towards robust Digital World AI agents**:  
- Perception + Reasoning + (Optional) Action Selection  
- Transparent reasoning traces (inspectable CoT)  
- Extensible to real automation (Playwright / Appium adapters)  

It is designed to be:  
- **Research-friendly** (swap models, log outputs)  
- **Explainable** (stores reasoning traces)  
- **Agent-ready** (outputs can plug into executors for automation)  

---

## Limitations & Future Work
- Current element detection (OmniParser) has limited coverage on diverse UIs.
- VLM struggles with fine-grained text-element alignment.
- LLM reasoning requires fine-tuning for robustness.
- Next step: explore reinforcement learning and better multimodal alignment.

---

##  License
MIT. See `LICENSE`.

Third-party components retain their own licenses:
- Detector weights/config under `weights/icon_detect/`: see `weights/icon_detect/LICENSE`
- VLM (e.g., `Qwen/Qwen2.5-VL`): Qwen Model License (Alibaba)
- Decision model (e.g., `openai/gpt-oss-20b`): see model card/license
- OCR (PaddleOCR/PaddlePaddle): Apache-2.0
