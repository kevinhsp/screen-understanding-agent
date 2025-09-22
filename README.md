# 🚀 Digital World Agent: Robust Screen Understanding & Decision Pipeline

This project is a **prototype Digital World Agent**. It takes a **raw UI screenshot** and produces structured representations of the screen, reasoning traces, and even task-oriented decisions. The goal: **bridge perception and reasoning for robust UI automation**.

---

## ✨ Features
- **Screen Parsing** → Detect UI elements (YOLO / OmniParser) + OCR.
- **Action Semantics** → VLM (default: Qwen2-VL) infers each element’s likely action(s) and provides a page summary.
- **Decision Agent (optional)** → An LLM (default: GPT-OSS-20B) reasons over the elements and picks *one best element+action* for a given task, outputting a transparent “thinking trace.”
- **Outputs** →
  - Annotated screenshot (`element_actions.png`)
  - Structured JSON with elements + actions (`_output.json`)
  - Decision reasoning trace (`_decision.json`)

---

## 🔧 Installation
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

## ▶️ Quickstart (One-liners)

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

## 📂 Outputs
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

## 🧠 How It Works
1. **UI Element Detection** → YOLO / OmniParser finds visual components.  
2. **VLM Classification** → Qwen2-VL assigns semantic actions (click, type, toggle…).  
3. **Screen Summary** → VLM generates a concise global description.  
4. **Decision Agent** → LLM integrates task + summary + per-element actions, and outputs a **single action with reasoning**.  

---

## 📌 Notes
- First run may download models from Hugging Face. Optional cache path:  
  ```bash
  export HF_HOME=./models/huggingface
  ```
- Change the VLM in code via `ProcessingConfig.vlm_model_name`.  
- `DW` and `DW_TASK` are interchangeable environment variables.  

---

## 🌍 Why This Matters
This pipeline is a **first step towards robust Digital World AI agents**:  
- Perception + Reasoning + (Optional) Action Selection  
- Transparent reasoning traces (inspectable CoT)  
- Extensible to real automation (Playwright / Appium adapters)  

It is designed to be:  
- **Research-friendly** (swap models, log outputs)  
- **Explainable** (stores reasoning traces)  
- **Agent-ready** (outputs can plug into executors for automation)  

---

## 📜 License
AGPL-3.0-only. See `LICENSE`.  

Third-party components retain their own licenses:  
- Detector (Ultralytics/YOLO icon_detect): AGPL-3.0 or commercial license  
- VLM (e.g., Qwen/Qwen2.5-VL): Qwen Model License (Alibaba)  
- Decision model (e.g., openai/gpt-oss-20b): see model card/license  
- OCR (PaddleOCR/PaddlePaddle): Apache-2.0  