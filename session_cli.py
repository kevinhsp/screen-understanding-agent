"""Interactive session runner for multi-step screenshots (English-only prompts).

Flow:
  1) Ask the user for the task (English)
  2) Create a new session folder under pipeline_outputs/sessions/
  3) For step N, the user drops N.png into the session folder
  4) Run the pipeline and decision agent; save step_N_* outputs
  5) Repeat for N+1 until the user quits
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from PIL import Image

from models import ProcessingConfig
from pipeline import ScreenUnderstandingPipeline
from decision_agent import DecisionAgent


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("\n==== Screen Understanding Session Runner ====")
    print("This is an interactive terminal runner. One task = one session.")
    print("Flow: 1) enter task (English) → 2) place 1.png → 3) results → 4) place 2.png → …; type 'q' anytime to quit.\n")

    # 1) Ask for task (English)
    task = ask("Please enter the overall task in English (or 'q' to quit): ").strip()
    if not task or task.lower() in {"q", "quit", "exit"}:
        print("Bye.")
        return 0

    # 2) Create session folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sessions_root = Path("pipeline_outputs") / "sessions"
    ensure_dir(sessions_root)
    session_dir = sessions_root / f"session_{ts}"
    ensure_dir(session_dir)
    (session_dir / "task.txt").write_text(task, encoding="utf-8")
    print(f"Session created: {session_dir}")
    print("Please place step screenshots named '1.png', '2.png', ... into this session folder.\n")

    # 3) Init pipeline and decision agent (lazy)
    cfg = ProcessingConfig(
        use_gpu=True,
        debug_mode=True,
        save_intermediate_results=False,
    )
    pipeline = ScreenUnderstandingPipeline(cfg)
    agent: Optional[DecisionAgent] = None
    dec_model = os.environ.get("DECIDER_MODEL") or getattr(cfg, "decider_model_name", "openai/gpt-oss-20b")

    step = 1
    while True:
        img_name = f"{step}.png"
        img_path = session_dir / img_name

        # 4) Wait for screenshot or quit
        print(f"[Step {step}] Put the screenshot at: {img_path}")
        cmd = ask("Press Enter when ready, or type 'q' to quit: ").strip().lower()
        if cmd in {"q", "quit", "exit"}:
            print("Session ended.")
            break

        if not img_path.exists():
            print("Image not found. Please check file name and location, then try again.\n")
            continue

        # 5) Process image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image: {e}")
            continue

        print(f"Processing {img_name} …")
        try:
            understanding = asyncio.run(pipeline.process(img, str(img_path)))
        except Exception as e:
            print(f"Processing failed: {e}")
            continue

        # 6) Save outputs into the session
        step_prefix = session_dir / f"step_{step}"
        ensure_dir(step_prefix)
        # Save understanding JSON
        out_understanding = step_prefix.with_name(f"step_{step}_screen_understanding.json")
        Path(out_understanding).write_text(json.dumps(understanding, ensure_ascii=False, indent=2), encoding="utf-8")

        # Copy annotated PNG from pipeline_outputs if available
        basename = img_path.stem
        anno_src = Path("pipeline_outputs") / f"{basename}_element_actions.png"
        anno_dst = step_prefix.with_name(f"step_{step}_element_actions.png")
        try:
            if anno_src.exists():
                anno_dst.write_bytes(anno_src.read_bytes())
        except Exception:
            pass

        # 7) Decision (init once)
        decision_obj = {}
        try:
            agent = getattr(pipeline, 'decision_agent', None)
            if agent is None:
                agent = DecisionAgent(model_name=dec_model, use_gpu=cfg.use_gpu)
            decision_obj = agent.decide(understanding, task)
        except Exception as e:
            decision_obj = {"error": f"decision failed: {e}"}

        dec_path = step_prefix.with_name(f"step_{step}_decision.json")
        Path(dec_path).write_text(json.dumps({"task": task, "decision": decision_obj}, ensure_ascii=False, indent=2), encoding="utf-8")

        # 8) Summary
        print("\n=== Step Completed ===")
        print(f"Understanding: {out_understanding}")
        if Path(anno_dst).exists():
            print(f"Annotation: {anno_dst}")
        print(f"Decision: {dec_path}")
        if isinstance(decision_obj, dict) and decision_obj.get("action"):
            print(f"Suggested action: {decision_obj.get('action')} on {decision_obj.get('element_id')}")
        print("=================\n")

        step += 1

    print(f"Session folder: {session_dir}")
    print("To continue this session later, run this tool again and add subsequent images (3.png, 4.png, …) to this folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

