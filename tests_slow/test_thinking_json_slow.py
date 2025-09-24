import json
import os
from pathlib import Path
import pytest


@pytest.mark.slow
def test_generate_thinking_from_screen_understanding():
    """
    Slow test: given a screen_understanding.json, run the real DecisionAgent
    to generate a thinking JSON next to it.

    Environment:
      - DW_UNDERSTANDING / SCREEN_UNDERSTANDING / UNDERSTANDING_JSON: input JSON path (required)
      - DW_TASK / TASK: natural language task (optional; default: 'Plan next steps')
      - DECIDER_MODEL: HF model id for DecisionAgent (optional)
      - DW_IMAGE / SCREEN_IMAGE / IMAGE_PATH / SCREENSHOT: original screenshot (optional; if set, also render annotated actions PNG)
    """
    # Resolve input path
    input_path = (
        os.environ.get("DW_UNDERSTANDING")
        or os.environ.get("SCREEN_UNDERSTANDING")
        or os.environ.get("UNDERSTANDING_JSON")
    )
    if not input_path:
        pytest.skip("Set DW_UNDERSTANDING (or SCREEN_UNDERSTANDING/UNDERSTANDING_JSON) to a screen_understanding.json path")
    inp = Path(input_path)
    assert inp.exists(), f"Input file not found: {inp}"

    # Load understanding
    understanding = json.loads(inp.read_text(encoding="utf-8"))

    # Determine task
    task = os.environ.get("DW_TASK") or os.environ.get("TASK") or "book a flight from boston to la on 10/5 and back on 10/8"

    # Optional session context
    prev_thinking = None
    actions_history = None
    try:
        # Look for sibling files commonly produced by session_cli
        base_dir = inp.parent
        tfile = base_dir.parent / "thinking_latest.txt"
        if tfile.exists():
            prev_thinking = tfile.read_text(encoding="utf-8").strip() or None
        hfile = base_dir.parent / "actions_history.json"
        if hfile.exists():
            actions_history = json.loads(hfile.read_text(encoding="utf-8"))
    except Exception:
        pass

    # Import real agent (requires transformers/torch available)
    try:
        from decision_agent import DecisionAgent
    except Exception as e:
        pytest.skip(f"decision_agent import failed: {e}")

    model_id = os.environ.get("DECIDER_MODEL") or "openai/gpt-oss-20b"
    agent = DecisionAgent(model_name=model_id, use_gpu=True)

    # Run planning
    obj = agent.plan(understanding, task, actions_history=actions_history, prev_thinking=prev_thinking, max_new_tokens=1200)
    assert isinstance(obj, dict), "plan() did not return dict"
    # assert set(["plan", "steps", "success_criteria"]).issubset(obj.keys()), "missing keys in thinking output"

    # Save next to the input file
    out_path = inp.with_name("thinking.json")
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    assert out_path.exists(), f"Failed to write thinking JSON: {out_path}"

    # Optional: render annotated actions image when an input screenshot is provided
    img_env = (
        os.environ.get("DW_IMAGE")
        or os.environ.get("SCREEN_IMAGE")
        or os.environ.get("IMAGE_PATH")
        or os.environ.get("SCREENSHOT")
    )
    if img_env:
        from PIL import Image
        from decision_agent import DecisionAgent as _DA
        img_path = Path(img_env)
        assert img_path.exists(), f"Image not found: {img_path}"
        img = Image.open(img_path).convert("RGB")
        ann = _DA.render_thinking_actions(img, understanding, obj)
        ann_path = out_path.with_name("thinking_actions.png")
        ann.save(ann_path, format="PNG")
        assert ann_path.exists(), f"Failed to write annotated actions PNG: {ann_path}"
