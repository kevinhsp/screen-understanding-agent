"""Slow test: DecisionAgent selects one element/action from pipeline output.

Run (example):
  RUN_REAL_MODELS=1 \
  DECIDER_MODEL=openai/gpt-oss-20b \
  REAL_USE_GPU=1 \
  pytest -q tests_slow/test_decision_agent.py -s -o log_cli=true --log-cli-level=INFO

This test loads the minimal screen understanding JSON produced by the pipeline
and asks the decision agent to choose a single element and action for a task.
"""

import os
import json
from pathlib import Path

import pytest

from decision_agent import DecisionAgent
from utils import vram_snapshot

RUN_REAL = os.environ.get("RUN_REAL_MODELS") == "1"


@pytest.mark.skipif(not RUN_REAL, reason="Set RUN_REAL_MODELS=1 to run real-model slow test")
def test_decision_agent_on_united_sample():
    # Locate the pipeline output JSON
    json_path = Path('pipeline_outputs/united_sample_screen_understanding_output.json')
    assert json_path.exists(), f"Pipeline output not found: {json_path}. Run the pipeline first."
    base = vram_snapshot("baseline")
    understanding = json.loads(json_path.read_text(encoding='utf-8'))

    # Task from env or default
    task = os.environ.get('DW_TASK') or 'Select LAX as departure city'
    model_id = os.environ.get('DECIDER_MODEL') or 'openai/gpt-oss-20b'
    use_gpu = os.environ.get('REAL_USE_GPU', '1') in {'1','true','True'}

    agent = DecisionAgent(model_name=model_id, use_gpu=use_gpu)
    decision = agent.decide(understanding, task)
    after_gpt = vram_snapshot("after_gpt")
    gpt_inc_mb = after_gpt["allocated_mb"] - base["allocated_mb"]
    print(f"[VRAM] GPT incremental ≈ {gpt_inc_mb:.0f} MB")
    print("Task:", task)
    print("Decision:", json.dumps(decision, ensure_ascii=False))

    # Basic validations
    assert isinstance(decision, dict)
    assert 'element_id' in decision and 'action' in decision
    assert isinstance(decision['element_id'], str)
    assert isinstance(decision['action'], str)
    allowed = {"click","type","toggle","open","navigate","select","none"}
    assert decision['action'] in allowed

    # Save decision next to the pipeline output for inspection
    out = json_path.with_name('united_sample_decision.from_test.json')
    out.write_text(json.dumps({'task': task, 'decision': decision}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved decision to {out}")

