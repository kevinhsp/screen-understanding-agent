"""Slow test: classify action for ONLY the first element via VLM (English-only).

Run (example):
  RUN_REAL_MODELS=1 \
  OMNIPARSER_WEIGHTS=./models/omniparser/icon_detect.pt \
  QWEN_MODEL='Qwen/Qwen2.5-VL-3B-Instruct' \
  pytest -q tests_slow/test_element_action_single_vlm.py -s -o log_cli=true --log-cli-level=INFO
"""

import os
import json
import asyncio
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from models import ProcessingConfig
from pipeline import ScreenUnderstandingPipeline


RUN_REAL = os.environ.get("RUN_REAL_MODELS") == "1"


def _annotate_one(image: Image.Image, el, action: dict):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    b = el.bbox
    x1, y1, x2, y2 = b.x, b.y, b.x + b.width, b.y + b.height
    color = (255, 0, 0)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    label = f"{el.id[8:]}"
    if action:
        pa = action.get('primary_action')
        desc = action.get('description')
        if pa:
            label += f" — {pa}"
        if desc:
            s = desc if len(desc) <= 24 else (desc[:21] + '...')
            label += f" · {s}"
    try:
        tw = draw.textlength(label, font=font)
        th = font.size if font else 10
    except Exception:
        tw, th = len(label) * 6, 12
    pad = 2
    bg = [x1, max(0, y1 - th - 2 * pad), x1 + int(tw) + 2 * pad, y1]
    draw.rectangle(bg, fill=color)
    draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill=(255, 255, 255), font=font)
    return img


@pytest.mark.skipif(not RUN_REAL, reason="Set RUN_REAL_MODELS=1 to run real-model slow test")
def test_single_element_action(tmp_path):
    t_total_start = time.perf_counter()
    img_path = Path(os.environ.get('IMAGE_PATH') or 'examples/amazon_sample.png')
    assert img_path.exists(), f"Image not found: {img_path}"

    weights = os.environ.get('OMNIPARSER_WEIGHTS','weights/icon_detect/model.pt')
    assert weights and Path(weights).exists(), "OMNIPARSER_WEIGHTS must point to a valid .pt file"

    qwen_model = os.environ.get('QWEN_MODEL') or 'Qwen/Qwen2.5-VL-7B-Instruct'
    use_gpu = os.environ.get('REAL_USE_GPU', '1') in {'1','true','True'}

    cfg = ProcessingConfig(
        use_gpu=use_gpu,
        omniparser_weights_path=weights,
        omniparser_allow_download=False,
        element_detection_threshold=0.15,
        iou_threshold=0.2,
        vlm_model_name=qwen_model,
        vlm_elements_max=1,
        debug_mode=True,
    )
    image = Image.open(img_path).convert('RGB')
    # t_vlm_start = time.perf_counter()
    pipe = ScreenUnderstandingPipeline(cfg)
    summary_actions = asyncio.run(pipe._run_vlm_generate_summary_actions(image))
    # Step 1
    t_ocr_start = time.perf_counter()
    ocr = asyncio.run(pipe._run_ocr(image))
    t_ocr = time.perf_counter() - t_ocr_start
    # Step 2
    t_detect_start = time.perf_counter()
    elems = asyncio.run(pipe._detect_elements(image))
    t_detect = time.perf_counter() - t_detect_start
    # Step 3
    t_merge_start = time.perf_counter()
    merged = asyncio.run(pipe._merge_ocr_elements(elems, ocr)) 
    t_merge = time.perf_counter() - t_merge_start
    print(f"Merged elements: {len(merged)}")
    # Pick the first candidate: prefer clickable; else first by reading order
    def order_key(e):
        return (e.bbox.y, e.bbox.x)
    clickable = sorted([e for e in merged if getattr(e, 'clickable', False)], key=order_key)
    if clickable:
        target = clickable[0]
    else:
        target = sorted(merged, key=order_key)[0]
    target.id = target.id or 'element_0'
    print(f"Target element: {target.id}, role={target.role.value}, text={target.text!r}")

    actions = asyncio.run(pipe._run_vlm_classify_element_actions(image, [target], ocr, summary_actions,max_k=1))
    # t_vlm = time.perf_counter() - t_vlm_start
    print("VLM raw actions:", actions)

    t_total = time.perf_counter() - t_total_start
    print(f"Timing: OCR={t_ocr:.3f}s, Detect={t_detect:.3f}s, Merge={t_merge:.3f}s, Total={t_total:.3f}s")

    # Save JSON
    side_json = img_path.with_suffix('.first_element_action.json')
    side_json.write_text(json.dumps(actions, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved first-element actions JSON: {side_json}")

    # Annotate PNG
    action_for_target = actions[0] if actions else {}
    annotated = _annotate_one(image, target, action_for_target)
    side_png = img_path.with_suffix('.first_element_action.png')
    annotated.save(side_png, format='PNG')
    print(f"Saved first-element actions PNG: {side_png}")
    assert isinstance(summary_actions, dict)
    # assert isinstance(actions, list)
