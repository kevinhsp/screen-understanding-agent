"""Slow test: YOLO -> OCR merge -> VLM action classification per element (English-only).

Outputs next to the image:
  - <image>.element_actions.json
  - <image>.element_actions.png (labels: id — primary_action · description)

Run:
  RUN_REAL_MODELS=1 OMNIPARSER_WEIGHTS=... QWEN_MODEL='Qwen/Qwen2.5-VL-3B-Instruct' \
  pytest -q tests_slow/test_element_actions_vlm.py -s -o log_cli=true --log-cli-level=INFO
"""

import os
import json
import asyncio
import time
from pathlib import Path
from utils import vram_snapshot
import pytest
from PIL import Image, ImageDraw, ImageFont

from models import ProcessingConfig
from pipeline import ScreenUnderstandingPipeline
from processors import QwenVLMProcessor


RUN_REAL = os.environ.get("RUN_REAL_MODELS") == "1"


def _annotate_actions(image: Image.Image, elements, actions_by_id):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for el in elements:
        if not getattr(el, 'clickable', False):
            continue
        b = el.bbox
        x1, y1, x2, y2 = b.x, b.y, b.x + b.width, b.y + b.height
        color = (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        a = actions_by_id.get(el.id, {})
        label = el.id[8:]
        pa = a.get('primary_action')
        desc = a.get('description')
        if pa:
            label += f"-{pa}"
        if desc:
            # keep short on canvas
            s = desc
            label += f"·{s}"
        try:
            tw = draw.textlength(label, font=font)
            th = font.size if font else 12
        except Exception:
            tw, th = len(label) * 6, 12
        pad = 2
        bg = [x1, max(0, y1 - th - 2 * pad), x1 + int(tw) + 2 * pad, y1]
        draw.rectangle(bg, fill=color)
        draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill=(255, 255, 255), font=font)
    return img


@pytest.mark.skipif(not RUN_REAL, reason="Set RUN_REAL_MODELS=1 to run real-model slow test")
def test_element_actions_vlm(tmp_path):
    t_total_start = time.perf_counter()
    img_path = Path(os.environ.get('IMAGE_PATH') or 'examples/amazon_sample.png')
    assert img_path.exists(), f"Image not found: {img_path}"

    weights = os.environ.get('OMNIPARSER_WEIGHTS','weights/icon_detect/model.pt')
    assert weights and Path(weights).exists(), "OMNIPARSER_WEIGHTS must point to a valid .pt file"

    qwen_model = os.environ.get('QWEN_MODEL') or 'Qwen/Qwen2.5-VL-7B-Instruct'
    use_gpu = os.environ.get('REAL_USE_GPU', '1') in {'1','true','True'}
    base = vram_snapshot("baseline")
    cfg = ProcessingConfig(
        use_gpu=use_gpu,
        omniparser_weights_path=weights,
        omniparser_allow_download=False,
        element_detection_threshold=0.15,
        iou_threshold=0.2,
        vlm_model_name=qwen_model,
        vlm_elements_max=38,
        batch_size=1,
        debug_mode=True,
    )

    pipe = ScreenUnderstandingPipeline(cfg)

    image = Image.open(img_path).convert('RGB')
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

    # VLM action classification
    # Count clickable and expected analyze count (limited by max_k)
    clickable_count = sum(1 for e in merged if getattr(e, 'clickable', False))
    to_analyze = min(clickable_count, cfg.vlm_elements_max)
    print(f"Preparing VLM action classification: clickable={clickable_count}, max_k={cfg.vlm_elements_max}, will_analyze={to_analyze}")

    t_vlm_start = time.perf_counter()
    summary_actions = asyncio.run(pipe._run_vlm_generate_summary_actions(image))
    actions = asyncio.run(pipe._run_vlm_classify_element_actions(image, merged, ocr, summary_actions,max_k=cfg.vlm_elements_max))
    t_vlm = time.perf_counter() - t_vlm_start
    print(f"Actions returned: {len(actions)}")
    t_total = time.perf_counter() - t_total_start
    print(f"Timing: OCR={t_ocr:.3f}s, Detect={t_detect:.3f}s, Merge={t_merge:.3f}s, VLM={t_vlm:.3f}s, Total={t_total:.3f}s")
    after_vlm = vram_snapshot("after_vlm")
    gpt_inc_mb = after_vlm["allocated_mb"] - base["allocated_mb"]
    # Save JSON
    side_json = img_path.with_suffix('.element_actions.json')
    side_json.write_text(json.dumps(actions, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved actions JSON: {side_json}")

    # Build id->action map
    a_by_id = {a.get('element_id'): a for a in actions if a.get('element_id')}
    annotated = _annotate_actions(image, merged, a_by_id)
    side_png = img_path.with_suffix('.element_actions.png')
    annotated.save(side_png, format='PNG')
    print(f"Saved actions PNG: {side_png}")

    assert isinstance(actions, list)
