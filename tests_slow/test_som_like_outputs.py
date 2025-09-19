"""Slow test: produce SOM-like outputs (annotated image, label coords, parsed content)

Opt-in via RUN_REAL_MODELS=1. This uses the in-repo SOM-style fused step
(USE_SOM_STEP123) to run YOLO+OCR de-dup and batch captions, then:
 - Builds an annotated image (base64) akin to dino_labled_img
 - Exports label_coordinates as {element_id: [x1,y1,x2,y2]}
 - Exports parsed_content_list combining OCR texts and element captions

Run:
  RUN_REAL_MODELS=1 pytest -q tests_slow/test_som_like_outputs.py -s \
      -o log_cli=true --log-cli-level=INFO
"""

import os
import io
import json
import base64
import asyncio
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from models import ProcessingConfig
from pipeline import ScreenUnderstandingPipeline


RUN_REAL = os.environ.get("RUN_REAL_MODELS") == "1"


def _annotate(image: Image.Image, elements) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    # Optional font; fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, el in enumerate(elements):
        b = el.bbox
        x1, y1, x2, y2 = b.x, b.y, b.x + b.width, b.y + b.height
        color = (255, 0, 0) if el.clickable else (0, 180, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = f"{idx}"
        tw, th = draw.textlength(tag, font=font), (font.size if font else 10)
        pad = 2
        draw.rectangle([x1, max(0, y1 - th - 2 * pad), x1 + int(tw) + 2 * pad, y1], fill=color)
        draw.text((x1 + pad, max(0, y1 - th - pad)), tag, fill=(255, 255, 255), font=font)
    return img


@pytest.mark.skipif(not RUN_REAL, reason="Set RUN_REAL_MODELS=1 to run real-model slow test")
def test_som_like_outputs(tmp_path, monkeypatch):
    # Force pipeline to use fused SOM step (Step 2+3 combined)
    monkeypatch.setenv("USE_SOM_STEP123", "1")

    img_path = Path(os.environ.get("IMAGE_PATH") or "examples/amazon_sample.png")
    assert img_path.exists(), f"Image not found: {img_path}"

    # OmniParser weights still needed for YOLO in SOM path
    weights = os.environ.get("OMNIPARSER_WEIGHTS")
    assert weights and Path(weights).exists(), "OMNIPARSER_WEIGHTS must point to a valid .pt file"

    cfg = ProcessingConfig(
        use_gpu=os.environ.get("REAL_USE_GPU", "1") in {"1", "true", "True"},
        omniparser_weights_path=weights,
        omniparser_allow_download=False,
        debug_mode=True,
    )

    pipe = ScreenUnderstandingPipeline(cfg)

    image = Image.open(img_path).convert("RGB")
    # Step 1: OCR
    ocr_results = asyncio.run(pipe._run_ocr(image))
    print(f"OCR regions: {len(ocr_results)}")

    # Fused Step 2+3
    assert hasattr(pipe.element_detector, "som_step123"), "som_step123 not available on detector"
    elements = asyncio.run(pipe.element_detector.som_step123(image, ocr_results))
    print(f"Fused elements: {len(elements)}")

    # Build SOM-like outputs
    # 1) Annotated image -> base64
    annotated = _annotate(image, elements)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    dino_labled_img = base64.b64encode(buf.getvalue()).decode("ascii")

    # 2) label_coordinates as {element_id: [x1,y1,x2,y2]}
    label_coordinates = {}
    for el in elements:
        b = el.bbox
        label_coordinates[el.id] = [int(b.x), int(b.y), int(b.x + b.width), int(b.y + b.height)]

    # 3) parsed_content_list: OCR + captions
    parsed_content_list = []
    for i, r in enumerate(ocr_results):
        if r.text:
            parsed_content_list.append(f"Text Box ID {i}: {r.text}")
    cap_index = 0
    for el in elements:
        desc = (el.attributes or {}).get("description")
        if desc:
            parsed_content_list.append(f"Icon Box ID {cap_index}: {desc}")
            cap_index += 1

    # Save outputs
    out = {
        "dino_labled_img": dino_labled_img,
        "label_coordinates": label_coordinates,
        "parsed_content_list": parsed_content_list,
        "elements": [e.to_dict() for e in elements],
    }
    out_json = tmp_path / "som_like_outputs.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved SOM-like outputs to: {out_json}")

    # Also save sidecar next to image for quick inspection
    side = img_path.with_suffix(".som_like.json")
    try:
        side.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Also saved sidecar: {side}")
    except Exception as e:
        print(f"Warning: failed to save sidecar: {e}")

    # Save annotated PNG next to the image
    png_side = img_path.with_suffix(".som_like.png")
    try:
        annotated.save(png_side, format="PNG")
        print(f"Saved annotated PNG: {png_side}")
    except Exception as e:
        print(f"Warning: failed to save annotated PNG: {e}")

    # Quick sanity assertions
    assert isinstance(dino_labled_img, str) and len(dino_labled_img) > 0
    assert isinstance(label_coordinates, dict)
    assert isinstance(parsed_content_list, list)
