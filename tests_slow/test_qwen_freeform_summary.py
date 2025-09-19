"""Slow test: Qwen2.5-VL free-form page summary on a single image.

Opt-in via RUN_REAL_MODELS=1 to avoid accidental large downloads.

Env vars:
- IMAGE_PATH   : image path (default: examples/amazon_sample.png)
- QWEN_MODEL   : model id (default: Qwen/Qwen2.5-VL-7B-Instruct)
- REAL_USE_GPU : 1 to enable GPU

Run:
  RUN_REAL_MODELS=1 python -m pytest -q tests_slow/test_qwen_freeform_summary.py -s
"""

import os
import pytest
from pathlib import Path
from PIL import Image
from models import ProcessingConfig

try:
    import alternative_ocr as alt_ocr  # optional to enrich context
except Exception:
    alt_ocr = None


RUN_REAL = os.environ.get("RUN_REAL_MODELS") == "1"


@pytest.mark.skipif(not RUN_REAL, reason="Set RUN_REAL_MODELS=1 to run real-model slow test")
def test_qwen_freeform_summary_on_example():
    import torch
    import transformers as tfm

    # 1) Image
    img_path = Path(os.environ.get('IMAGE_PATH') or 'examples/amazon_sample.png')
    assert img_path.exists(), f"Image not found: {img_path} (set IMAGE_PATH to override)"
    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 2) Model
    model_id = os.environ.get('QWEN_MODEL') or 'Qwen/Qwen2.5-VL-7B-Instruct'
    use_gpu = os.environ.get('REAL_USE_GPU', '0') in {'1', 'true', 'True'}

    processor = tfm.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    Qwen25VL = getattr(tfm, 'Qwen2_5_VLForConditionalGeneration', None)
    if Qwen25VL is None:
        # fallback (older environments)
        model = tfm.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_gpu and torch.cuda.is_available() else torch.float32,
            device_map="cuda:0" if use_gpu and torch.cuda.is_available() else None
        ).eval()
    else:
        model = Qwen25VL.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_gpu and torch.cuda.is_available() else torch.float32,
            device_map="cuda:0" if use_gpu and torch.cuda.is_available() else None
        ).eval()

    # 2.5) Optional OCR context to boost coverage
    ocr_context = []
    if alt_ocr is not None:
        try:
            # Prefer easyocr if available for UI screenshots
            proc = alt_ocr.EasyOCRProcessor(ProcessingConfig(use_gpu=False, ocr_confidence_threshold=0.3))
            ocr_results = asyncio.run(proc.process(image))
            ocr_context = [r.text for r in ocr_results if r.text]
        except Exception:
            try:
                proc = alt_ocr.TesseractOCRProcessor(ProcessingConfig(use_gpu=False, ocr_confidence_threshold=0.3))
                ocr_results = asyncio.run(proc.process(image))
                ocr_context = [r.text for r in ocr_results if r.text]
            except Exception:
                ocr_context = []
    ocr_snippet = "\n".join(ocr_context[:100]) if ocr_context else ""

    # 3) Prompt: focus on summary + many high‑confidence actions (no entities)
    system_hint = (
        "You are an expert in webpage understanding. Analyze the website screenshot and return an English answer with EXACTLY two sections, in plain text (no code, no JSON, no markdown).\n\n"
        "1) Summary (3–5 sentences): Identify the site/page type (e.g., home/search/product list/detail) and describe key modules (top navigation: account, orders, cart, delivery location, search bar, banners/categories). Avoid counting elements.\n"
        "2) High‑Confidence Actions: Enumerate AS MANY actionable steps as you can that you are confident about (confidence ≥ 0.8). Use short verb phrases and include a short target label if visible (e.g., 'open cart – Cart', 'view orders – & Orders', 'search products – Search bar'). Prefer unique, meaningful actions. Aim for at least 20 items if visible; otherwise list all you can find. Use one item per line, with a hyphen.\n\n"
        + (f"Context OCR texts (subset):\n{ocr_snippet}\n\n" if ocr_snippet else "")
        + "Return only the two sections in English without any extra commentary."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_hint},
                {"type": "image", "image": image},
            ],
        }
        ]

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt")

    # move to model device
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            temperature=0.2,
            num_beams=1,
        )

    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print("\nQWEN FREEFORM SUMMARY:\n", text[:1200])

    # Save next to image
    out_path = img_path.with_suffix('.qwen_freeform.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved freeform summary to: {out_path}")
