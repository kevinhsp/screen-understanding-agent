"""
alternative_ocr.py - Optional OCR backends used by the pipeline when
3rd-party packages may or may not be installed. Provides a safe Hybrid
backend that gracefully falls back to a no-op implementation instead of
raising, so tests can run without heavyweight OCR installs.

Backends implemented:
  - HybridOCRProcessor: tries Tesseract -> EasyOCR -> TrOCR -> NoOp
  - TesseractOCRProcessor: uses pytesseract image_to_data for word boxes
  - EasyOCRProcessor: uses easyocr.Reader when available
  - TrOCRProcessor: skeleton using HuggingFace TrOCR (optional)
  - NoOpOCRProcessor: returns [] (safe fallback)
"""

from typing import List, Optional
import logging
import importlib

import numpy as np
from PIL import Image

from models import OCRResult, BoundingBox, ProcessingConfig
from processors import OCRProcessor


logger = logging.getLogger(__name__)


class NoOpOCRProcessor(OCRProcessor):
    """Fallback OCR that returns no text regions."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        logger.warning("Using NoOpOCRProcessor (no OCR packages available)")

    async def process(self, image: Image.Image) -> List[OCRResult]:
        return []


class TesseractOCRProcessor(OCRProcessor):
    """OCR using pytesseract, if installed.

    Produces word-level boxes via image_to_data so downstream merging
    logic has spatial information to work with.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        try:
            self.pytesseract = importlib.import_module("pytesseract")
        except Exception as e:
            raise ImportError(f"pytesseract not available: {e}")

    async def process(self, image: Image.Image) -> List[OCRResult]:
        img = image.convert("RGB")
        data = self.pytesseract.image_to_data(img, output_type=self.pytesseract.Output.DICT)

        results: List[OCRResult] = []
        n = len(data.get("text", []))
        # pytesseract conf is a string of numbers 0..100 for words; may be '-1'
        thr = float(self.config.ocr_confidence_threshold or 0.0) * 100.0
        for i in range(n):
            try:
                text = (data["text"][i] or "").strip()
                if not text:
                    continue
                conf_raw = data.get("conf", ["-1"])[i]
                conf = float(conf_raw) if conf_raw not in (None, "-1", "") else -1.0
                if conf >= thr:
                    x = int(data.get("left", [0])[i])
                    y = int(data.get("top", [0])[i])
                    w = int(data.get("width", [0])[i])
                    h = int(data.get("height", [0])[i])
                    if w <= 0 or h <= 0:
                        continue
                    results.append(
                        OCRResult(
                            text=text,
                            bbox=BoundingBox(x=x, y=y, width=w, height=h),
                            confidence=max(0.0, min(1.0, conf / 100.0)),
                            language=self.config.ocr_language,
                        )
                    )
            except Exception:
                # be lenient; skip malformed rows
                continue

        logger.info(f"Tesseract OCR found {len(results)} text regions")
        return results


class EasyOCRProcessor(OCRProcessor):
    """OCR using easyocr.Reader, if installed."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        try:
            easyocr_mod = importlib.import_module("easyocr")
        except Exception as e:
            raise ImportError(f"easyocr not available: {e}")
        langs = [config.ocr_language or "en"]
        # GPU if requested and available; easyocr checks CUDA under the hood
        self.reader = easyocr_mod.Reader(langs, gpu=bool(config.use_gpu))

    async def process(self, image: Image.Image) -> List[OCRResult]:
        img = np.array(image.convert("RGB"))
        # detail=1 -> [[box(points), text, conf], ...]
        result = self.reader.readtext(img, detail=1)
        out: List[OCRResult] = []
        thr = float(self.config.ocr_confidence_threshold or 0.0)
        for item in result:
            try:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                coords, text, conf = item[0], (item[1] or "").strip(), float(item[2] or 0.0)
                if not text or conf < thr:
                    continue
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                out.append(
                    OCRResult(
                        text=text,
                        bbox=BoundingBox(x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1)),
                        confidence=conf,
                        language=self.config.ocr_language,
                    )
                )
            except Exception:
                continue
        logger.info(f"EasyOCR found {len(out)} text regions")
        return out


class TrOCRProcessor(OCRProcessor):
    """OCR using HuggingFace TrOCR (transformers), if available.

    This processor returns a single full-image text line unless enhanced with
    detection; intended as a best-effort option if transformers are installed.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        try:
            transformers = importlib.import_module("transformers")
            self.AutoProcessor = getattr(transformers, "AutoProcessor")
            self.V2S = getattr(transformers, "AutoModelForVision2Seq")
        except Exception as e:
            raise ImportError(f"transformers (TrOCR) not available: {e}")

        model_name = getattr(config, "trocr_model_name", "microsoft/trocr-base-printed")
        self.processor = self.AutoProcessor.from_pretrained(model_name)
        self.model = self.V2S.from_pretrained(model_name)
        self.model.eval()

    async def process(self, image: Image.Image) -> List[OCRResult]:
        import torch

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            gen = self.model.generate(pixel_values)
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        if not text:
            return []
        # No localization; return a single box covering the image
        w, h = image.size
        return [
            OCRResult(
                text=text,
                bbox=BoundingBox(x=0, y=0, width=int(w), height=int(h)),
                confidence=1.0,
                language=self.config.ocr_language,
            )
        ]


class HybridOCRProcessor(OCRProcessor):
    """Try multiple OCR options in priority order, fallback to NoOp."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._impl: Optional[OCRProcessor] = None

        # Priority: Tesseract -> EasyOCR -> TrOCR -> NoOp
        for cls in (TesseractOCRProcessor, EasyOCRProcessor, TrOCRProcessor):
            try:
                self._impl = cls(config)
                logger.info(f"HybridOCR selected: {cls.__name__}")
                break
            except Exception as e:
                logger.warning(f"HybridOCR option {cls.__name__} unavailable: {e}")

        if self._impl is None:
            self._impl = NoOpOCRProcessor(config)

    async def process(self, image: Image.Image) -> List[OCRResult]:
        return await self._impl.process(image)
