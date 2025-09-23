"""
processors.py - OCR, OmniParser and VLM processing modules
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import torch
import os
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
# YOLO is lazily imported where needed to reduce import-time failures
from pathlib import Path
from models import (
    OCRResult, UIElement, BoundingBox, ElementRole, ElementState,
    Affordance, ExtractedEntity, ScreenType, ActionType, ProcessingConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ============= OCR Module =============

class OCRProcessor(ABC):
    """Base class for OCR processors"""

    @abstractmethod
    async def process(self, image: Image.Image) -> List[OCRResult]:
        pass


class PaddleOCRProcessor(OCRProcessor):
    """PaddleOCR v4/v5 processor with full GPU support"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        logger.info("Initializing PaddleOCR with GPU support...")

        # Lazy import to avoid global dependency conflicts
        try:
            import importlib
            paddle_mod = importlib.import_module('paddle')
            paddleocr_mod = importlib.import_module('paddleocr')
            PaddleOCR = getattr(paddleocr_mod, 'PaddleOCR')
        except Exception as e:
            raise ImportError(f"PaddleOCR not available: {e}")

        # Force GPU usage for RTX 5090
        device = 'gpu' if config.use_gpu and torch.cuda.is_available() else 'cpu'

        # Set PaddlePaddle to use GPU
        if device == 'gpu':
            paddle_mod.set_device('gpu:0')
            logger.info(f"PaddlePaddle using GPU: {paddle_mod.get_device()}")

        # Initialize PaddleOCR with GPU support
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=config.ocr_language,
            use_gpu=(device == 'gpu'),
            gpu_mem=8000,
            use_tensorrt=True,
            precision='fp16',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            rec_batch_num=6,
            max_text_length=25,
            use_space_char=True,
            show_log=config.debug_mode
        )

        if device == 'gpu':
            logger.info("PaddleOCR initialized with GPU acceleration (RTX 5090 optimized)")

    async def process(self, image: Image.Image) -> List[OCRResult]:
        """Execute OCR recognition with GPU acceleration"""
        logger.info("Running PaddleOCR with GPU acceleration...")

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR on GPU
        result = self.ocr.ocr(img_array, cls=True)

        ocr_results = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    # Extract coordinates and text
                    coords = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]

                    # Convert coordinates to bounding box
                    x_coords = [p[0] for p in coords]
                    y_coords = [p[1] for p in coords]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    if confidence >= self.config.ocr_confidence_threshold:
                        ocr_results.append(OCRResult(
                            text=text,
                            bbox=BoundingBox(
                                x=int(x_min),
                                y=int(y_min),
                                width=int(x_max - x_min),
                                height=int(y_max - y_min)
                            ),
                            confidence=confidence,
                            language=self.config.ocr_language
                        ))

        logger.info(f"OCR detected {len(ocr_results)} text regions")
        return ocr_results


# ============= OmniParser v2 Module =============

class OmniParserV2:
    """OmniParser v2 - UI element detection using YOLO and Florence-2"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing OmniParser v2 on {self.device}...")

        # Initialize YOLO for clickable element detection
        self._init_yolo()

        # Defer Florence-2 loading to avoid occupying GPU memory
        # when only YOLO detection is needed. It will be initialized
        # on-demand inside som_step123 if captioning is requested.
        self.florence_processor = None
        self.florence_model = None

        logger.info("OmniParser v2 initialized (Florence lazy-loaded)")

    def _ensure_florence(self):
        """Lazily initialize Florence model if not already loaded."""
        try:
            if self.florence_model is None or self.florence_processor is None:
                self._init_florence()
        except Exception as e:
            logger.error(f"Failed to lazy-load Florence-2: {e}")

    def _init_yolo(self):
        """Initialize YOLO model for UI element detection"""
        if self.config.omniparser_weights_path:
            p = Path(self.config.omniparser_weights_path)
            if p.exists():
                weights_path = p
        # Lazy import YOLO to avoid global dependency requirements
        from ultralytics import YOLO
        self.yolo_model = YOLO(str(weights_path))
        # Try to capture class name mapping from the model for better role mapping
        self._yolo_names = None
        try:
            names = getattr(self.yolo_model.model, 'names', None)
            if isinstance(names, dict):
                self._yolo_names = names
            elif isinstance(names, (list, tuple)):
                self._yolo_names = {i: n for i, n in enumerate(names)}
        except Exception:
            self._yolo_names = None

        if self.device == 'cuda':
            try:
                # Only valid for PyTorch models; exported formats may not support .to()
                self.yolo_model.to('cuda')
                logger.info("YOLO model loaded on RTX 5090")
            except Exception as e:
                logger.warning(f"YOLO .to('cuda') failed: {e}; will pass device during inference")

    async def som_step123(self, image: Image.Image, ocr_results: List[OCRResult]) -> List[UIElement]:
        """Unified Step1-3 using a SOM-like pipeline: YOLO detect + OCR boxes de-dup + batch caption.

        - Runs YOLO to get candidate UI boxes.
        - Removes boxes that strongly overlap OCR text regions.
        - Optionally captions remaining icon/image/button regions in batch via Florence (if available).
        - Returns a combined UIElement list (text + detected elements) with descriptions attached.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        W, H = image.size

        # Prepare OCR boxes (normalized xyxy)
        ocr_norm: List[List[float]] = []
        for r in ocr_results or []:
            x1 = float(r.bbox.x)
            y1 = float(r.bbox.y)
            x2 = float(r.bbox.x + r.bbox.width)
            y2 = float(r.bbox.y + r.bbox.height)
            ocr_norm.append([x1 / W, y1 / H, x2 / W, y2 / H])

        # YOLO inference
        import numpy as _np
        img_array = _np.array(image)
        infer_kwargs = {'conf': self.config.element_detection_threshold}
        if self.device == 'cuda':
            infer_kwargs['device'] = 0
        yolo_res = self.yolo_model(img_array, **infer_kwargs)

        det_items: List[Dict[str, Any]] = []
        for r in yolo_res:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                det_items.append({
                    'xyxy_pix': [x1, y1, x2, y2],
                    'xyxy': [x1 / W, y1 / H, x2 / W, y2 / H],
                    'class_id': cls,
                    'score': conf,
                })

        # Remove detections that collide with OCR text (IoU in normalized space)
        iou_th = getattr(self.config, 'iou_threshold', 0.7) or 0.7
        kept_items: List[Dict[str, Any]] = []
        for it in det_items:
            bx = it['xyxy']
            overlap = any(self._iou_norm(bx, ob) > iou_th for ob in ocr_norm)
            if not overlap:
                kept_items.append(it)

        # Prepare crops for captioning (ICON/IMAGE/BUTTON typically)
        crops: List[Image.Image] = []
        kept_with_crop_idx: List[int] = []
        margin_ratio = 0.15
        for idx, it in enumerate(kept_items):
            x1, y1, x2, y2 = it['xyxy_pix']
            w = max(1, int(x2 - x1))
            h = max(1, int(y2 - y1))
            if w < 10 or h < 10:
                continue
            m = int(max(w, h) * margin_ratio)
            cx1 = max(0, int(x1) - m)
            cy1 = max(0, int(y1) - m)
            cx2 = min(W, int(x2) + m)
            cy2 = min(H, int(y2) + m)
            crop = image.crop((cx1, cy1, cx2, cy2))
            if crop.mode != 'RGB':
                crop = crop.convert('RGB')
            crops.append(crop)
            kept_with_crop_idx.append(idx)

        captions: List[str] = [None] * len(kept_items)
        # Initialize Florence lazily only if we actually have crops to caption
        if crops and (self.florence_model is None or self.florence_processor is None):
            self._ensure_florence()
        if self.florence_model and crops:
            try:
                caps = await self._batch_florence_caption(crops)
                for j, ci in enumerate(kept_with_crop_idx):
                    captions[ci] = caps[j] if j < len(caps) else None
            except Exception as e:
                logger.debug(f"Batch Florence caption failed: {e}")

        # Build UIElements: text boxes first, then detections
        elements: List[UIElement] = []
        # Text elements from OCR
        for r in ocr_results or []:
            elements.append(UIElement(
                id='',
                role=ElementRole.TEXT,
                text=r.text,
                state=ElementState.ENABLED,
                bbox=r.bbox,
                confidence=r.confidence,
                clickable=False,
                attributes={'source': 'som_ocr'}
            ))

        # Detected elements with captions
        for it, desc in zip(kept_items, captions):
            x1, y1, x2, y2 = it['xyxy_pix']
            bbox = BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            role = self._map_yolo_class_to_role(int(it.get('class_id', 0)))
            label = None
            try:
                if getattr(self, '_yolo_names', None) is not None:
                    label = self._yolo_names.get(int(it.get('class_id', 0)))
            except Exception:
                label = None
            attrs: Dict[str, Any] = {'source': 'som_detect', 'yolo_class': int(it.get('class_id', 0)), 'yolo_label': label}
            if desc:
                attrs['description'] = desc
                attrs['description_source'] = 'florence_batch'
            elements.append(UIElement(
                id='',
                role=role,
                text='',
                state=ElementState.ENABLED,
                bbox=bbox,
                confidence=float(it.get('score', 0.0)),
                clickable=self._is_clickable_role(role),
                attributes=attrs
            ))

        # Heuristic: promote likely search input when overlapping a 'search'-like OCR text and aspect ratio is wide
        search_words = {'search', 'search amazon'}
        def has_search_text(b: BoundingBox) -> bool:
            for r in ocr_results or []:
                t = (r.text or '').strip().lower()
                if not t:
                    continue
                if any(sw in t for sw in search_words):
                    if self._calculate_iou(b, r.bbox) > 0.1:
                        return True
            return False
        for el in elements:
            if el.role in {ElementRole.CONTAINER, ElementRole.IMAGE, ElementRole.TEXT}:
                w, h = max(1, el.bbox.width), max(1, el.bbox.height)
                if w / h >= 3.5 and has_search_text(el.bbox):
                    el.role = ElementRole.INPUT
                    el.clickable = True
                    if isinstance(el.attributes, dict):
                        el.attributes['role_inferred'] = 'search_input'

        # Optional: synthesize likely search input if YOLO missed the outer box
        try:
            if os.environ.get('ENABLE_SYNTH_SEARCH', '1') in {'1','true','True'}:
                self._maybe_add_synthetic_search_input(elements, ocr_results, W, H)
        except Exception as e:
            logger.debug(f"Synthetic search input step skipped: {e}")

        # Stable reading order and assign IDs
        def _order_key(el: UIElement):
            return (el.bbox.y, el.bbox.x)
        elements.sort(key=_order_key)
        for i, el in enumerate(elements):
            el.id = f"element_{i}"

        logger.info(f"SOM-like fused elements: {len(elements)}")
        return elements

    def _iou_norm(self, a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
        bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    async def _batch_florence_caption(self, crops: List[Image.Image]) -> List[str]:
        """Batch caption a list of crops using Florence-2; returns list of strings (may contain '')."""
        if not self.florence_model or not self.florence_processor or not crops:
            return [''] * len(crops)
        # Choose a concise task
        task = os.environ.get('FLORENCE_TASK') or '<CAPTION>'
        # Build inputs
        inputs = self.florence_processor(
            text=[task] * len(crops),
            images=crops,
            return_tensors='pt'
        )
        try:
            if self.device == 'cuda':
                inputs = {k: (v.cuda() if hasattr(v, 'cuda') else v) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = self.florence_model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=1,
                    do_sample=False
                )
            decoded = self.florence_processor.batch_decode(out_ids, skip_special_tokens=True)
            # Simple cleanup
            import re as _re
            cleaned = []
            for t in decoded:
                s = str(t or '').strip()
                s = _re.sub(r"<[^>]+>", ' ', s)
                s = _re.sub(r"\s+", ' ', s).strip()
                cleaned.append(s)
            return cleaned
        except Exception as e:
            logger.debug(f"Florence batch decode failed: {e}")
            return [''] * len(crops)

    def _maybe_add_synthetic_search_input(self,
                                          elements: List[UIElement],
                                          ocr_results: List[OCRResult],
                                          W: int,
                                          H: int) -> None:
        # If an INPUT already overlaps a 'search' OCR, do nothing
        def has_input_overlapping(bbox: BoundingBox) -> bool:
            return any(el.role == ElementRole.INPUT and self._calculate_iou(el.bbox, bbox) > 0.15 for el in elements)

        # Find candidate OCRs containing search-like text
        cand_ocr: List[OCRResult] = []
        for r in ocr_results or []:
            t = (r.text or '').strip().lower()
            if not t:
                continue
            if any(kw in t for kw in ['search', 'search amazon', '搜索']):
                cand_ocr.append(r)

        for r in cand_ocr:
            if has_input_overlapping(r.bbox):
                continue
            # Propose a wide box centered on the OCR text, with min aspect ratio
            x, y, w, h = r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height
            if w <= 0 or h <= 0:
                continue
            # Target width: enlarge to at least 5x text width, cap within image bounds; or 50% of image width
            target_w = max(int(w * 5.0), int(W * 0.5))
            target_w = min(target_w, int(W * 0.9))
            # Target height: slightly taller than text
            target_h = max(int(h * 1.8), h + 6)
            # Ensure aspect ratio >= 3.5
            if target_w / max(1, target_h) < 3.5:
                target_w = int(3.5 * max(1, target_h))
                target_w = min(target_w, int(W * 0.95))
            cx = x + w // 2
            cy = y + h // 2
            nx1 = max(0, cx - target_w // 2)
            nx2 = min(W, nx1 + target_w)
            ny1 = max(0, y - int(0.4 * target_h))
            ny2 = min(H, ny1 + target_h)
            # Build bbox and validate size
            bb = BoundingBox(int(nx1), int(ny1), int(nx2 - nx1), int(ny2 - ny1))
            if bb.width < 30 or bb.height < 10:
                continue
            # Avoid heavy overlap with existing large clickable boxes (likely already good)
            conflict = any(self._calculate_iou(bb, el.bbox) > 0.5 and el.clickable for el in elements)
            if conflict:
                continue
            elements.append(UIElement(
                id='',
                role=ElementRole.INPUT,
                text='',
                state=ElementState.ENABLED,
                bbox=bb,
                confidence=0.4,
                clickable=True,
                attributes={'source': 'synth_search', 'role_inferred': 'search_input_from_text', 'based_on_ocr': r.text}
            ))

    def _init_florence(self):
        """Initialize Florence-2 model for region description"""
        model_name = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-base")

        try:
            logger.info(f"Loading Florence-2 model: {model_name}")
            self.florence_processor = AutoProcessor.from_pretrained(model_name,trust_remote_code=True)
            if self.device == 'cuda':
                self.florence_model = AutoModelForCausalLM.from_pretrained(
                    'models/omniparser/icon_caption_florence',
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # FP16 for RTX 5090
                ).to("cuda")

                logger.info("Florence-2 loaded on RTX 5090 with FP16 precision")
            else:
                self.florence_model = AutoModelForCausalLM.from_pretrained(
                    'models/omniparser/icon_caption_florence',
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )

        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            self.florence_processor = None
            self.florence_model = None

    async def detect_elements(self, image: Image.Image) -> List[UIElement]:
        """Detect UI elements using YOLO"""
        logger.info("Detecting UI elements with YOLO...")

        # Ensure 3-channel RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Run YOLO detection
        infer_kwargs = {
            'conf': self.config.element_detection_threshold
        }
        if self.device == 'cuda':
            # Some exported models don't support .to('cuda'); pass device at inference time
            infer_kwargs['device'] = 0
        results = self.yolo_model(img_array, **infer_kwargs)

        elements = []
        element_id = 0

        # Process YOLO results
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Get class and confidence
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    conf = float(box.conf[0])

                    # Map YOLO class to ElementRole
                    role = self._map_yolo_class_to_role(cls)

                    # Create UIElement
                    yolo_label = None
                    try:
                        if getattr(self, '_yolo_names', None) is not None:
                            yolo_label = self._yolo_names.get(cls)
                    except Exception:
                        yolo_label = None
                    elements.append(UIElement(
                        id=f"element_{element_id}",
                        role=role,
                        text="",  # Will be filled by OCR merge
                        state=ElementState.ENABLED,
                        bbox=BoundingBox(x1, y1, x2-x1, y2-y1),
                        confidence=conf,
                        clickable=self._is_clickable_role(role),
                        attributes={"yolo_class": cls, "yolo_label": yolo_label}
                    ))
                    element_id += 1

        logger.info(f"Detected {len(elements)} UI elements")
        return elements

    async def _add_icon_descriptions(self, image: Image.Image, elements: List[UIElement]) -> List[UIElement]:
        """Add descriptions to icon elements using Florence-2.

        Notes:
        - Expands crop region slightly to provide more context to Florence.
        - Records description_source as 'florence' or 'florence_raw' when model produced text,
          otherwise fallback_* when synthesized downstream.
        """
        logger.info("Adding icon descriptions with Florence-2...")

        for element in elements:
            if element.role in [ElementRole.ICON, ElementRole.IMAGE, ElementRole.BUTTON]:
                # Skip extremely small crops
                if element.bbox.width < 10 or element.bbox.height < 10:
                    continue

                # Expand the element crop by 15% margin to give context
                bbox = element.bbox
                W, H = image.size
                margin = int(max(bbox.width, bbox.height) * 0.15)
                x1 = max(0, bbox.x - margin)
                y1 = max(0, bbox.y - margin)
                x2 = min(W, bbox.x + bbox.width + margin)
                y2 = min(H, bbox.y + bbox.height + margin)
                cropped = image.crop((x1, y1, x2, y2))

                # Ensure RGB for HF processors
                if cropped.mode != 'RGB':
                    cropped = cropped.convert('RGB')

                # Get description from Florence-2
                desc_pack = await self._get_florence_description(cropped)
                if desc_pack and isinstance(desc_pack, tuple):
                    description, src = desc_pack
                else:
                    description, src = (None, None)

                if description:
                    element.attributes['description'] = description
                    element.attributes['description_source'] = src or 'florence'
                    if not element.text:
                        element.text = description
                        element.attributes['label_source'] = 'florence'

        return elements

    async def _get_florence_description(self, image: Image.Image) -> Optional[Tuple[str, str]]:
        """Get image description using Florence-2.

        Returns (description, source_tag) where source_tag is 'florence' for parsed
        post-processed text or 'florence_raw' when falling back to raw decode.
        """
        if not self.florence_model:
            return None

        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tasks = []
            # Allow override task via env; else try a cascade of tasks
            env_task = os.environ.get('FLORENCE_TASK')
            if env_task:
                tasks.append(env_task)
            tasks.extend(["<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>", "<CAPTION>"])

            for task_prompt in tasks:
                try:
                    inputs = self.florence_processor(
                        text=task_prompt,
                        images=image,
                        return_tensors="pt"
                    )
                    if self.device == 'cuda':
                        inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

                    with torch.no_grad():
                        generated_ids = self.florence_model.generate(
                            **inputs,
                            max_new_tokens=64,
                            num_beams=3,
                            do_sample=False
                        )

                    generated_text = self.florence_processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=False
                    )[0]

                    # First attempt: Florence post-processing
                    try:
                        parsed = self.florence_processor.post_process_generation(
                            generated_text,
                            task=task_prompt,
                            image_size=image.size
                        )
                        if isinstance(parsed, dict) and task_prompt in parsed and parsed[task_prompt]:
                            return (str(parsed[task_prompt]).strip(), 'florence')
                    except Exception:
                        pass

                    # Fallback: simple cleanup of raw decoded text
                    raw = str(generated_text or '').strip()
                    if raw:
                        # Remove obvious special tokens like <...>
                        import re as _re
                        cleaned = _re.sub(r"<[^>]+>", " ", raw)
                        cleaned = _re.sub(r"\s+", " ", cleaned).strip()
                        if cleaned:
                            return (cleaned, 'florence_raw')
                except Exception:
                    # Try next task
                    continue

            return None

        except Exception as e:
            logger.debug(f"Florence-2 description failed: {e}")
            return None

    def _map_yolo_class_to_role(self, cls: int) -> ElementRole:
        """Map YOLO class ID to ElementRole"""
        # Prefer string label mapping when available
        try:
            label = None
            if getattr(self, '_yolo_names', None) is not None:
                label = str(self._yolo_names.get(int(cls)) or '').lower()
            if label:
                if any(k in label for k in ['input', 'textbox', 'text_box', 'searchbar', 'search_bar', 'search']):
                    return ElementRole.INPUT
                if 'button' in label or 'btn' in label:
                    return ElementRole.BUTTON
                if 'link' in label:
                    return ElementRole.LINK
                if 'checkbox' in label or 'check' in label:
                    return ElementRole.CHECKBOX
                if 'radio' in label:
                    return ElementRole.RADIO
                if 'dropdown' in label or 'select' in label:
                    return ElementRole.DROPDOWN
                if 'tab' in label:
                    return ElementRole.TAB
                if 'menu' in label:
                    return ElementRole.MENU
                if 'slider' in label:
                    return ElementRole.SLIDER
                if 'toggle' in label or 'switch' in label:
                    return ElementRole.TOGGLE
                if 'icon' in label:
                    return ElementRole.ICON
                if 'image' in label or 'img' in label or 'logo' in label:
                    return ElementRole.IMAGE
                if 'text' in label or 'label' in label:
                    return ElementRole.TEXT
        except Exception:
            pass
        # Fallback id-based mapping (adjust per weights)
        mapping = {
            0: ElementRole.BUTTON,
            1: ElementRole.INPUT,
            2: ElementRole.CHECKBOX,
            3: ElementRole.RADIO,
            4: ElementRole.DROPDOWN,
            5: ElementRole.LINK,
            6: ElementRole.IMAGE,
            7: ElementRole.ICON,
            8: ElementRole.TEXT,
            9: ElementRole.TAB,
            10: ElementRole.MENU,
            11: ElementRole.SLIDER,
            12: ElementRole.TOGGLE,
        }
        return mapping.get(int(cls), ElementRole.CONTAINER)

    def _is_clickable_role(self, role: ElementRole) -> bool:
        """Check if a role is typically clickable"""
        clickable_roles = {
            ElementRole.BUTTON,
            ElementRole.LINK,
            ElementRole.CHECKBOX,
            ElementRole.RADIO,
            ElementRole.TAB,
            ElementRole.MENU,
            ElementRole.TOGGLE,
            ElementRole.ICON,
            ElementRole.DROPDOWN
        }
        return role in clickable_roles

    async def merge_with_ocr(self, elements: List[UIElement], ocr_results: List[OCRResult]) -> List[UIElement]:
        """Merge OCR results with detected elements"""
        logger.info("Merging OCR results with detected elements...")

        for element in elements:
            best_match = None
            best_iou = 0.0

            for ocr in ocr_results:
                iou = self._calculate_iou(element.bbox, ocr.bbox)
                if iou > best_iou and iou > self.config.iou_threshold:
                    best_iou = iou
                    best_match = ocr

            if best_match and not element.text:
                element.text = best_match.text
                element.confidence = (element.confidence + best_match.confidence) / 2
                element.attributes['label_source'] = 'iou'
                element.attributes['label_iou'] = round(best_iou, 3)

            # Proximity-based fallback if no IoU match
            if not element.text:
                # Expand element box slightly and try overlap
                margin = max(10, int(max(element.bbox.width, element.bbox.height) * 0.3))
                expanded = BoundingBox(
                    x=max(0, element.bbox.x - margin),
                    y=max(0, element.bbox.y - margin),
                    width=element.bbox.width + 2 * margin,
                    height=element.bbox.height + 2 * margin,
                )

                prox_match = None
                prox_dist = 1e9

                for ocr in ocr_results:
                    # If OCR overlaps expanded box, consider it
                    if self._calculate_iou(expanded, ocr.bbox) > 0.0:
                        # distance from OCR center to element box edges (0 if inside)
                        ocx, ocy = ocr.bbox.center()
                        ex1, ey1 = element.bbox.x, element.bbox.y
                        ex2, ey2 = element.bbox.x + element.bbox.width, element.bbox.y + element.bbox.height
                        dx = 0.0
                        if ocx < ex1:
                            dx = ex1 - ocx
                        elif ocx > ex2:
                            dx = ocx - ex2
                        dy = 0.0
                        if ocy < ey1:
                            dy = ey1 - ocy
                        elif ocy > ey2:
                            dy = ocy - ey2
                        d = dx * dx + dy * dy
                        if d < prox_dist:
                            prox_dist = d
                            prox_match = ocr

                if prox_match:
                    element.text = prox_match.text
                    element.confidence = (element.confidence + prox_match.confidence) / 2
                    element.attributes['label_source'] = 'proximity'
                    element.attributes['label_distance'] = float(prox_dist)

                # Collect nearby OCR texts for context (top-3 by distance)
                nearby = []
                for ocr in ocr_results:
                    if self._calculate_iou(expanded, ocr.bbox) > 0.0:
                        # Only consider texts OUTSIDE the element box for nearby_texts
                        if self._calculate_iou(element.bbox, ocr.bbox) > 0.0:
                            continue
                        ocx, ocy = ocr.bbox.center()
                        ex1, ey1 = element.bbox.x, element.bbox.y
                        ex2, ey2 = element.bbox.x + element.bbox.width, element.bbox.y + element.bbox.height
                        dx = 0.0
                        if ocx < ex1:
                            dx = ex1 - ocx
                        elif ocx > ex2:
                            dx = ocx - ex2
                        dy = 0.0
                        if ocy < ey1:
                            dy = ey1 - ocy
                        elif ocy > ey2:
                            dy = ocy - ey2
                        d = dx * dx + dy * dy
                        nearby.append((d, ocr))
                nearby.sort(key=lambda t: t[0])
                element.attributes['nearby_texts'] = [
                    {"text": o.text, "distance": float(d), "conf": float(o.confidence)}
                    for d, o in nearby[:3]
                ]

        # Add OCR results that don't match any element as TEXT elements
        for ocr in ocr_results:
            matched = False
            for element in elements:
                if self._calculate_iou(element.bbox, ocr.bbox) > self.config.iou_threshold:
                    matched = True
                    break

            if not matched:
                elements.append(UIElement(
                    id=f"text_{len(elements)}",
                    role=ElementRole.TEXT,
                    text=ocr.text,
                    state=ElementState.ENABLED,
                    bbox=ocr.bbox,
                    confidence=ocr.confidence,
                    clickable=False
                ))

        # Optional fallback (disabled by default). Enable via env ENABLE_ICON_DESCRIPTION_FALLBACK=1
        if os.environ.get('ENABLE_ICON_DESCRIPTION_FALLBACK', '0') in {'1', 'true', 'True'}:
            try:
                for element in elements:
                    if element.role in [ElementRole.ICON, ElementRole.IMAGE, ElementRole.BUTTON]:
                        if not element.attributes.get('description'):
                            label = (element.text or '').strip()
                            if label:
                                action = 'Click'
                                if element.role == ElementRole.IMAGE:
                                    action = 'Open'
                                elif element.role == ElementRole.ICON:
                                    action = 'Activate'
                                element.attributes['description'] = f"{action} '{label}'"
                                element.attributes['description_source'] = 'fallback'
                            else:
                                # Use nearby OCR context if available
                                ctx = element.attributes.get('nearby_texts') or []
                                if ctx:
                                    element.attributes['description'] = f"{element.role.value.title()} near '{ctx[0]['text']}'"
                                    element.attributes['description_source'] = 'fallback_nearby'
                                else:
                                    element.attributes['description'] = element.role.value
                                    element.attributes['description_source'] = 'fallback_generic'
            except Exception:
                pass

        return elements

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


# ============= VLM Module =============

class VLMProcessor(ABC):
    """Base class for Vision Language Model processors"""

    @abstractmethod
    async def analyze(self,
                     image: Image.Image,
                     texts: List[str],
                     elements: List[UIElement]) -> Dict[str, Any]:
        pass


class ScreenAIProcessor(VLMProcessor):
    """Google's ScreenAI processor using Hugging Face"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing ScreenAI with model: {config.vlm_model_name}")

        # Load ScreenAI or alternative VLM model
        try:
            # Note: Using Florence-2 as ScreenAI alternative since it's available
            model_name = config.vlm_model_name or "microsoft/Florence-2-large"
            if 'qwen' in (model_name or '').lower():
                raise ValueError("Qwen model detected. Use QwenVLMProcessor for Qwen/VL models.")

            if self.device == 'cuda':
                # Optimized for RTX 5090
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # FP16 for RTX 5090
                    device_map="cuda:0"
                ).eval()

                logger.info(f"VLM model loaded on RTX 5090 with FP16 precision")
            else:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                ).eval()

                logger.info(f"VLM model loaded on CPU")

        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise

    async def analyze(self,
                     image: Image.Image,
                     texts: List[str],
                     elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze screen content using VLM"""
        logger.info("Analyzing screen with VLM...")

        # Perform multiple analysis tasks
        screen_type = await self._analyze_screen_type(image)
        summary = await self._generate_summary(image, texts, elements)
        affordances = await self._detect_affordances(image, elements)
        entities = self._extract_entities_from_elements(texts, elements)

        analysis = {
            "screen_type": screen_type,
            "summary": summary,
            "affordances": affordances,
            "entities": entities,
            "confidence": 0.85,
            "warnings": []
        }

        return analysis

    async def _analyze_screen_type(self, image: Image.Image) -> ScreenType:
        """Analyze screen type using VLM"""
        task = "<OD>"  # Object detection task

        inputs = self.processor(
            text=task,
            images=image,
            return_tensors="pt"
        )

        if self.device == 'cuda':
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3
            )

        result = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(result, task=task, image_size=image.size)

        # Infer screen type from detected objects
        if task in parsed:
            objects = parsed[task].get('labels', [])
            return self._infer_screen_type_from_objects(objects)

        return ScreenType.UNKNOWN

    async def _generate_summary(self, image: Image.Image, texts: List[str], elements: List[UIElement]) -> str:
        """Generate screen summary using VLM"""
        task = "<DETAILED_CAPTION>"

        inputs = self.processor(
            text=task,
            images=image,
            return_tensors="pt"
        )

        if self.device == 'cuda':
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=3
            )

        result = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(result, task=task, image_size=image.size)

        if task in parsed:
            return parsed[task]

        # Fallback summary
        interactive_count = sum(1 for e in elements if e.is_interactive())
        return f"Screen with {len(elements)} elements ({interactive_count} interactive)"

    async def _detect_affordances(self, image: Image.Image, elements: List[UIElement]) -> List[Affordance]:
        """Detect possible actions on the screen"""
        affordances = []

        # Create affordances based on detected elements
        for element in elements:
            if not element.is_interactive():
                continue

            if element.role == ElementRole.BUTTON:
                affordances.append(Affordance(
                    action_type=ActionType.CLICK,
                    target_element=element.id,
                    description=f"Click {element.text or 'button'}",
                    priority=8 if element.text else 5
                ))
            elif element.role == ElementRole.INPUT:
                affordances.append(Affordance(
                    action_type=ActionType.TYPE,
                    target_element=element.id,
                    description=f"Type in {element.text or 'input field'}",
                    priority=7
                ))
            elif element.role == ElementRole.LINK:
                affordances.append(Affordance(
                    action_type=ActionType.CLICK,
                    target_element=element.id,
                    description=f"Navigate to {element.text or 'link'}",
                    priority=6
                ))
            elif element.role in [ElementRole.CHECKBOX, ElementRole.RADIO]:
                affordances.append(Affordance(
                    action_type=ActionType.CLICK,
                    target_element=element.id,
                    description=f"Toggle {element.text or 'option'}",
                    priority=5
                ))
            elif element.role == ElementRole.DROPDOWN:
                affordances.append(Affordance(
                    action_type=ActionType.SELECT,
                    target_element=element.id,
                    description=f"Select from {element.text or 'dropdown'}",
                    priority=6
                ))

        # Sort by priority
        affordances.sort(key=lambda x: x.priority, reverse=True)

        return affordances[:10]

    def _infer_screen_type_from_objects(self, objects: List[str]) -> ScreenType:
        """Infer screen type from detected objects"""
        objects_lower = [obj.lower() for obj in objects]

        if any(word in objects_lower for word in ['login', 'password', 'signin', 'username']):
            return ScreenType.LOGIN
        elif any(word in objects_lower for word in ['dashboard', 'chart', 'graph', 'metric']):
            return ScreenType.DASHBOARD
        elif any(word in objects_lower for word in ['form', 'input', 'textbox', 'field']):
            return ScreenType.FORM
        elif any(word in objects_lower for word in ['list', 'table', 'row', 'column']):
            return ScreenType.LIST
        elif any(word in objects_lower for word in ['settings', 'preferences', 'config']):
            return ScreenType.SETTINGS
        elif any(word in objects_lower for word in ['search', 'query', 'find']):
            return ScreenType.SEARCH
        elif any(word in objects_lower for word in ['cart', 'checkout', 'payment']):
            return ScreenType.CHECKOUT
        elif any(word in objects_lower for word in ['chat', 'message', 'conversation']):
            return ScreenType.CHAT
        elif any(word in objects_lower for word in ['error', 'warning', 'alert']):
            return ScreenType.ERROR

        return ScreenType.UNKNOWN

    def _extract_entities_from_elements(self, texts: List[str], elements: List[UIElement]) -> List[ExtractedEntity]:
        """Extract entities from text and elements"""
        entities = []

        all_text = ' '.join(texts) + ' '.join([e.text for e in elements if e.text])

        # Extract emails
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', all_text)
        for email in emails:
            entities.append(ExtractedEntity(
                entity_type="email",
                value=email,
                confidence=0.95,
                source='regex'
            ))

        # Extract prices
        prices = re.findall(r'[$\u20ac\u00a3\u00a5]\s*\d+(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?\s*(?:USD|EUR|GBP)', all_text)
        for price in prices:
            entities.append(ExtractedEntity(
                entity_type="price",
                value=price,
                confidence=0.9,
                source='regex'
            ))

        # Extract dates
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}', all_text)
        for date in dates:
            entities.append(ExtractedEntity(
                entity_type="date",
                value=date,
                confidence=0.85,
                source='regex'
            ))

        # Extract phone numbers
        phones = re.findall(r'(?:\+\d{1,3}\s?)?(?:\(\d{1,4}\)\s?)?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}', all_text)
        for phone in phones:
            if len(phone) >= 10:
                entities.append(ExtractedEntity(
                    entity_type="phone",
                    value=phone,
                    confidence=0.8,
                    source='regex'
                ))

        return entities


class QwenVLMProcessor(VLMProcessor):
    """Qwen2.5-VL adapter for screen analysis.

    Uses chat template with one image and structured prompt to obtain
    screen_type, summary, and optionally actions/entities in JSON.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing Qwen VLM with model: {config.vlm_model_name}")

        try:
            import transformers as _tfm
            model_name = self._sanitize_model_id(config.vlm_model_name or '')
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            # Prefer the dedicated Qwen2.5-VL class when available
            Qwen25VL = getattr(_tfm, 'Qwen2_5_VLForConditionalGeneration', None)
        
            self.model = Qwen25VL.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="cuda:0" if self.device == 'cuda' else None
            ).eval()
        except Exception as e:
            logger.error(f"Failed to load Qwen VLM: {e}")
            raise

    # Context manager helpers so callers can ensure cleanup
    def __enter__(self): 
        return self

    def __exit__(self):
        try:
            self.unload()
        except Exception:
            pass

    def unload(self) -> None:
        """Release model/processors and free GPU memory.

        Moves model to CPU (if possible), deletes references, and triggers
        CUDA cache cleanup. This helps when you only need a one-off call
        like generate_summary_actions and want the GPU fully freed after.
        """
        try:
            # Move model off GPU first to shrink GPU cache pressure
            if getattr(self, 'model', None) is not None:
                try:
                    self.model.to('cpu')
                except Exception:
                    pass
        except Exception:
            pass
        # Drop references
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.processor
        except Exception:
            pass
        # Best-effort CUDA/cache cleanup
        try:
            import torch, gc
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    # Collect IPC memory from forks if any
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            gc.collect()
        except Exception:
            pass

    @staticmethod
    def _sanitize_model_id(raw: str) -> str:
        """Extract a valid HF repo id from a possibly noisy string.

        Keeps the first match of `<org>/<name>` where org/name are in
        `[A-Za-z0-9._-]` and do not start/end with '.' or '-'.
        """
        raw = (raw or '').strip()
        # quick accept if it looks clean
        import re as _re
        pattern = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
        m = _re.search(pattern, raw)
        cleaned = m.group(0) if m else raw.split()[0] if raw else raw
        if cleaned != raw:
            logger.warning(f"Sanitized VLM model id from '{raw}' to '{cleaned}'")
        return cleaned

    async def analyze(self,
                      image: Image.Image,
                      texts: List[str],
                      elements: List[UIElement]) -> Dict[str, Any]:
        # Trim and prioritize element context
        elements_for_prompt = self._select_elements_for_prompt(elements)
        prompt = self._build_prompt(texts, elements_for_prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[image], return_tensors="pt")

        # Move tensors to model device if needed
        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = torch.device('cuda' if self.device == 'cuda' else 'cpu')
        inputs = {k: (v.to(model_device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(384, self.config.vlm_max_tokens or 384),
                do_sample=False,  # enforce deterministic JSON
                temperature=max(0.1, (self.config.vlm_temperature or 0.1)),
                num_beams=1
            )

        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        parsed = self._parse_response(text)

        # Heuristics + rule-based fallback when JSON is missing fields
        if not parsed.get('screen_type'):
            parsed['screen_type'] = self._infer_screen_type(texts, elements)
        if not parsed.get('summary'):
            parsed['summary'] = self._fallback_summary(elements)

        # Affordances from VLM JSON, then add rule-based if empty
        affordances: List[Affordance] = []
        for a in parsed.get('affordances', []) or []:
            try:
                affordances.append(Affordance(
                    action_type=ActionType[a.get('action_type','CLICK').upper()] if isinstance(a.get('action_type'), str) else ActionType.CLICK,
                    target_element=a.get('target_element'),
                    description=a.get('description',''),
                    priority=int(a.get('priority', 5))
                ))
            except Exception:
                continue
        if not affordances:
            affordances = self._rule_affordances(elements)

        # Entities from VLM JSON + regex fallback
        entities: List[ExtractedEntity] = []
        for e in parsed.get('entities', []) or []:
            try:
                entities.append(ExtractedEntity(
                    entity_type=str(e.get('entity_type','unknown')),
                    value=str(e.get('value','')),
                    confidence=float(e.get('confidence', 0.8)),
                    source='vlm'
                ))
            except Exception:
                continue
        if not entities:
            entities = self._regex_entities(texts, elements)

        return {
            'screen_type': parsed.get('screen_type', ScreenType.UNKNOWN),
            'summary': parsed.get('summary', ''),
            'affordances': affordances,
            'entities': entities,
            'confidence': float(parsed.get('confidence', 0.85)),
            'warnings': parsed.get('warnings', [])
        }

    async def classify_element_actions(self,
                                       image: Image.Image,
                                       elements: List[UIElement],
                                       ocr_results: List[OCRResult],
                                       summary_actions:Dict[str, Any] = {},
                                       max_k: int = 52) -> List[Dict[str, Any]]:
        """Classify primary actions per element using the cropped region image.

        For each candidate element, crop a tight region around its bounding box
        (with small padding) and ask the VLM to classify the primary action.

        Returns a list of objects: {
          element_id, primary_action, description, secondary_actions, confidence
        }
        """
        def area(el: UIElement) -> int:
            return max(1, el.bbox.width) * max(1, el.bbox.height)

        # Only analyze clickable elements (as requested)
        clickable = [e for e in elements if getattr(e, 'clickable', False)]
        clickable.sort(key=lambda e: -area(e))
        cands = clickable[: max_k]

        if not cands:
            return []

        W, H = image.size

        schema = (
            "Return ONLY a JSON object with keys: "
            "element_id, primary_action(one of: click,type,toggle,open,navigate,select,none), "
            "description(<=15 words), secondary_actions(array of strings), confidence(0..1)."
        )

        # Strong rules to bias arrival/departure classification using nearest outside labels
        rules = (
            "Strong Labeling Rules (follow strictly; override all priors):\n"
            "- If the nearest outside label contains any of [to, destination, arrive, arrival, inbound, 到, 目的地, 到达], classify this element as arrival.\n"
            "- If it contains any of [from, origin, depart, departure, outbound, 出发, 起点, 出发地], classify this element as departure.\n"
            "- If both appear, choose the one closer to the element box edge; if tie, prefer explicit forms (arrival/departure > to/from). If still ambiguous, set role='unknown'.\n"
            "- Do not infer arrival/departure from airport codes or logos. Ignore global priors if they conflict with the nearest label.\n"
            "- Text inside the element box (placeholder/value) is not a label; use nearest outside texts only.\n\n"
        )

        results: List[Dict[str, Any]] = []

        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = torch.device('cuda' if self.device == 'cuda' else 'cpu')

        def _progress(msg: str):
            if getattr(self.config, 'debug_mode', False):
                try:
                    print(msg)
                except Exception:
                    pass
                try:
                    logger.info(msg)
                except Exception:
                    pass

        total = len(cands)
        _progress(f"[VLM actions] analyzing {total} clickable elements (max_k={max_k})")
        last = len(cands) - 1
        for idx, el in enumerate(cands, start=1):
            b = el.bbox
            pad_x = max(4, int(0.1 * b.width))
            pad_y = max(4, int(0.1 * b.height))
            x1 = max(0, b.x - pad_x)
            y1 = max(0, b.y - pad_y)
            x2 = min(W, b.x + b.width + pad_x)
            y2 = min(H, b.y + b.height + pad_y)
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = b.x, b.y, b.x + max(1, b.width), b.y + max(1, b.height)
            crop = image.crop((x1, y1, x2, y2))
            role = getattr(el.role, 'value', str(el.role))
            text_val = (el.text or '').strip()
            line = (
                f"{el.id}|role={role}|text={text_val[:60]}"
            )

            # Provide nearby OCR texts to help disambiguate label association
            nearby_top = ""
            try:
                nb = None
                if isinstance(getattr(el, 'attributes', None), dict):
                    nb = el.attributes.get('nearby_texts')
                if isinstance(nb, list) and nb:
                    texts = []
                    for item in nb[:5]:
                        t = str((item or {}).get('text') or '').strip()
                        if t:
                            texts.append(t)
                    if texts:
                        joined = " | ".join(texts)
                        line = line + f"\nNearby texts (closest first): {joined}"
                        nearby_top = f"Nearest outside texts: {joined}\n\n"
                        # Debug log to confirm nearby texts are included in the VLM context
                        if getattr(self.config, 'debug_mode', False):
                            try:
                                print(f"[VLM actions] {el.id} nearby_texts: {texts}")
                                logger.info(f"[VLM actions] {el.id} nearby_texts: {texts}")
                                print(f"[VLM actions] element line: {line}")
                                logger.info(f"[VLM actions] element line: {line}")
                            except Exception:
                                pass
            except Exception:
                pass

            prompt = (
                "You are an expert in webpage understanding. Given a webpage summary, high confidence actions and cropped region image of a clickable part of a webpage, decide what the user can do with this part. Describe as simple and clear as you can. Use English only and reference element id exactly.\n\n"
                f"{nearby_top}"
                f"Webpage summary:\n{summary_actions.get('summary','')}\n\n"
                f"High‑Confidence Actions:\n{str(summary_actions.get('hc_actions',''))}\n\n"
                f"Output schema: {schema}\n\nElement:\n{line}"
            )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": crop},
                ],
            }]

            chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[chat], images=[crop], return_tensors="pt")
            inputs = {k: (v.to(model_device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(192, self.config.vlm_max_tokens or 192),
                    do_sample=False,
                    temperature=max(0.1, (self.config.vlm_temperature or 0.1)),
                    num_beams=1,
                )
            text_out = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            import json as _json
            parsed: Dict[str, Any] = {}
            try:
                s = text_out.strip()
                if s.startswith('[') and s.endswith(']'):
                    arr = _json.loads(s)
                    if isinstance(arr, list) and arr:
                        parsed = arr[0] if isinstance(arr[0], dict) else {}
                else:
                    i = s.find('{'); j = s.rfind('}')
                    payload = s[i:j+1] if (i != -1 and j != -1 and j > i) else s
                    obj = _json.loads(payload)
                    if isinstance(obj, dict):
                        parsed = obj
            except Exception:
                parsed = {}

            allowed = {"click","type","toggle","open","navigate","select","none"}
            eid = str(parsed.get('element_id') or el.id)
            pa = str(parsed.get('primary_action','none')).lower().strip()
            if pa not in allowed:
                pa = 'none'
            desc = str(parsed.get('description','')).strip()
            sec = parsed.get('secondary_actions') or []
            if not isinstance(sec, list):
                sec = []
            try:
                conf = float(parsed.get('confidence', 0.7))
            except Exception:
                conf = 0.7
            _progress(f"[VLM actions] {idx}/{total} el={el.id} action = {pa} description={desc} ")
            results.append({
                'element_id': eid,
                'primary_action': pa,
                'description': desc,
                'secondary_actions': sec,
                'confidence': conf,
            })
            # if(idx == last):
            #     for k in ("input_ids","pixel_values","attention_mask","labels"):
            #         if k in inputs:
            #             del inputs[k]
            #     del inputs
            #     del out_ids

        return results

    async def generate_summary_actions(self, image: Image.Image) -> Dict[str, Any]:
        """Generate English output: Summary + High‑Confidence Actions.

        This is intended for human inspection (not structured JSON). It asks
        the model to produce two sections in plain text.
        """
        # Prepare OCR snippet (limit size) if enabled
        # texts_snippet = "\n".join([t for t in texts if t][:100]) if getattr(self.config, 'vlm_freeform_include_ocr', False) else ""
        results: Dict[str, Any] = {}
        schema = (
            "Return ONLY a JSON object with keys: "
            "summary, high_confidence_actions"
        )
        system_hint = (
            "You are an expert in webpage understanding. Analyze the website screenshot and return an English answer with EXACTLY two sections, in plain text (no code, no JSON, no markdown).\n\n"
            "1) Summary (3–5 sentences): Identify the site/page type (e.g., home/search/product list/detail) and describe key modules (top navigation: account, orders, cart, delivery location, search bar, banners/categories). Avoid counting elements.\n"
            "2) High‑Confidence Actions: Enumerate AS MANY actionable steps as you can that you are confident about (confidence ≥ 0.8). Use short verb phrases and include a short target label if visible (e.g., 'open cart – Cart', 'view orders – & Orders', 'search products – Search bar'). Prefer unique, meaningful actions. Aim for at least 20 items if visible; otherwise list all you can find. Use one item per line, with a hyphen.\n\n"
            f"Output schema: {schema}\n"
        )

        # Optionally inject OCR context to enrich coverage (disabled by default)
        # if texts_snippet:
        #     system_hint += f"Context OCR texts (subset):\n{texts_snippet}\n\n"
        # system_hint += "Return only the two sections in English without any extra commentary."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_hint},
                    {"type": "image", "image": image},
                ],
            }
        ]
        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = torch.device('cuda' if self.device == 'cuda' else 'cpu')

        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[image], return_tensors="pt")
        inputs = {k: (v.to(model_device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(768, int(self.config.vlm_max_tokens or 768)),
                do_sample=False,
                temperature=0.2,
                num_beams=1,
            )

        text_out = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        import json as _json
        parsed: Dict[str, Any] = {}
        try:
            s = text_out.strip()
            if s.startswith('[') and s.endswith(']'):
                arr = _json.loads(s)
                if isinstance(arr, list) and arr:
                    parsed = arr[0] if isinstance(arr[0], dict) else {}
            else:
                i = s.find('{'); j = s.rfind('}')
                payload = s[i:j+1] if (i != -1 and j != -1 and j > i) else s
                obj = _json.loads(payload)
                if isinstance(obj, dict):
                    parsed = obj
        except Exception:
            parsed = {}
        summary = str(parsed.get('summary','')).strip()
        hc_actions = parsed.get('high_confidence_actions')
        import copy
        results["hc_actions"] = copy.deepcopy(hc_actions) if isinstance(hc_actions, list) else []
        results["summary"] = summary
        return results

    def _build_prompt(self, texts: List[str], elements: List[UIElement]) -> str:
        # Build concise context from OCR and selected elements
        texts_str = '\n'.join([t for t in texts if t][:40])
        # Group key actions if possible
        header = []
        for el in elements:
            label = (el.text or el.attributes.get('description') or '').strip()
            if el.role in [ElementRole.INPUT] and label:
                header.append(f"Search: {label}")
            if el.role in [ElementRole.BUTTON, ElementRole.ICON, ElementRole.LINK] and label:
                if any(k in label.lower() for k in ['account', 'login', 'sign in']):
                    header.append(f"Account: {label}")
                if any(k in label.lower() for k in ['orders']):
                    header.append(f"Orders: {label}")
                if any(k in label.lower() for k in ['cart']):
                    header.append(f"Cart: {label}")
        header_str = '\n'.join(header[:6])

        elem_lines = []
        for el in elements:
            role = el.role.value if hasattr(el.role, 'value') else str(el.role)
            label = (el.text or el.attributes.get('description') or '').strip()
            if not label:
                continue
            elem_lines.append(f"- {role}: {label}")
        elems_str = '\n'.join(elem_lines[:40])

        # JSON schema example for stronger adherence
        example = (
            '{\n'
            '  "screen_type": "list",\n'
            '  "summary": "E-commerce homepage with search, account, orders and cart.",\n'
            '  "affordances": [\n'
            '    {"action_type":"click","target_element":"element_orders","description":"View orders","priority":8},\n'
            '    {"action_type":"type","target_element":"element_search","description":"Search products","priority":9}\n'
            '  ],\n'
            '  "entities": [\n'
            '    {"entity_type":"zipcode","value":"02142","confidence":0.9}\n'
            '  ]\n'
            '}'
        )

        preface = (
            "You are an assistant for UI screen understanding. Respond in English. "
            "Output exactly ONE JSON object with keys: screen_type (one of login/dashboard/form/list/settings/search/checkout/chat/error/unknown), "
            "summary (1-2 sentences), affordances (list of {action_type, target_element, description, priority 0-10}), "
            "entities (list of {entity_type, value, confidence 0-1}). No extra text.\n\n"
        )
        example_block = f"Example JSON (follow this structure, replace with this screen content):\n{example}\n\n" if self.config.vlm_prompt_example else ""
        tail = "Return only the JSON object wrapped in <json> and </json> tags." if self.config.vlm_json_tag else "Return only the JSON object."

        return (
            preface +
            f"Context header (if present):\n{header_str}\n\n" +
            f"OCR texts (subset):\n{texts_str}\n\n" +
            f"Elements (subset):\n{elems_str}\n\n" +
            example_block +
            tail
        )

    def _select_elements_for_prompt(self, elements: List[UIElement]) -> List[UIElement]:
        if not elements:
            return []
        # Prefer clickable elements with labels
        labeled = [
            e for e in elements
            if e.clickable and ((e.text and e.text.strip()) or (e.attributes.get('description') or '').strip())
        ]
        # Add search inputs even if not clickable
        inputs = [e for e in elements if e.role == ElementRole.INPUT and (e.text or e.attributes.get('description'))]
        pool = labeled + [e for e in inputs if e not in labeled]
        # Sort by confidence desc
        pool.sort(key=lambda e: float(e.confidence or 0.0), reverse=True)
        # Apply min label length and clickable-only if requested
        min_len = max(0, int(self.config.vlm_min_label_len or 0))
        if min_len > 0:
            pool = [e for e in pool if len((e.text or e.attributes.get('description') or '').strip()) >= min_len]
        if getattr(self.config, 'vlm_clickable_only', False):
            pool = [e for e in pool if e.clickable]
        limit = int(self.config.vlm_elements_max or 52)
        return pool[:limit]

    def _regex_entities(self, texts: List[str], elements: List[UIElement]) -> List[ExtractedEntity]:
        # Reuse simple regex extraction similar to ScreenAIProcessor
        all_text = ' '.join(texts) + ' ' + ' '.join([(e.text or '') for e in elements if e.text])
        entities: List[ExtractedEntity] = []
        # zipcode / location-like (very rough)
        for m in re.findall(r'\b\d{5}(?:-\d{4})?\b', all_text):
            entities.append(ExtractedEntity(entity_type='zipcode', value=m, confidence=0.8, source='regex'))
        # price
        for m in re.findall(r'[$\u20ac\u00a3\u00a5]\s*\d+(?:[.,]\d{2})?', all_text):
            entities.append(ExtractedEntity(entity_type='price', value=m, confidence=0.85, source='regex'))
        # email
        for m in re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', all_text):
            entities.append(ExtractedEntity(entity_type='email', value=m, confidence=0.9, source='regex'))
        return entities

    def _rule_affordances(self, elements: List[UIElement]) -> List[Affordance]:
        aff: List[Affordance] = []
        for el in elements:
            if not el.clickable:
                continue
            label = (el.text or el.attributes.get('description') or '').strip() or el.role.value
            if el.role == ElementRole.BUTTON or el.role == ElementRole.LINK or el.role == ElementRole.ICON:
                pr = 8 if label else 5
                aff.append(Affordance(action_type=ActionType.CLICK, target_element=el.id, description=f"Click {label}", priority=pr))
            elif el.role == ElementRole.INPUT:
                aff.append(Affordance(action_type=ActionType.TYPE, target_element=el.id, description=f"Type in {label}", priority=7))
            elif el.role == ElementRole.DROPDOWN:
                aff.append(Affordance(action_type=ActionType.SELECT, target_element=el.id, description=f"Select from {label}", priority=6))
        # De-dup by target/description
        seen = set()
        uniq: List[Affordance] = []
        for a in aff:
            key = (a.action_type.value, a.target_element or '', a.description.strip())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(a)
        return uniq[:10]

    def _parse_response(self, text: str) -> Dict[str, Any]:
        # Try to extract JSON from free-form text
        try:
            return json.loads(text)
        except Exception:
            pass
        # If model wraps JSON in <json>...</json>
        if getattr(self.config, 'vlm_json_tag', False) and '<json>' in text and '</json>' in text:
            try:
                b = text.index('<json>') + len('<json>')
                e = text.rindex('</json>')
                return json.loads(text[b:e].strip())
            except Exception:
                pass
        # Find first {...} block
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            return {}
        return {}

    def _infer_screen_type(self, texts: List[str], elements: List[UIElement]) -> ScreenType:
        words = ' '.join([t.lower() for t in texts]) + ' ' + ' '.join([(e.text or '').lower() for e in elements])
        if any(k in words for k in ['login', 'signin', 'password', 'username']):
            return ScreenType.LOGIN
        if any(k in words for k in ['dashboard', 'chart', 'graph', 'metric']):
            return ScreenType.DASHBOARD
        if any(k in words for k in ['form', 'input', 'textbox', 'field']):
            return ScreenType.FORM
        if any(k in words for k in ['list', 'table', 'row', 'column']):
            return ScreenType.LIST
        if any(k in words for k in ['settings', 'preference', 'config']):
            return ScreenType.SETTINGS
        if any(k in words for k in ['search', 'find', 'query']):
            return ScreenType.SEARCH
        if any(k in words for k in ['cart', 'checkout', 'order', 'payment']):
            return ScreenType.CHECKOUT
        if any(k in words for k in ['chat', 'message', 'conversation']):
            return ScreenType.CHAT
        if any(k in words for k in ['error', 'warning', 'failed']):
            return ScreenType.ERROR
        return ScreenType.UNKNOWN

    def _fallback_summary(self, elements: List[UIElement]) -> str:
        interactive = sum(1 for e in elements if hasattr(e, 'is_interactive') and e.is_interactive())
        return f"Screen with {len(elements)} elements ({interactive} interactive)"


class MultiVLMProcessor(VLMProcessor):
    """Aggregate results from multiple VLMs and pick/merge best."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processors: List[VLMProcessor] = []

        # Always try ScreenAI-style processor as baseline
        try:
            self.processors.append(ScreenAIProcessor(config))
        except Exception as e:
            logger.warning(f"ScreenAIProcessor unavailable: {e}")

        # TODO: add Qwen2-VL / InternVL adapters here when available

        if not self.processors:
            raise RuntimeError("No VLM processors available for MultiVLMProcessor")

        logger.info(f"MultiVLMProcessor initialized with {len(self.processors)} VLM(s)")

    async def analyze(self,
                      image: Image.Image,
                      texts: List[str],
                      elements: List[UIElement]) -> Dict[str, Any]:
        analyses: List[Dict[str, Any]] = []
        for proc in self.processors:
            try:
                res = await proc.analyze(image, texts, elements)
                analyses.append(res)
            except Exception as e:
                logger.warning(f"VLM processor failed: {e}")

        if not analyses:
            raise RuntimeError("All VLM analyses failed")

        # Pick the highest confidence analysis (fallback: first)
        best = max(analyses, key=lambda a: a.get('confidence', 0.0))
        return best
