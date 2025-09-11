"""
processors.py - OCR, OmniParser and VLM processing modules
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import paddle
from PIL import Image
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoModel
)
from paddleocr import PaddleOCR
import cv2

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
    """PaddleOCR v4/v5 processor"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        logger.info("Initializing PaddleOCR...")
        paddle.device.set_device('gpu:0' if config.use_gpu else 'cpu')
        # Initialize PaddleOCR with GPU support if available
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang=config.ocr_language,
            # show_log=config.debug_mode
        )


    async def process(self, image: Image.Image) -> List[OCRResult]:
        """Execute OCR recognition"""
        logger.info("Running PaddleOCR processing...")

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        result = self.ocr.predict(img_array)

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


# ============= OmniParser Module =============

class OmniParserV2:
    """OmniParser v2 - UI element detection using Hugging Face models"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        logger.info("Initializing OmniParser v2...")

        # Load OmniParser model from Hugging Face
        model_name = "microsoft/OmniParser"  # Example model name
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if config.use_gpu else torch.float32
            )

            if config.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()

        except Exception as e:
            logger.warning(f"Could not load OmniParser model: {e}")
            logger.info("Using fallback detection method")
            self.processor = None
            self.model = None

    async def detect_elements(self, image: Image.Image) -> List[UIElement]:
        """Detect UI elements in the image"""
        logger.info("Detecting UI elements with OmniParser...")

        if self.model is None:
            # Fallback to basic detection
            return await self._fallback_detection(image)

        # Process image with model
        inputs = self.processor(images=image, return_tensors="pt")

        if self.config.use_gpu and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Parse model outputs to extract elements
        elements = self._parse_model_outputs(outputs, image.size)

        # Filter by confidence threshold
        elements = [e for e in elements if e.confidence >= self.config.element_detection_threshold]

        # Limit number of elements
        if len(elements) > self.config.max_elements:
            elements = sorted(elements, key=lambda x: x.confidence, reverse=True)[:self.config.max_elements]

        logger.info(f"Detected {len(elements)} UI elements")
        return elements

    async def _fallback_detection(self, image: Image.Image) -> List[UIElement]:
        """Fallback detection using traditional CV methods"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        elements = []
        element_id = 0

        # Detect buttons using edge detection and contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter small contours
            if w < 30 or h < 20:
                continue

            # Heuristic to determine element type
            aspect_ratio = w / h
            if 2 <= aspect_ratio <= 5 and 30 <= h <= 60:
                role = ElementRole.BUTTON
            elif 3 <= aspect_ratio <= 10 and 25 <= h <= 40:
                role = ElementRole.INPUT
            else:
                role = ElementRole.CONTAINER

            elements.append(UIElement(
                id=f"element_{element_id}",
                role=role,
                text="",
                state=ElementState.ENABLED,
                bbox=BoundingBox(x, y, w, h),
                confidence=0.7,
                clickable=(role in [ElementRole.BUTTON, ElementRole.INPUT, ElementRole.LINK])
            ))
            element_id += 1

        return elements

    def _parse_model_outputs(self, outputs: Any, image_size: Tuple[int, int]) -> List[UIElement]:
        """Parse model outputs to extract UI elements"""
        elements = []
        width, height = image_size

        # This is a placeholder - actual implementation depends on model output format
        # Assuming outputs contain bounding boxes and classifications

        if hasattr(outputs, 'pred_boxes') and hasattr(outputs, 'pred_classes'):
            boxes = outputs.pred_boxes[0].cpu().numpy()
            classes = outputs.pred_classes[0].cpu().numpy()
            scores = outputs.scores[0].cpu().numpy() if hasattr(outputs, 'scores') else [0.8] * len(boxes)

            role_mapping = {
                0: ElementRole.BUTTON,
                1: ElementRole.INPUT,
                2: ElementRole.TEXT,
                3: ElementRole.IMAGE,
                4: ElementRole.LINK,
                5: ElementRole.CHECKBOX,
                6: ElementRole.DROPDOWN,
                # Add more mappings based on model's class definitions
            }

            for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                # Convert normalized coordinates to pixels
                x1, y1, x2, y2 = box
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)

                role = role_mapping.get(cls, ElementRole.CONTAINER)

                elements.append(UIElement(
                    id=f"element_{idx}",
                    role=role,
                    text="",
                    state=ElementState.ENABLED,
                    bbox=BoundingBox(x1, y1, x2-x1, y2-y1),
                    confidence=float(score),
                    clickable=(role in [ElementRole.BUTTON, ElementRole.INPUT, ElementRole.LINK])
                ))

        return elements

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

        # Add OCR results that don't match any element as TEXT elements
        matched_ocr = set()
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
    """ using Hugging Face"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        logger.info(f"Initializing ScreenAI with model: {config.vlm_model_name}")

        # Load ScreenAI model from Hugging Face
        try:
            # Note: Replace with actual ScreenAI model when available
            # Currently using a placeholder vision-language model
            model_name = config.vlm_model_name or "microsoft/Florence-2-large"

            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if config.use_gpu else torch.float32
            )

            if config.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("ScreenAI model loaded on GPU")
            else:
                logger.info("ScreenAI model loaded on CPU")

        except Exception as e:
            logger.error(f"Failed to load ScreenAI model: {e}")
            raise

    async def analyze(self,
                     image: Image.Image,
                     texts: List[str],
                     elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze screen content using ScreenAI"""
        logger.info("Analyzing screen with ScreenAI...")

        # Build comprehensive prompt
        prompt = self._build_prompt(texts, elements)

        # Process with model
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            max_length=self.config.vlm_max_tokens
        )

        if self.config.use_gpu and torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.vlm_max_tokens,
                temperature=self.config.vlm_temperature,
                do_sample=True
            )

        # Decode outputs
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Parse response to structured format
        analysis = self._parse_vlm_response(response, texts, elements)

        return analysis

    def _build_prompt(self, texts: List[str], elements: List[UIElement]) -> str:
        """Build comprehensive prompt for VLM"""

        # Prepare element descriptions
        interactive_elements = [e for e in elements if e.is_interactive()]
        element_descriptions = []
        for e in interactive_elements[:20]:  # Limit to avoid token overflow
            desc = f"- {e.role.value}"
            if e.text:
                desc += f": '{e.text}'"
            element_descriptions.append(desc)

        # Prepare text snippets
        text_snippets = texts[:15]  # Limit text snippets

        prompt = f"""Analyze this screen image and provide a structured understanding.

Detected text on screen:
{chr(10).join(text_snippets)}

Interactive UI elements found:
{chr(10).join(element_descriptions)}

Please provide:
1. Screen type (login/dashboard/form/list/detail/settings/search/checkout/chat/media/error/unknown)
2. A brief summary of the screen's purpose and content (2-3 sentences)
3. Main actions a user can perform (list up to 5 most important)
4. Key information entities visible (dates, prices, emails, names, etc.)
5. Any warnings or important notices

Format your response as:
SCREEN_TYPE: [type]
SUMMARY: [brief description]
ACTIONS: [action1], [action2], ...
ENTITIES: [entity1: value1], [entity2: value2], ...
WARNINGS: [any warnings or None]
"""

        return prompt

    def _parse_vlm_response(self, response: str, texts: List[str], elements: List[UIElement]) -> Dict[str, Any]:
        """Parse VLM response to structured format"""

        # Initialize default values
        screen_type = ScreenType.UNKNOWN
        summary = "Unable to determine screen content"
        affordances = []
        entities = []
        warnings = []

        # Parse response lines
        lines = response.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('SCREEN_TYPE:'):
                type_str = line.replace('SCREEN_TYPE:', '').strip().lower()
                try:
                    screen_type = ScreenType[type_str.upper()]
                except:
                    screen_type = ScreenType.UNKNOWN

            elif line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()

            elif line.startswith('ACTIONS:'):
                actions_str = line.replace('ACTIONS:', '').strip()
                actions = [a.strip() for a in actions_str.split(',')]
                affordances = self._create_affordances(actions, elements)

            elif line.startswith('ENTITIES:'):
                entities_str = line.replace('ENTITIES:', '').strip()
                entities = self._extract_entities(entities_str, texts)

            elif line.startswith('WARNINGS:'):
                warning = line.replace('WARNINGS:', '').strip()
                if warning.lower() != 'none':
                    warnings.append(warning)

        # If parsing fails, use fallback analysis
        if screen_type == ScreenType.UNKNOWN:
            screen_type = self._infer_screen_type(elements, texts)

        if not summary or summary == "Unable to determine screen content":
            summary = self._generate_summary(screen_type, elements, texts)

        if not affordances:
            affordances = self._infer_affordances(elements, screen_type)

        if not entities:
            entities = self._extract_entities_from_text(texts)

        return {
            "screen_type": screen_type,
            "summary": summary,
            "affordances": affordances,
            "entities": entities,
            "warnings": warnings,
            "confidence": 0.85  # Placeholder confidence
        }

    def _create_affordances(self, actions: List[str], elements: List[UIElement]) -> List[Affordance]:
        """Create affordance objects from action descriptions"""
        affordances = []

        for idx, action in enumerate(actions[:10]):  # Limit to 10 actions
            action_lower = action.lower()

            # Determine action type
            if 'click' in action_lower or 'tap' in action_lower:
                action_type = ActionType.CLICK
            elif 'type' in action_lower or 'enter' in action_lower or 'input' in action_lower:
                action_type = ActionType.TYPE
            elif 'scroll' in action_lower:
                action_type = ActionType.SCROLL
            elif 'select' in action_lower or 'choose' in action_lower:
                action_type = ActionType.SELECT
            elif 'swipe' in action_lower:
                action_type = ActionType.SWIPE
            else:
                action_type = ActionType.CLICK

            # Try to match with element
            target_element = None
            for element in elements:
                if element.text and element.text.lower() in action_lower:
                    target_element = element.id
                    break

            affordances.append(Affordance(
                action_type=action_type,
                target_element=target_element,
                description=action,
                priority=10 - idx  # Higher priority for earlier actions
            ))

        return affordances

    def _extract_entities(self, entities_str: str, texts: List[str]) -> List[ExtractedEntity]:
        """Extract entities from parsed string"""
        entities = []

        # Parse entity pairs
        entity_pairs = entities_str.split(',')
        for pair in entity_pairs:
            if ':' in pair:
                entity_type, value = pair.split(':', 1)
                entity_type = entity_type.strip()
                value = value.strip()

                # Determine entity type
                if '@' in value:
                    entity_type = 'email'
                elif '$' in value or '€' in value or '£' in value:
                    entity_type = 'price'
                elif any(char.isdigit() for char in value) and ('/' in value or '-' in value):
                    entity_type = 'date'

                entities.append(ExtractedEntity(
                    entity_type=entity_type,
                    value=value,
                    confidence=0.8,
                    source='vlm'
                ))

        return entities

    def _infer_screen_type(self, elements: List[UIElement], texts: List[str]) -> ScreenType:
        """Infer screen type from elements and text"""
        text_combined = ' '.join(texts).lower()

        # Check for specific keywords
        if 'login' in text_combined or 'sign in' in text_combined or 'password' in text_combined:
            return ScreenType.LOGIN
        elif 'dashboard' in text_combined or 'overview' in text_combined:
            return ScreenType.DASHBOARD
        elif 'settings' in text_combined or 'preferences' in text_combined:
            return ScreenType.SETTINGS
        elif 'search' in text_combined and any(e.role == ElementRole.INPUT for e in elements):
            return ScreenType.SEARCH
        elif 'checkout' in text_combined or 'payment' in text_combined or 'cart' in text_combined:
            return ScreenType.CHECKOUT
        elif 'chat' in text_combined or 'message' in text_combined:
            return ScreenType.CHAT
        elif 'error' in text_combined or 'not found' in text_combined:
            return ScreenType.ERROR

        # Check element composition
        input_count = sum(1 for e in elements if e.role == ElementRole.INPUT)
        button_count = sum(1 for e in elements if e.role == ElementRole.BUTTON)

        if input_count >= 3:
            return ScreenType.FORM
        elif button_count > 5:
            return ScreenType.LIST

        return ScreenType.UNKNOWN

    def _generate_summary(self, screen_type: ScreenType, elements: List[UIElement], texts: List[str]) -> str:
        """Generate summary based on screen type and content"""
        summaries = {
            ScreenType.LOGIN: "Authentication screen with user credentials input",
            ScreenType.DASHBOARD: "Main dashboard displaying overview information",
            ScreenType.FORM: "Data input form with multiple fields",
            ScreenType.LIST: "List view displaying multiple items",
            ScreenType.SETTINGS: "Settings or configuration screen",
            ScreenType.SEARCH: "Search interface for finding content",
            ScreenType.CHECKOUT: "Checkout or payment processing screen",
            ScreenType.CHAT: "Messaging or chat interface",
            ScreenType.ERROR: "Error or warning message display",
            ScreenType.UNKNOWN: "Screen with mixed content elements"
        }

        base_summary = summaries.get(screen_type, "Screen content")

        # Add element count info
        interactive_count = sum(1 for e in elements if e.is_interactive())
        if interactive_count > 0:
            base_summary += f" with {interactive_count} interactive elements"

        return base_summary

    def _infer_affordances(self, elements: List[UIElement], screen_type: ScreenType) -> List[Affordance]:
        """Infer possible actions from elements"""
        affordances = []

        # Add actions for interactive elements
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

        # Add screen-level actions
        if screen_type == ScreenType.LIST:
            affordances.append(Affordance(
                action_type=ActionType.SCROLL,
                description="Scroll to see more items",
                priority=4
            ))

        # Sort by priority
        affordances.sort(key=lambda x: x.priority, reverse=True)

        return affordances[:10]  # Return top 10 actions

    def _extract_entities_from_text(self, texts: List[str]) -> List[ExtractedEntity]:
        """Extract entities from text using regex patterns"""
        entities = []

        for text in texts:
            # Extract emails
            emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            for email in emails:
                entities.append(ExtractedEntity(
                    entity_type="email",
                    value=email,
                    context=text,
                    confidence=0.95,
                    source='regex'
                ))

            # Extract prices
            prices = re.findall(r'[$\u20ac\u00a3\u00a5]\s*\d+(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?\s*(?:USD|EUR|GBP)', text)
            for price in prices:
                entities.append(ExtractedEntity(
                    entity_type="price",
                    value=price,
                    context=text,
                    confidence=0.9,
                    source='regex'
                ))

            # Extract dates
            dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}', text)
            for date in dates:
                entities.append(ExtractedEntity(
                    entity_type="date",
                    value=date,
                    context=text,
                    confidence=0.85,
                    source='regex'
                ))

            # Extract phone numbers
            phones = re.findall(r'(?:\+\d{1,3}\s?)?(?:\(\d{1,4}\)\s?)?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}', text)
            for phone in phones:
                if len(phone) >= 10:  # Basic validation
                    entities.append(ExtractedEntity(
                        entity_type="phone",
                        value=phone,
                        context=text,
                        confidence=0.8,
                        source='regex'
                    ))

            # Extract URLs
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
            for url in urls:
                entities.append(ExtractedEntity(
                    entity_type="url",
                    value=url,
                    context=text,
                    confidence=0.95,
                    source='regex'
                ))

        return entities


class MultiVLMProcessor(VLMProcessor):
    """Processor that combines multiple VLM models for better accuracy"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processors = []

        # Initialize multiple VLM processors
        logger.info("Initializing Multi-VLM processor...")

        # Add ScreenAI as primary
        self.processors.append(ScreenAIProcessor(config))

        # Add other models if configured
        if config.use_multiple_vlms:
            # Could add Qwen2-VL, InternVL, etc.
            pass

    async def analyze(self,
                     image: Image.Image,
                     texts: List[str],
                     elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze using multiple models and combine results"""

        if len(self.processors) == 1:
            return await self.processors[0].analyze(image, texts, elements)

        # Run all processors in parallel
        tasks = [p.analyze(image, texts, elements) for p in self.processors]
        results = await asyncio.gather(*tasks)

        # Combine results (voting, averaging, etc.)
        return self._combine_results(results)

    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple VLM models"""

        # Simple voting for screen type
        screen_types = [r.get("screen_type", ScreenType.UNKNOWN) for r in results]
        most_common_type = max(set(screen_types), key=screen_types.count)

        # Combine summaries
        summaries = [r.get("summary", "") for r in results if r.get("summary")]
        combined_summary = summaries[0] if summaries else "Unable to determine screen content"

        # Merge affordances (deduplicate)
        all_affordances = []
        seen_actions = set()
        for r in results:
            for aff in r.get("affordances", []):
                action_key = (aff.action_type, aff.target_element)
                if action_key not in seen_actions:
                    all_affordances.append(aff)
                    seen_actions.add(action_key)

        # Merge entities (deduplicate)
        all_entities = []
        seen_entities = set()
        for r in results:
            for ent in r.get("entities", []):
                entity_key = (ent.entity_type, ent.value)
                if entity_key not in seen_entities:
                    all_entities.append(ent)
                    seen_entities.add(entity_key)

        # Average confidence scores
        confidences = [r.get("confidence", 0) for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "screen_type": most_common_type,
            "summary": combined_summary,
            "affordances": all_affordances,
            "entities": all_entities,
            "confidence": avg_confidence,
            "warnings": []
        }