"""
pipeline.py - Main screen understanding pipeline orchestrator
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np

from models import (
    ScreenUnderstanding, ProcessingConfig, UIElement, 
    OCRResult, Affordance, ExtractedEntity, ScreenType
)
from processors import (
    PaddleOCRProcessor, OmniParserV2, ScreenAIProcessor, 
    MultiVLMProcessor, OCRProcessor, VLMProcessor
)
from decision_agent import DecisionAgent
import re
import os
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


class ScreenUnderstandingPipeline:
    """Main pipeline for screen understanding"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize pipeline with configuration"""
        self.config = config or ProcessingConfig()
        
        logger.info("Initializing Screen Understanding Pipeline...")
        logger.info(f"Configuration: {self.config.to_dict()}")
        
        # Initialize components
        self._initialize_components()
        
        # Create output directory if saving intermediate results
        if self.config.save_intermediate_results:
            self.output_dir = Path("pipeline_outputs")
            self.output_dir.mkdir(exist_ok=True)
            logger.info(f"Intermediate results will be saved to {self.output_dir}")
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        logger.info("Initializing pipeline components...")
        
        # Initialize OCR processor (selectable backend)
        self.ocr_processor = self._init_ocr_backend()
        
        # Initialize element detector
        self.element_detector = OmniParserV2(self.config)
        
        # Initialize VLM processor(s)
        if self.config.use_multiple_vlms:
            self.vlm_processor = MultiVLMProcessor(self.config)
        else:
            # Robustly select VLM processor based on model name
            raw = str(self.config.vlm_model_name or '')
            name = raw.lower()
            try:
                from processors import QwenVLMProcessor
                logger.info(f"Selecting QwenVLMProcessor for model '{raw}'")
                self.vlm_processor = QwenVLMProcessor(self.config)
            except Exception as e:
                logger.error(f"VLM selection failed for model '{raw}': {e}")

        logger.info("All components initialized successfully")

        # Optionally preload the decision agent (GPT) at startup
        self.decision_agent = None
        try:
            if getattr(self.config, 'decider_enabled', False):
                from decision_agent import DecisionAgent
                self.decision_agent = DecisionAgent(
                    model_name=getattr(self.config, 'decider_model_name', 'openai/gpt-oss-20b'),
                    use_gpu=self.config.use_gpu,
                )
                logger.info("DecisionAgent initialized and ready")
        except Exception as e:
            logger.warning(f"DecisionAgent preload failed (will attempt later if needed): {e}")

    def _init_ocr_backend(self) -> OCRProcessor:
        """Initialize OCR backend based on config, with graceful fallbacks."""
        backend = (self.config.ocr_backend or 'auto').lower()
        logger.info(f"Selecting OCR backend: {backend}")

        def try_alt(name: str):
            try:
                alt = importlib.import_module('alternative_ocr')
                cls = getattr(alt, name)
                return cls(self.config)
            except Exception as e:
                logger.warning(f"Alternative OCR '{name}' unavailable: {e}")
                return None

        if backend in {"paddle", "ppocr", "paddleocr"}:
            try:
                return PaddleOCRProcessor(self.config)
            except Exception as e:
                logger.warning(f"PaddleOCR init failed, falling back: {e}")

        if backend in {"easyocr"}:
            proc = try_alt('EasyOCRProcessor')
            if proc:
                return proc
        elif backend in {"trocr"}:
            proc = try_alt('TrOCRProcessor')
            if proc:
                return proc
        elif backend in {"tesseract"}:
            proc = try_alt('TesseractOCRProcessor')
            if proc:
                return proc
        elif backend in {"hybrid", "auto"}:
            proc = try_alt('HybridOCRProcessor')
            if proc:
                return proc
            # if hybrid unavailable, try simple ones
            for cls_name in ["EasyOCRProcessor", "TrOCRProcessor", "TesseractOCRProcessor"]:
                proc = try_alt(cls_name)
                if proc:
                    return proc

        # Final fallback: Paddle if available, else raise
        try:
            return PaddleOCRProcessor(self.config)
        except Exception as e:
            raise RuntimeError(
                "No OCR backend available. Install one of: easyocr, pytesseract, transformers (TrOCR), or PaddleOCR."
            ) from e
    
    async def process(self, 
                     image: Image.Image, 
                     image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a screen image and return understanding results
        
        Args:
            image: PIL Image object
            image_path: Optional path to the image file
            
        Returns:
            Minimal screen understanding dict with keys: summary, affordance, elements
        """
        start_time = time.time()
        logger.info(f"Starting screen understanding pipeline...")
        
        if image_path:
            logger.info(f"Processing image: {image_path}")
        
        # Validate image
        if not self._validate_image(image):
            raise ValueError("Invalid image provided")
        
        try:
            # Step 1: OCR Recognition
            ocr = await self._run_ocr(image)
            
            # Step 2/3: Element Detection (+ optional fused SOM-like pipeline)
            elems = await self._detect_elements(image)
            # Step 3: Merge OCR with Elements
            merged = await self._merge_ocr_elements(elems, ocr)
            
            # Step 4: VLM Analysis (summary + per-element actions)
            summary_actions = await self._run_vlm_generate_summary_actions(image)
            actions = await self._run_vlm_classify_element_actions(
                image, merged, ocr, summary_actions, max_k=self.config.vlm_elements_max
            )

            # Enrich actions with element coordinates for automation (bbox + center)
            try:
                id_to_elem = {e.id: e for e in merged if getattr(e, 'id', None)}
                for a in actions:
                    eid = a.get('element_id')
                    el = id_to_elem.get(eid)
                    if not el:
                        continue
                    b = el.bbox
                    a['bbox'] = {
                        'x': int(b.x), 'y': int(b.y), 'width': int(b.width), 'height': int(b.height)
                    }
                    a['center'] = {
                        'x': int(b.x + b.width // 2),
                        'y': int(b.y + b.height // 2),
                    }
            except Exception as e:
                logger.warning(f"Failed to attach element coordinates: {e}")

            # Step 5: Create final minimal understanding
            understanding = self._create_understanding(
                summary_actions, actions, time.time() - start_time
            )
            
            # Save annotated PNG of element actions similar to tests_slow/test_element_actions_vlm.py
            try:
                if image_path:
                    from PIL import ImageDraw, ImageFont
                    a_by_id = {a.get('element_id'): a for a in (actions or []) if a.get('element_id')}
                    annotated = image.convert('RGB').copy()
                    draw = ImageDraw.Draw(annotated)
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None
                    for el in merged:
                        if not getattr(el, 'clickable', False):
                            continue
                        b = el.bbox
                        x1, y1, x2, y2 = b.x, b.y, b.x + b.width, b.y + b.height
                        color = (255, 0, 0)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        act = a_by_id.get(el.id, {})
                        label = el.id[8:] if isinstance(el.id, str) and len(el.id) > 8 else el.id
                        pa = act.get('primary_action') if isinstance(act, dict) else None
                        desc = act.get('description') if isinstance(act, dict) else None
                        if pa:
                            label += f"-{pa}"
                        if desc:
                            s = desc if len(desc) <= 40 else (desc[:37] + '...')
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
                    out_dir = Path("pipeline_outputs")
                    out_dir.mkdir(exist_ok=True)
                    base = Path(image_path).stem
                    png_path = out_dir / f"{base}_element_actions.png"
                    annotated.save(png_path, format='PNG')
                    logger.info(f"Annotated actions PNG saved to {png_path}")
            except Exception as e:
                logger.warning(f"Failed to save annotated PNG: {e}")

            logger.info(f"Pipeline completed in {understanding.get('processing_time_s', 0):.2f}s")
            return understanding
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validate input image"""
        if image is None:
            return False
        
        # Check image dimensions
        width, height = image.size
        if width < 50 or height < 50:
            logger.warning(f"Image too small: {width}x{height}")
            return False
        
        if width > 10000 or height > 10000:
            logger.warning(f"Image too large: {width}x{height}")
            return False
        
        return True
    
    async def _run_ocr(self, image: Image.Image) -> List[OCRResult]:
        """Run OCR processing"""
        logger.info("Step 1: Running OCR...")
        start = time.time()
        
        ocr_results = await self.ocr_processor.process(image)
        
        elapsed = (time.time() - start) * 1000
        logger.info(f"OCR completed in {elapsed:.2f}ms, found {len(ocr_results)} text regions")
        
        if self.config.debug_mode:
            for idx, result in enumerate(ocr_results[:5]):
                logger.debug(f"  OCR {idx}: '{result.text}' (conf: {result.confidence:.2f})")
        
        return ocr_results
    
    async def _detect_elements(self, image: Image.Image) -> List[UIElement]:
        """Detect UI elements"""
        logger.info("Step 2: Detecting UI elements...")
        start = time.time()
        
        elements = await self.element_detector.detect_elements(image)
        
        elapsed = (time.time() - start) * 1000
        logger.info(f"Element detection completed in {elapsed:.2f}ms, found {len(elements)} elements")
        
        if self.config.debug_mode:
            role_counts = {}
            for elem in elements:
                role_counts[elem.role.value] = role_counts.get(elem.role.value, 0) + 1
            logger.debug(f"  Element types: {role_counts}")
        
        return elements
    
    async def _merge_ocr_elements(self, 
                                  elements: List[UIElement], 
                                  ocr_results: List[OCRResult]) -> List[UIElement]:
        """Merge OCR results with detected elements"""
        logger.info("Step 3: Merging OCR with elements...")
        start = time.time()
        
        merged_elements = await self.element_detector.merge_with_ocr(elements, ocr_results)
        
        elapsed = (time.time() - start) * 1000
        elements_with_text = sum(1 for e in merged_elements if e.text)
        logger.info(f"Merge completed in {elapsed:.2f}ms, {elements_with_text} elements have text")
        
        return merged_elements
    
    async def _run_vlm_classify_element_actions(self,
                                       image: Image.Image,
                                       elements: List[UIElement],
                                       ocr_results: List[OCRResult],
                                       summary_actions:Dict[str, Any] = {},
                                       max_k: int = 52) -> List[Dict[str, Any]]:
        return await self.vlm_processor.classify_element_actions(
            image, elements, ocr_results, summary_actions, max_k=max_k
        )
    
    async def _run_vlm_generate_summary_actions(self,
                                       image: Image.Image)-> Dict[str, Any]:
        return await self.vlm_processor.generate_summary_actions(image)
    
    async def _run_vlm_analysis(self, 
                                image: Image.Image,
                                ocr_results: List[OCRResult],
                                elements: List[UIElement]) -> Dict[str, Any]:
        """Run VLM analysis"""
        logger.info("Step 4: Running VLM analysis...")
        start = time.time()

        # Extract texts for VLM
        texts = [ocr.text for ocr in ocr_results]

        # Minimal mode: use Qwen freeform to produce summary + actions
        if getattr(self.config, 'dw_minimal_output', False) and hasattr(self.vlm_processor, 'generate_freeform_summary_actions'):
            try:
                free_text = await getattr(self.vlm_processor, 'generate_freeform_summary_actions')(image, texts, elements)
            except Exception as e:
                logger.warning(f"Freeform VLM generation failed: {e}")
                free_text = f"[freeform_error] {e}"
            # Minimal mode: use raw freeform text as summary (no cleanup), parse actions
            summary_text = free_text.strip() if isinstance(free_text, str) else str(free_text)
            # If the freeform contains an explicit 'Summary:' section, extract the line after it
            extracted = self._extract_summary_after_label(summary_text)
            if extracted:
                summary_text = extracted
            _, actions = self._parse_freeform_summary_actions(free_text)
            analysis = {
                'summary': summary_text,
                'affordances': actions,
                'confidence': 0.9,
                'warnings': [],
                '__freeform_text': free_text
            }
        else:
            # Run structured VLM analysis
            analysis = await self.vlm_processor.analyze(image, texts, elements)
        
        elapsed = (time.time() - start) * 1000
        logger.info(f"VLM analysis completed in {elapsed:.2f}ms")
        
        if self.config.debug_mode:
            logger.debug(f"  Screen type: {analysis.get('screen_type', 'unknown')}")
            logger.debug(f"  Summary: {analysis.get('summary', '')[:100]}...")
            logger.debug(f"  Affordances: {len(analysis.get('affordances', []))}")
            logger.debug(f"  Entities: {len(analysis.get('entities', []))}")
        
        return analysis


    
    def _create_understanding(self,
                            summary_actions: Dict[str, Any],
                            element_actions: List[Dict[str, Any]],
                            processing_time: float) -> Dict[str, Any]:
        """Create final minimal understanding dict.

        Output keys:
          - summary: str
          - affordance: List[str] (high-confidence actions)
          - elements: List[Dict] with keys {element_id, primary_action, description, secondary_actions, confidence, bbox{x,y,width,height}, center{x,y}}
        """
        logger.info("Step 5: Creating final minimal understanding...")

        result: Dict[str, Any] = {
            "summary": str(summary_actions.get("summary", "")),
            # Use singular key name as requested: "affordance"
            "affordance": list(summary_actions.get("hc_actions", []) or []),
            "elements": element_actions or [],
            # Store processing time in seconds
            "processing_time_s": float(processing_time),
        }

        logger.info(f"Understanding created: {len(result.get('elements', []))} elements, {len(result.get('affordance', []))} affordances")
        return result
    
    async def _save_intermediate_results(self,
                                        image_path: str,
                                        ocr_results: List[OCRResult],
                                        elements: List[UIElement],
                                        vlm_analysis: Dict[str, Any],
                                        understanding: ScreenUnderstanding):
        """Save intermediate results for debugging"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = Path(image_path).stem
        output_prefix = self.output_dir / f"{base_name}_{timestamp}"
        
        # Save OCR results
        ocr_data = [ocr.to_dict() for ocr in ocr_results]
        with open(f"{output_prefix}_ocr.json", "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, indent=2, ensure_ascii=False)
        
        # Save elements
        elements_data = [elem.to_dict() for elem in elements]
        with open(f"{output_prefix}_elements.json", "w", encoding="utf-8") as f:
            json.dump(elements_data, f, indent=2, ensure_ascii=False)
        
        # Save VLM analysis
        with open(f"{output_prefix}_vlm.json", "w", encoding="utf-8") as f:
            json.dump(vlm_analysis, f, indent=2, ensure_ascii=False, default=str)
        
        # Save final understanding
        with open(f"{output_prefix}_understanding.json", "w", encoding="utf-8") as f:
            json.dump(understanding.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Intermediate results saved to {output_prefix}_*.json")
    
    async def process_batch(self, 
                           image_paths: List[str],
                           save_results: bool = True) -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        logger.info(f"Processing batch of {len(image_paths)} images...")
        results: List[Dict[str, Any]] = []
        out_dir = Path("pipeline_outputs")
        out_dir.mkdir(exist_ok=True)
        for idx, image_path in enumerate(image_paths):
            logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
            
            try:
                image = Image.open(image_path).convert('RGB')
                understanding = await self.process(image, image_path)
                results.append(understanding)
                
                if save_results:
                    base = Path(image_path).stem if image_path else "output"
                    output_path = out_dir / f"{base}_screen_understanding_output.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(understanding, f, indent=2, ensure_ascii=False)
                    logger.info(f"Results saved to {output_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append(None)
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch processing completed: {successful}/{len(image_paths)} successful")
        
        return results


async def main():
    """Example usage of the pipeline"""
    
    # Create configuration
    config = ProcessingConfig(
        ocr_language="en",
        vlm_model_name="Qwen/Qwen2.5-VL-3B-Instruct",  # Replace with actual model
        use_gpu=True,
        debug_mode=True,
        save_intermediate_results=True,
        enable_entity_extraction=True,
        enable_affordance_detection=True,
        ocr_backend="hybrid"
    )
    
    # Initialize pipeline
    pipeline = ScreenUnderstandingPipeline(config)
    
    # Process a single image (allow override via env IMAGE_PATH)
    image_path = os.environ.get('IMAGE_PATH') or "examples/united_sample.png"
    
    # Create a sample image for testing
    img = Image.open(image_path).convert('RGB')
    
    try:
        # Process the image
        understanding = await pipeline.process(img, image_path)

        # If minimal dict is returned, print and save accordingly, then exit
        print("\n" + "="*60)
        print("SCREEN UNDERSTANDING (MINIMAL)")
        print("="*60)
        print(f"Summary: {understanding.get('summary','')}")
        print(f"Processing Time: {understanding.get('processing_time_s', 0):.2f}s")
        print(f"Affordance count: {len(understanding.get('affordance', []))}")
        print(f"Elements (actions) count: {len(understanding.get('elements', []))}")
        # Save JSON under pipeline_outputs/<basename>_screen_understanding_output.json
        out_dir = Path("pipeline_outputs")
        out_dir.mkdir(exist_ok=True)
        base = Path(image_path).stem if image_path else "output"
        output_path = out_dir / f"{base}_screen_understanding_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(understanding, f, indent=2, ensure_ascii=False)
        print(f"\nComplete results saved to {output_path}")
        # pipeline.vlm_processor.__exit__()
        # Optional: decision step using openai/gpt-oss-20b when a DW_TASK is provided
        # Allow either DW or DW_TASK as the task variable name
        task = os.environ.get('DW') or os.environ.get('DW_TASK')
        if task:
            try:
                # Reuse preloaded agent if available; otherwise create on the fly
                agent = getattr(pipeline, 'decision_agent', None)
                if agent is None:
                    # Allow DECIDER_MODEL env override (default: openai/gpt-oss-20b)
                    model_id = os.environ.get('DECIDER_MODEL') or getattr(pipeline.config, 'decider_model_name', 'openai/gpt-oss-20b')
                    agent = DecisionAgent(model_name=model_id, use_gpu=pipeline.config.use_gpu)
                decision = agent.decide(understanding, task)
                dec_path = out_dir / f"{base}_decision.json"
                with open(dec_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'task': task,
                        'decision': decision
                    }, f, indent=2, ensure_ascii=False)
                print(f"Decision saved to {dec_path}")
            except Exception as e:
                print(f"Decision step failed: {e}")
        return

        
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
