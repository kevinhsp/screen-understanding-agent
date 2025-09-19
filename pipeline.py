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
                     image_path: Optional[str] = None) -> ScreenUnderstanding:
        """
        Process a screen image and return understanding results
        
        Args:
            image: PIL Image object
            image_path: Optional path to the image file
            
        Returns:
            ScreenUnderstanding object with complete analysis
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
            ocr_results = await self._run_ocr(image)
            
            # Step 2/3: Element Detection (+ optional fused SOM-like pipeline)
            elements = await self._detect_elements(image)
            # Step 3: Merge OCR with Elements
            elements = await self._merge_ocr_elements(elements, ocr_results)
            
            # Step 4: VLM Analysis
            vlm_analysis = await self._run_vlm_analysis(image, ocr_results, elements)
            
            # Step 5: Create final understanding
            understanding = self._create_understanding(
                vlm_analysis, elements, ocr_results, time.time() - start_time
            )
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                await self._save_intermediate_results(
                    image_path or "unknown",
                    ocr_results,
                    elements,
                    vlm_analysis,
                    understanding
                )
            
            logger.info(f"Pipeline completed in {understanding.processing_time_ms:.2f}ms")
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

    def _parse_freeform_summary_actions(self, text: str) -> (str, List[Affordance]):
        """Parse freeform output into summary text and Affordance list.

        Heuristics:
        - Summary: lines before the first bullet/action line.
        - Actions: lines starting with '-', '•', '–' or containing ' - ' / ' – '.
        """
        lines = [l.rstrip() for l in text.splitlines()]
        # Identify action lines
        def is_action_line(l: str) -> bool:
            s = l.strip()
            return s.startswith(('-', '•', '–')) or ' - ' in s or ' – ' in s

        first_action_idx = next((i for i, l in enumerate(lines) if is_action_line(l)), None)
        if first_action_idx is None:
            summary = text.strip()
            actions: List[Affordance] = []
            return summary, actions

        summary_lines = [l.strip() for l in lines[:first_action_idx] if l.strip()]
        summary = ' '.join(summary_lines).strip()

        action_lines = [l.strip() for l in lines[first_action_idx:] if is_action_line(l)]
        affordances: List[Affordance] = []
        for ln in action_lines:
            s = ln.lstrip('-•–').strip()
            # keep short target label after a dash if present
            desc = s
            affordances.append(Affordance(
                action_type=ActionType.CLICK,
                target_element=None,
                description=desc,
                priority=8
            ))
        # Deduplicate by description
        seen = set()
        uniq: List[Affordance] = []
        for a in affordances:
            key = a.description.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(a)
        return summary or text.strip(), uniq[:50]

    def _extract_summary_after_label(self, text: str) -> Optional[str]:
        """Extract the line immediately following a 'Summary:' label (case-insensitive).

        Rules:
        - If 'Summary:' appears with content on the same line, return content after ':' on that line.
        - Else, return the first non-empty line immediately after the line that contains 'Summary:'.
        - If not found, return None.
        """
        try:
            if not isinstance(text, str):
                return None
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if re.match(r"^\s*summary\s*:.*$", line, re.I):
                    # Content on the same line after 'Summary:'
                    m = re.match(r"^\s*summary\s*:\s*(.*)$", line, re.I)
                    if m and m.group(1).strip():
                        return m.group(1).strip()
                    # Otherwise take next non-empty line
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        return lines[j].strip()
                    return None
            return None
        except Exception:
            return None

    def _strip_chat_preamble(self, text: str) -> str:
        """Remove typical chat-template role markers from the beginning of text.

        Handles plain markers (system/user/assistant) and special-token style
        markers (<|im_start|>system ... <|im_end|> etc.). Returns stripped text.
        """
        if not isinstance(text, str):
            return text
        s = text
        try:
            # Remove plain 'system ... user' preface if echoed
            s = re.sub(r"(?is)^\s*system\s+.*?\n\s*user\s+", "", s, count=1)
            # Drop a leading 'assistant' label line if present
            s = re.sub(r"(?im)^assistant\s*\n", "", s)
            # Remove special token style roles
            s = re.sub(r"(?is)<\|im_start\|>system.*?<\|im_end\|>\s*", "", s)
            s = re.sub(r"(?is)<\|im_start\|>user\s*", "", s)
            s = re.sub(r"(?is)<\|im_start\|>assistant\s*", "", s)
        except Exception:
            pass
        return s.strip()

    def _extract_summary_from_freeform(self, text: str) -> str:
        """Extract the first natural-language summary paragraph from freeform output.

        Heuristics:
        - Skip instruction-like lines ("You are an expert", "Analyze the website", etc.)
        - If a '1) Summary' header exists, start after it
        - Stop when hitting the next numbered header (e.g., '2)') or a bullet/action line
        - Fallback to first 2 sentences if nothing extracted
        """
        if not isinstance(text, str):
            return str(text)
        s = text.strip()
        # If there's an explicit '1) Summary' header, start after it
        m = re.search(r"(?im)^\s*1\)\s*summary.*?$", s)
        if m:
            s = s[m.end():].lstrip('\n')
        lines = s.splitlines()
        def is_action_line(l: str) -> bool:
            t = l.strip()
            return t.startswith(('-', '•', '–')) or ' - ' in t or ' – ' in t
        def is_instruction(l: str) -> bool:
            low = l.lower()
            return any(k in low for k in [
                'you are an expert', 'analyze the website', 'no code', 'no json', 'no markdown',
                'return only', 'context ocr texts', 'high-confidence actions', 'high-confidence entities'
            ]) or re.match(r"^\s*\d+\)\s*", l)
        # If there's a literal 'Summary:' header, start after it
        start_idx = 0
        for i, l in enumerate(lines):
            if re.match(r"^\s*summary\s*:?\s*$", l, re.I):
                start_idx = i + 1
                break
        summary_lines = []
        for l in lines[start_idx:]:
            if not l.strip():
                # keep paragraph spacing until we collected something
                if summary_lines:
                    break
                continue
            # Stop at next section or actions header
            if re.match(r"^\s*2\)\b", l) or re.match(r"^\s*high[- ]?confidence\b", l, re.I):
                break
            if is_instruction(l) or is_action_line(l):
                if summary_lines:
                    break
                else:
                    continue
            summary_lines.append(l.strip())
        summary = ' '.join(summary_lines).strip()
        # Remove a possible leading 'Summary:' prefix
        summary = re.sub(r"(?i)^\s*summary\s*:\s*", "", summary).strip()
        if summary:
            return summary
        # Fallback: take first 2 sentences
        sentences = re.split(r"(?<=[.!?])\s+", s)
        return ' '.join(sentences[:2]).strip()
    
    def _create_understanding(self,
                            vlm_analysis: Dict[str, Any],
                            elements: List[UIElement],
                            ocr_results: List[OCRResult],
                            processing_time: float) -> ScreenUnderstanding:
        """Create final ScreenUnderstanding object"""
        logger.info("Step 5: Creating final understanding...")
        
        # Extract components from VLM analysis
        screen_type = vlm_analysis.get("screen_type", ScreenType.UNKNOWN)
        summary = vlm_analysis.get("summary", "")
        affordances = vlm_analysis.get("affordances", [])
        entities = vlm_analysis.get("entities", [])
        warnings = vlm_analysis.get("warnings", [])
        
        # Calculate confidence scores
        confidence_scores = {
            "overall": vlm_analysis.get("confidence", 0.0),
            "ocr": sum(ocr.confidence for ocr in ocr_results) / len(ocr_results) if ocr_results else 0.0,
            "element_detection": sum(e.confidence for e in elements) / len(elements) if elements else 0.0,
            "vlm": vlm_analysis.get("confidence", 0.0)
        }
        
        # Create metadata
        metadata = {
            "image_size": None,  # Will be set if image provided
            "ocr_text_count": len(ocr_results),
            "element_count": len(elements),
            "interactive_element_count": sum(1 for e in elements if e.is_interactive()),
            "entity_count": len(entities),
            "affordance_count": len(affordances),
            "processing_stages": {
                "ocr": type(self.ocr_processor).__name__,
                "element_detection": "OmniParser",
                "vlm": self.config.vlm_model_name
            },
            "freeform_text": vlm_analysis.get('__freeform_text') if isinstance(vlm_analysis, dict) else None
        }
        
        # Create understanding object
        understanding = ScreenUnderstanding(
            screen_type=screen_type,
            summary=summary,
            affordances=affordances,
            entities=entities,
            elements=elements,
            confidence_scores=confidence_scores,
            metadata=metadata,
            warnings=warnings,
            processing_time_ms=processing_time * 1000
        )
        
        logger.info(f"Understanding created: {screen_type.value} screen with {len(elements)} elements")
        
        return understanding
    
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
                           save_results: bool = True) -> List[ScreenUnderstanding]:
        """Process multiple images in batch"""
        logger.info(f"Processing batch of {len(image_paths)} images...")
        
        results = []
        for idx, image_path in enumerate(image_paths):
            logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
            
            try:
                image = Image.open(image_path)
                understanding = await self.process(image, image_path)
                results.append(understanding)
                
                if save_results:
                    output_path = Path(image_path).with_suffix('.json')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        if getattr(self.config, 'dw_minimal_output', False):
                            minimal = {
                                'summary': understanding.summary,
                                'affordances': [a.to_dict() for a in understanding.affordances],
                                'elements': [e.to_dict() for e in understanding.elements],
                            }
                            json.dump(minimal, f, indent=2, ensure_ascii=False)
                        else:
                            json.dump(understanding.to_dict(), f, indent=2, ensure_ascii=False)
                    logger.info(f"Results saved to {output_path}")

                    # In minimal mode also save freeform raw text next to image for inspection
                    if getattr(self.config, 'dw_minimal_output', False):
                        free_text = understanding.metadata.get('freeform_text') if isinstance(understanding.metadata, dict) else None
                        if free_text:
                            txt_path = Path(image_path).with_suffix('.vlm_freeform.txt')
                            try:
                                with open(txt_path, 'w', encoding='utf-8') as tf:
                                    tf.write(free_text)
                                logger.info(f"Freeform VLM text saved to {txt_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save freeform text to {txt_path}: {e}")
                    
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
        vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",  # Replace with actual model
        use_gpu=True,
        debug_mode=True,
        save_intermediate_results=True,
        enable_entity_extraction=True,
        enable_affordance_detection=True,
        ocr_backend="hybrid"
    )
    
    # Initialize pipeline
    pipeline = ScreenUnderstandingPipeline(config)
    
    # Process a single image
    image_path = "examples/login_screen.png"  # Replace with actual image
    
    # Create a sample image for testing
    sample_image = Image.new('RGB', (1024, 768), color='white')
    
    try:
        # Process the image
        understanding = await pipeline.process(sample_image, "sample_image")
        
        # Print results
        print("\n" + "="*60)
        print("SCREEN UNDERSTANDING RESULTS")
        print("="*60)
        print(f"Screen Type: {understanding.screen_type.value}")
        print(f"Summary: {understanding.summary}")
        print(f"Processing Time: {understanding.processing_time_ms:.2f}ms")
        print(f"Overall Confidence: {understanding.confidence_scores.get('overall', 0):.2%}")
        
        print("\n📋 DETECTED ELEMENTS:")
        for elem in understanding.elements[:5]:
            print(f"  • {elem.role.value}: {elem.text or '[no text]'} "
                  f"({elem.state.value}) [conf: {elem.confidence:.2f}]")
        
        if len(understanding.elements) > 5:
            print(f"  ... and {len(understanding.elements) - 5} more elements")
        
        print("\n🎯 POSSIBLE ACTIONS:")
        for aff in understanding.get_high_priority_actions():
            print(f"  • {aff.action_type.value}: {aff.description} "
                  f"[priority: {aff.priority}]")
        
        print("\n📊 EXTRACTED ENTITIES:")
        for entity in understanding.entities[:10]:
            print(f"  • {entity.entity_type}: {entity.value} "
                  f"[conf: {entity.confidence:.2f}]")
        
        if understanding.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in understanding.warnings:
                print(f"  • {warning}")
        
        # Export to JSON
        output_file = "screen_understanding_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(understanding.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n✅ Complete results saved to {output_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
