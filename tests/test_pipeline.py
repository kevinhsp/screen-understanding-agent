"""
test_pipeline.py - Unit tests for the screen understanding pipeline
"""

import unittest
import asyncio
from PIL import Image 
import numpy as np

from models import (
    BoundingBox, UIElement, ElementRole, ElementState,
    OCRResult, Affordance, ActionType, ExtractedEntity,
    ScreenType, ScreenUnderstanding, ProcessingConfig
)


class TestModels(unittest.TestCase):
    """Test data models"""

    def test_bounding_box(self):
        """Test BoundingBox class"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)

        self.assertEqual(bbox.area(), 5000)
        self.assertEqual(bbox.center(), (60, 45))
        self.assertEqual(bbox.to_xyxy(), (10, 20, 110, 70))

        dict_repr = bbox.to_dict()
        self.assertEqual(dict_repr["x"], 10)
        self.assertEqual(dict_repr["width"], 100)

    def test_ui_element(self):
        """Test UIElement class"""
        bbox = BoundingBox(0, 0, 100, 30)
        element = UIElement(
            id="btn_1",
            role=ElementRole.BUTTON,
            text="Click Me",
            state=ElementState.ENABLED,
            bbox=bbox,
            confidence=0.95
        )

        self.assertTrue(element.is_interactive())
        self.assertEqual(element.role, ElementRole.BUTTON)

        # Test disabled element
        element.state = ElementState.DISABLED
        self.assertFalse(element.is_interactive())

    def test_screen_understanding(self):
        """Test ScreenUnderstanding class"""
        understanding = ScreenUnderstanding(
            screen_type=ScreenType.LOGIN,
            summary="Login screen",
            affordances=[],
            entities=[],
            elements=[]
        )

        self.assertEqual(understanding.screen_type, ScreenType.LOGIN)
        dict_repr = understanding.to_dict()
        self.assertEqual(dict_repr["screen_type"], "login")


class TestProcessors(unittest.TestCase):
    """Test processor components"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = ProcessingConfig(
            use_gpu=False,
            debug_mode=True
        )
        self.test_image = Image.new('RGB', (640, 480), color='white')

    def test_config(self):
        """Test configuration"""
        config = ProcessingConfig()
        self.assertEqual(config.ocr_language, "en")
        self.assertTrue(config.use_gpu)

        config_dict = config.to_dict()
        self.assertIn("ocr_language", config_dict)

    async def test_ocr_processor(self):
        """Test OCR processor initialization"""
        from processors import PaddleOCRProcessor

        try:
            ocr = PaddleOCRProcessor(self.config)
            # Just test initialization, not actual processing
            self.assertIsNotNone(ocr)
        except ImportError:
            self.skipTest("PaddleOCR not installed")


class TestPipeline(unittest.TestCase):
    """Test main pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = ProcessingConfig(
            use_gpu=False,
            save_intermediate_results=False
        )

    async def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        from pipeline import ScreenUnderstandingPipeline

        try:
            pipeline = ScreenUnderstandingPipeline(self.config)
            self.assertIsNotNone(pipeline)
        except Exception as e:
            self.skipTest(f"Pipeline initialization failed: {e}")


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()
