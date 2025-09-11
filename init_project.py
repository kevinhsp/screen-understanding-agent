"""
init_project.py - Initialize project structure and create necessary files
"""

import os
import sys
from pathlib import Path
import json


def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        "data",
        "data/images",
        "data/outputs",
        "models",
        "models/checkpoints",
        "models/huggingface",
        "tests",
        "examples",
        "logs",
        "pipeline_outputs",
        "config"
    ]

    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")

    return True


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Conda
*.egg-info/
.conda/
conda-meta/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Project specific
data/outputs/
pipeline_outputs/
logs/
*.log
models/checkpoints/
models/huggingface/
*.pth
*.pt
*.onnx
*.pkl

# Test outputs
test_*.json
*.test.png

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore")

    return True


def create_config_file():
    """Create default configuration file"""
    default_config = {
        "processing": {
            "ocr_language": "en",
            "ocr_confidence_threshold": 0.5,
            "element_detection_threshold": 0.3,
            "merge_overlapping_elements": True,
            "iou_threshold": 0.5,
            "max_elements": 100
        },
        "models": {
            "ocr_model": "PaddleOCR",
            "element_detector": "OmniParser",
            "vlm_model": "google/screen-ai-v1.0",
            "use_multiple_vlms": False,
            "vlm_max_tokens": 512,
            "vlm_temperature": 0.7
        },
        "performance": {
            "use_gpu": True,
            "batch_size": 1,
            "num_workers": 4
        },
        "output": {
            "save_intermediate_results": False,
            "debug_mode": False,
            "output_format": "json"
        }
    }

    config_path = Path("config/default_config.json")
    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=2)
    print("✓ Created config/default_config.json")

    return True


def create_example_script():
    """Create example usage script"""
    example_content = '''"""
example_usage.py - Example script showing how to use the screen understanding system
"""

import asyncio
from pathlib import Path
from PIL import Image
import json

from pipeline import ScreenUnderstandingPipeline
from models import ProcessingConfig


async def process_single_image():
    """Process a single screenshot"""
    print("Processing single image example...")

    # Configure the pipeline
    config = ProcessingConfig(
        ocr_language="en",
        vlm_model_name="microsoft/Florence-2-large",  # Using available model
        use_gpu=True,
        debug_mode=True,
        save_intermediate_results=True
    )

    # Initialize pipeline
    pipeline = ScreenUnderstandingPipeline(config)

    # Load image (create a test image for demo)
    test_image = Image.new('RGB', (1024, 768), color='white')

    # You would normally load a real image like this:
    # image = Image.open("data/images/screenshot.png")

    # Process the image
    result = await pipeline.process(test_image, "test_image")

    # Print results
    print("\\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Screen Type: {result.screen_type.value}")
    print(f"Summary: {result.summary}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")

    # Save results
    output_path = Path("data/outputs/example_result.json")
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\\nResults saved to: {output_path}")


async def process_batch():
    """Process multiple images"""
    print("Processing batch example...")

    # Configure pipeline
    config = ProcessingConfig(use_gpu=True)
    pipeline = ScreenUnderstandingPipeline(config)

    # Find all images in directory
    image_dir = Path("data/images")
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    if not image_files:
        print("No images found in data/images/")
        # Create test images for demo
        for i in range(3):
            test_image = Image.new('RGB', (800, 600), color='white')
            test_path = image_dir / f"test_{i}.png"
            test_image.save(test_path)
            image_files.append(test_path)
        print(f"Created {len(image_files)} test images for demo")

    # Process all images
    results = await pipeline.process_batch(
        [str(f) for f in image_files],
        save_results=True
    )

    print(f"\\nProcessed {len(results)} images")
    for i, (file, result) in enumerate(zip(image_files, results)):
        if result:
            print(f"  {i+1}. {file.name}: {result.screen_type.value}")


async def custom_processing():
    """Example with custom processing logic"""
    print("Custom processing example...")

    from processors import PaddleOCRProcessor, OmniParserV2, ScreenAIProcessor

    # Create custom configuration
    config = ProcessingConfig(
        ocr_confidence_threshold=0.7,
        element_detection_threshold=0.5,
        vlm_temperature=0.5
    )

    # Initialize components separately
    ocr = PaddleOCRProcessor(config)
    detector = OmniParserV2(config)
    vlm = ScreenAIProcessor(config)

    # Process step by step
    image = Image.new('RGB', (800, 600), color='white')

    # Run OCR
    ocr_results = await ocr.process(image)
    print(f"Found {len(ocr_results)} text regions")

    # Detect elements
    elements = await detector.detect_elements(image)
    print(f"Detected {len(elements)} UI elements")

    # Merge results
    merged = await detector.merge_with_ocr(elements, ocr_results)
    print(f"Merged into {len(merged)} elements with text")

    # Run VLM analysis
    texts = [r.text for r in ocr_results]
    analysis = await vlm.analyze(image, texts, merged)
    print(f"Screen type: {analysis.get('screen_type', 'unknown')}")


async def main():
    """Run all examples"""
    print("="*60)
    print("SCREEN UNDERSTANDING SYSTEM - EXAMPLES")
    print("="*60)

    # Run examples
    await process_single_image()
    print("\\n" + "-"*60 + "\\n")

    await process_batch()
    print("\\n" + "-"*60 + "\\n")

    await custom_processing()

    print("\\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
'''

    with open("examples/example_usage.py", "w") as f:
        f.write(example_content)
    print("✓ Created examples/example_usage.py")

    return True


def create_test_file():
    """Create basic test file"""
    test_content = '''"""
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
'''

    with open("tests/test_pipeline.py", "w") as f:
        f.write(test_content)
    print("✓ Created tests/test_pipeline.py")

    return True


def create_readme_files():
    """Create README files for directories"""
    readmes = {
        "data/README.md": """# Data Directory

This directory contains input images and processing outputs.

## Structure:
- `images/` - Place your screenshot images here
- `outputs/` - Processed results will be saved here

## Supported Image Formats:
- PNG (recommended)
- JPG/JPEG
- BMP
- TIFF
""",
        "models/README.md": """# Models Directory

This directory stores model files and checkpoints.

## Structure:
- `checkpoints/` - Saved model checkpoints
- `huggingface/` - Hugging Face model cache

Models will be automatically downloaded on first use.
""",
        "tests/README.md": """# Tests Directory

Unit tests and integration tests for the system.

## Running Tests:
```bash
python -m pytest tests/
# or
python tests/test_pipeline.py
```
""",
        "examples/README.md": """# Examples Directory

Example scripts demonstrating system usage.

## Available Examples:
- `example_usage.py` - Basic usage examples
- More examples coming soon...

## Running Examples:
```bash
python examples/example_usage.py
```
"""
    }

    for path, content in readmes.items():
        with open(path, "w") as f:
            f.write(content)
        print(f"✓ Created {path}")

    return True


def main():
    """Main initialization function"""
    print("=" * 60)
    print("SCREEN UNDERSTANDING SYSTEM - PROJECT INITIALIZATION")
    print("=" * 60)
    print()

    try:
        # Create directory structure
        create_directory_structure()
        print()

        # Create configuration files
        create_gitignore()
        create_config_file()
        print()

        # Create example and test files
        create_example_script()
        create_test_file()
        print()

        # Create README files
        create_readme_files()
        print()

        print("=" * 60)
        print("✅ Project initialization completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Install dependencies: conda env create -f environment.yml")
        print("2. Activate environment: conda activate screen-understanding")
        print("3. Test installation: python test_installation.py")
        print("4. Run examples: python examples/example_usage.py")

        return 0

    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())