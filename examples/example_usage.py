"""
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
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Screen Type: {result.screen_type.value}")
    print(f"Summary: {result.summary}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")

    # Save results
    output_path = Path("data/outputs/example_result.json")
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")


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

    print(f"\nProcessed {len(results)} images")
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
    print("\n" + "-"*60 + "\n")

    await process_batch()
    print("\n" + "-"*60 + "\n")

    await custom_processing()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
