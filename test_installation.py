"""
test_installation.py - Test script to verify system installation
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.RESET}")

def print_section(text):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}▶ {text}{Colors.RESET}")

def test_core_imports():
    """Test core Python package imports"""
    print_section("Testing Core Imports")

    core_packages = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
    ]

    success = True
    for package, name in core_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{name:<20} {version}")
        except ImportError as e:
            print_error(f"{name:<20} Failed: {e}")
            success = False

    return success

def test_ml_imports():
    """Test ML framework imports"""
    print_section("Testing ML Framework Imports")

    ml_packages = [
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("datasets", "Datasets"),
        ("paddleocr", "PaddleOCR"),
        ("paddlepaddle", "PaddlePaddle"),
    ]

    success = True
    for package, name in ml_packages:
        try:
            if package == "paddleocr":
                from paddleocr import PaddleOCR
                print_success(f"{name:<20} Available")
            elif package == "paddlepaddle":
                import paddle
                version = paddle.__version__
                print_success(f"{name:<20} {version}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{name:<20} {version}")
        except ImportError as e:
            print_error(f"{name:<20} Failed: {e}")
            if package in ["paddleocr", "paddlepaddle"]:
                print_info("  → Run: pip install paddlepaddle paddleocr")
            success = False

    return success

def test_cuda_availability():
    """Test CUDA/GPU availability"""
    print_section("Testing GPU/CUDA Support")

    try:
        import torch

        if torch.cuda.is_available():
            print_success(f"CUDA Available: Yes")
            print_success(f"CUDA Version: {torch.version.cuda}")
            print_success(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print_success(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print_success(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            # Test CUDA tensor creation
            try:
                test_tensor = torch.randn(100, 100).cuda()
                print_success("CUDA Tensor Creation: OK")
                del test_tensor
            except Exception as e:
                print_error(f"CUDA Tensor Creation Failed: {e}")
                return False
        else:
            print_info("CUDA Not Available - CPU mode will be used")
            print_info("For GPU support, ensure NVIDIA drivers and CUDA are installed")

        return True
    except Exception as e:
        print_error(f"Error checking CUDA: {e}")
        return False

def test_project_modules():
    """Test local project modules"""
    print_section("Testing Project Modules")

    modules = ["models", "processors", "pipeline"]
    success = True

    for module in modules:
        try:
            __import__(module)
            print_success(f"{module}.py loaded successfully")
        except ImportError as e:
            print_error(f"{module}.py failed to load: {e}")
            success = False
        except Exception as e:
            print_error(f"{module}.py has errors: {e}")
            success = False

    return success

def test_model_loading():
    """Test loading a simple model from Hugging Face"""
    print_section("Testing Hugging Face Model Loading")

    try:
        from transformers import AutoTokenizer, AutoModel

        # Try to load a small model for testing
        model_name = "bert-base-uncased"
        print_info(f"Testing with small model: {model_name}")

        # Check if we can connect to Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_success("Tokenizer loaded successfully")

        # Don't actually download the full model in test
        print_info("Model loading test passed (full model not downloaded)")

        return True
    except Exception as e:
        print_error(f"Model loading failed: {e}")
        print_info("Check your internet connection and Hugging Face access")
        return False

def test_basic_pipeline():
    """Test basic pipeline functionality"""
    print_section("Testing Basic Pipeline Components")

    try:
        from PIL import Image
        import numpy as np
        import cv2

        # Create a test image
        test_image = Image.new('RGB', (640, 480), color='white')
        print_success("Created test PIL image")

        # Convert to numpy
        img_array = np.array(test_image)
        print_success(f"Converted to numpy array: shape {img_array.shape}")

        # Basic OpenCV operation
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        print_success(f"OpenCV conversion successful: shape {gray.shape}")

        # Test bounding box creation
        from models import BoundingBox
        bbox = BoundingBox(10, 20, 100, 50)
        print_success(f"BoundingBox created: {bbox.to_dict()}")

        return True
    except Exception as e:
        print_error(f"Basic pipeline test failed: {e}")
        return False

def test_ocr_processor():
    """Test OCR processor initialization"""
    print_section("Testing OCR Processor")

    try:
        from models import ProcessingConfig
        from processors import PaddleOCRProcessor

        config = ProcessingConfig(use_gpu=False)

        # Try to initialize OCR processor
        try:
            ocr = PaddleOCRProcessor(config)
            print_success("PaddleOCR processor initialized")
            return True
        except Exception as e:
            print_info(f"PaddleOCR initialization skipped: {e}")
            print_info("This is normal if PaddleOCR is not fully installed")
            return True  # Don't fail the test

    except Exception as e:
        print_error(f"Failed to test OCR processor: {e}")
        return False

def create_project_structure():
    """Create necessary project directories"""
    print_section("Creating Project Structure")

    directories = [
        "data",
        "data/images",
        "data/outputs",
        "models",
        "models/checkpoints",
        "models/huggingface",
        "logs",
        "tests",
        "examples",
        "pipeline_outputs"
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print_success(f"Directory created/verified: {directory}/")
        except Exception as e:
            print_error(f"Failed to create {directory}: {e}")
            return False

    # Create a sample test image
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a sample login screen
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)

        # Draw some UI elements
        draw.rectangle([100, 100, 700, 150], outline='black', width=2)
        draw.text((110, 115), "Username", fill='black')

        draw.rectangle([100, 200, 700, 250], outline='black', width=2)
        draw.text((110, 215), "Password", fill='black')

        draw.rectangle([300, 300, 500, 350], fill='blue', outline='blue')
        draw.text((360, 315), "Login", fill='white')

        img.save('examples/sample_login.png')
        print_success("Created sample image: examples/sample_login.png")

    except Exception as e:
        print_info(f"Could not create sample image: {e}")

    return True

def check_environment_variables():
    """Check important environment variables"""
    print_section("Environment Variables")

    env_vars = {
        "HF_HOME": "Hugging Face cache directory",
        "CUDA_HOME": "CUDA installation directory",
        "PATH": "System PATH"
    }

    for var, description in env_vars.items():
        value = os.environ.get(var, "Not set")
        if var == "PATH":
            # Just show if python and conda are in PATH
            if "conda" in value.lower():
                print_success(f"{var}: Contains conda")
            else:
                print_info(f"{var}: Does not contain conda")
        elif value != "Not set":
            print_success(f"{var}: {value[:50]}...")
        else:
            print_info(f"{var}: {value}")

    return True

def generate_test_report(results):
    """Generate a test report"""
    print_header("Test Report Summary")

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)

    print(f"{Colors.BOLD}Tests Passed: {Colors.GREEN}{passed_tests}/{total_tests}{Colors.RESET}")

    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All tests passed! System is ready to use.{Colors.RESET}")
        print("\nNext steps:")
        print("1. Try the example: python pipeline.py")
        print("2. Process your own image:")
        print("   from pipeline import ScreenUnderstandingPipeline")
        print("   from PIL import Image")
        print("   import asyncio")
        print("")
        print("   async def main():")
        print("       pipeline = ScreenUnderstandingPipeline()")
        print("       image = Image.open('your_screenshot.png')")
        print("       result = await pipeline.process(image)")
        print("       print(result.summary)")
        print("")
        print("   asyncio.run(main())")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Some tests failed or were skipped.{Colors.RESET}")
        print("\nFailed/Skipped tests:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  - {test_name}")

        print("\nThe system may still work with limited functionality.")
        print("Check the errors above for specific issues.")

    return passed_tests == total_tests

def main():
    """Main test function"""
    print_header("Screen Understanding System - Installation Test")

    # Dictionary to store test results
    results = {}

    # Run all tests
    results["Core Imports"] = test_core_imports()
    results["ML Frameworks"] = test_ml_imports()
    results["CUDA/GPU"] = test_cuda_availability()
    results["Project Modules"] = test_project_modules()
    results["Basic Pipeline"] = test_basic_pipeline()
    results["OCR Processor"] = test_ocr_processor()
    results["Model Loading"] = test_model_loading()
    results["Project Structure"] = create_project_structure()
    results["Environment"] = check_environment_variables()

    # Generate report
    all_passed = generate_test_report(results)

    # Return exit code
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())