#!/bin/bash

# Screen Understanding System - Setup Script
# This script sets up the conda environment and installs all dependencies

set -e  # Exit on error

echo "==========================================="
echo "Screen Understanding System - Setup Script"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_success "Conda is installed"

# Get CUDA version if available
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | sed 's/\.//')
    print_info "Detected CUDA version: ${CUDA_VERSION:0:2}.${CUDA_VERSION:2}"
    
    # Adjust CUDA toolkit version in environment.yml based on detected version
    if [[ ${CUDA_VERSION:0:2} == "12" ]]; then
        print_info "Using CUDA 12.x compatible packages"
        sed -i 's/pytorch-cuda=11.8/pytorch-cuda=12.1/g' environment.yml
        sed -i 's/cudatoolkit=11.8/cudatoolkit=12.1/g' environment.yml
    fi
else
    print_info "No CUDA detected, will use CPU-only packages"
    # Remove CUDA dependencies from environment.yml
    sed -i '/pytorch-cuda/d' environment.yml
    sed -i '/cudatoolkit/d' environment.yml
fi

# Create conda environment
print_info "Creating conda environment 'screen-understanding'..."
if conda env list | grep -q "screen-understanding"; then
    print_info "Environment already exists. Removing old environment..."
    conda env remove -n screen-understanding -y
fi

conda env create -f environment.yml
print_success "Conda environment created"

# Activate environment
print_info "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate screen-understanding

# Verify Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
print_success "Python version: $PYTHON_VERSION"

# Install additional packages that might need special handling
print_info "Installing additional packages..."

# Install PaddleOCR with proper dependencies
if [[ -n "$CUDA_VERSION" ]]; then
    print_info "Installing PaddlePaddle with GPU support..."
    pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    print_info "Installing PaddlePaddle with CPU support..."
    pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# Verify PyTorch installation
print_info "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
print_success "PyTorch installed successfully"

# Verify transformers installation
print_info "Verifying Transformers installation..."
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
print_success "Transformers installed successfully"

# Download models from Hugging Face (optional - can be done at runtime)
print_info "Setting up Hugging Face cache directory..."
export HF_HOME="./models/huggingface"
mkdir -p $HF_HOME

# Create project structure
print_info "Creating project structure..."
mkdir -p data/images
mkdir -p data/outputs
mkdir -p models/checkpoints
mkdir -p logs
mkdir -p tests
mkdir -p examples

# Create example test script
cat > test_installation.py << 'EOF'
"""Test script to verify installation"""
import sys
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModel
import paddleocr

def test_imports():
    """Test if all major imports work"""
    print("Testing imports...")
    
    modules = {
        "PyTorch": torch,
        "PIL": Image,
        "OpenCV": cv2,
        "NumPy": np,
        "Transformers": "transformers",
        "PaddleOCR": paddleocr
    }
    
    for name, module in modules.items():
        try:
            if isinstance(module, str):
                __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {name}: {e}")
            return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("ℹ CUDA is not available, will use CPU")
    
    return True

def test_basic_pipeline():
    """Test basic pipeline components"""
    print("\nTesting basic pipeline...")
    
    # Create a test image
    test_image = Image.new('RGB', (640, 480), color='white')
    print("✓ Created test image")
    
    # Convert to numpy
    img_array = np.array(test_image)
    print("✓ Converted to numpy array")
    
    # Basic OpenCV operation
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    print("✓ OpenCV color conversion works")
    
    return True

def main():
    print("="*50)
    print("Screen Understanding System - Installation Test")
    print("="*50)
    
    success = True
    
    success = test_imports() and success
    success = test_cuda() and success
    success = test_basic_pipeline() and success
    
    print("\n" + "="*50)
    if success:
        print("✅ All tests passed! System is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    print("="*50)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
EOF

print_success "Test script created"

# Run installation test
print_info "Running installation test..."
python test_installation.py

# Create README
cat > README.md << 'EOF'
# Screen Understanding System

AI-powered screen understanding system using OCR, OmniParser, and Vision-Language Models.

## Features

- **OCR Recognition**: PaddleOCR v4/v5 for text extraction
- **Element Detection**: OmniParser v2 for UI element detection
- **VLM Analysis**: ScreenAI and other VLMs for screen understanding
- **Entity Extraction**: Automatic extraction of dates, prices, emails, etc.
- **Action Detection**: Identify possible user actions on the screen

## Installation

1. Run the setup script:
```bash
bash setup.sh
```

2. Activate the environment:
```bash
conda activate screen-understanding
```

## Usage

```python
from pipeline import ScreenUnderstandingPipeline
from models import ProcessingConfig
from PIL import Image

# Configure pipeline
config = ProcessingConfig(
    use_gpu=True,
    vlm_model_name="google/screen-ai-v1.0"
)

# Initialize pipeline
pipeline = ScreenUnderstandingPipeline(config)

# Process image
image = Image.open("screenshot.png")
understanding = await pipeline.process(image)

# Access results
print(f"Screen type: {understanding.screen_type}")
print(f"Summary: {understanding.summary}")
for action in understanding.affordances:
    print(f"Action: {action.description}")
```

## Project Structure

```
.
├── models.py          # Data structures and types
├── processors.py      # OCR, OmniParser, and VLM processors
├── pipeline.py        # Main processing pipeline
├── environment.yml    # Conda environment configuration
├── pyproject.toml     # Poetry configuration (alternative)
├── setup.sh          # Setup script
├── data/             # Data directory
│   ├── images/       # Input images
│   └── outputs/      # Processing outputs
├── models/           # Model storage
│   ├── checkpoints/  # Model checkpoints
│   └── huggingface/  # HuggingFace cache
└── tests/            # Test files
```

## Models

The system uses models from Hugging Face. On first run, models will be automatically downloaded.

### Available Models:
- **OCR**: PaddleOCR v4/v5
- **Element Detection**: OmniParser (Microsoft)
- **VLM**: ScreenAI (Google), Qwen2-VL, InternVL 2.5

## License

MIT License
EOF

print_success "README created"

# Final message
echo ""
echo "==========================================="
print_success "Setup completed successfully!"
echo "==========================================="
echo ""
echo "To get started:"
echo "  1. Activate the environment: conda activate screen-understanding"
echo "  2. Test the installation: python test_installation.py"
echo "  3. Run the example: python pipeline.py"
echo ""
echo "For GPU support, make sure you have compatible NVIDIA drivers installed."
echo "Check the README.md for more information."