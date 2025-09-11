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
