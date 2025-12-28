# EfficientSAM3 ARM Inference Package

This package provides ARM-optimized inference for EfficientSAM3, allowing you to run the model on Apple Silicon (M1/M2/M3) and other ARM devices without requiring CUDA or Triton dependencies.

## Features

✅ **No CUDA/Triton Required** - Pure PyTorch implementation  
✅ **Apple Silicon Optimized** - Supports MPS (Metal Performance Shaders)  
✅ **CPU Fallback** - Works on any ARM device  
✅ **Drop-in Replacement** - Compatible API with the original processor  
✅ **Batch Processing** - Process multiple images efficiently  
✅ **Visualization Tools** - Built-in result visualization

## Installation

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install PyTorch (MPS support for Apple Silicon)
pip install torch torchvision

# Install other dependencies
pip install pillow numpy matplotlib
```

### Project Structure

```
efficientsam3/
├── efficientsam3_arm/          # ARM inference package
│   ├── __init__.py
│   ├── model_builder_arm.py    # Model loading with Triton patches
│   ├── processor_arm.py        # ARM-optimized processor
│   └── utils_arm.py            # Utility functions
├── sam3/                       # Original sam3 package
├── weights/                    # Model checkpoints
│   └── efficient_sam3_repvit_s.pt
└── test_arm_setup.py           # Setup verification script
```

## Quick Start

### 1. Verify Setup

```bash
python test_arm_setup.py --checkpoint weights/efficient_sam3_repvit_s.pt
```

This will test:
- ✓ PyTorch installation and device availability
- ✓ Model loading
- ✓ Processor creation  
- ✓ Image processing
- ✓ Text-based grounding

### 2. Basic Usage

```python
from efficientsam3_arm import build_sam3_model_arm, Sam3ProcessorARM
from PIL import Image

# Load model
model = build_sam3_model_arm(
    checkpoint_path="weights/efficient_sam3_repvit_s.pt",
    device="mps"  # or "cpu"
)

# Create processor
processor = Sam3ProcessorARM(model, device="mps")

# Process image
image = Image.open("image.jpg")
state = processor.set_image(image)
state = processor.set_text_prompt("person. car. dog.", state)

# Get results
masks = state["masks"]      # Binary masks [N, H, W]
boxes = state["boxes"]      # Bounding boxes [N, 4] in xyxy format
scores = state["scores"]    # Confidence scores [N]
```

### 3. Command-Line Interface

```bash
# Single image
python efficientsam3_arm_demo.py \
    --checkpoint weights/efficient_sam3_repvit_s.pt \
    --image path/to/image.jpg \
    --text_prompt "person. car." \
    --visualize \
    --output results/

# Batch processing
python efficientsam3_arm_demo.py \
    --checkpoint weights/efficient_sam3_repvit_s.pt \
    --image_dir path/to/images/ \
    --text_prompt "person" \
    --output results/ \
    --visualize
```

## API Reference

### `build_sam3_model_arm()`

Load EfficientSAM3 model for ARM inference.

```python
model = build_sam3_model_arm(
    checkpoint_path: str,           # Path to .pt checkpoint
    device: str = "cpu",            # "cpu", "mps", or "cuda"
    backbone_type: str = "repvit",  # Backbone architecture
    model_name: str = "s",          # Model size variant
    bpe_path: str = None,           # BPE tokenizer path (auto-detected)
    enable_inst_interactivity: bool = False,  # SAM2 mode
    verbose: bool = True,           # Print loading info
)
```

### `Sam3ProcessorARM`

ARM-optimized processor for inference.

```python
processor = Sam3ProcessorARM(
    model,                          # EfficientSAM3 model
    resolution: int = 1008,         # Input resolution
    device: str = "cpu",            # Device
    confidence_threshold: float = 0.5,  # Detection threshold
    use_fp16: bool = False,         # FP16 precision (not for MPS)
    optimize_for_inference: bool = True,  # Apply optimizations
)
```

#### Methods

**`set_image(image, state=None)`**
```python
state = processor.set_image(image)
# image: PIL.Image, torch.Tensor, or np.ndarray
# Returns: state dict with image encoding
```

**`set_text_prompt(prompt, state)`**
```python
state = processor.set_text_prompt("person. car.", state)
# prompt: Text description (separate objects with '. ')
# Returns: state dict with detection results
```

**`add_geometric_prompt(box, label, state)`**
```python
state = processor.add_geometric_prompt([0.5, 0.5, 0.3, 0.4], True, state)
# box: [center_x, center_y, width, height] normalized to [0, 1]
# label: True for positive, False for negative
# Returns: updated state with results
```

**`set_confidence_threshold(threshold, state=None)`**
```python
state = processor.set_confidence_threshold(0.7, state)
# threshold: New confidence threshold (0.0 to 1.0)
```

**`reset_all_prompts(state)`**
```python
processor.reset_all_prompts(state)
# Removes all prompts and results from state
```

### Utility Functions

**`get_optimal_device()`**
```python
from efficientsam3_arm import get_optimal_device
device = get_optimal_device()  # Returns "mps" or "cpu"
```

**`load_image(image_path)`**
```python
from efficientsam3_arm import load_image
image = load_image("path/to/image.jpg")
```

**`visualize_results()`**
```python
from efficientsam3_arm import visualize_results
fig = visualize_results(
    image,
    masks=masks,
    boxes=boxes,
    scores=scores,
    save_path="result.png",
)
```

**`save_results()`**
```python
from efficientsam3_arm import save_results
save_results(masks, boxes, scores, output_dir="results/", prefix="result")
```

## Examples

### Example 1: Object Detection with Text Prompt

```python
from efficientsam3_arm import build_sam3_model_arm, Sam3ProcessorARM, load_image
import matplotlib.pyplot as plt

# Setup
model = build_sam3_model_arm("weights/efficient_sam3_repvit_s.pt", device="mps")
processor = Sam3ProcessorARM(model, device="mps", confidence_threshold=0.5)

# Load and process
image = load_image("images/persons.jpg")
state = processor.set_image(image)
state = processor.set_text_prompt("person. face.", state)

# Display results
print(f"Found {len(state['scores'])} objects")
for i, score in enumerate(state['scores']):
    print(f"  Object {i+1}: {score:.3f} confidence")
```

### Example 2: Interactive Box Prompts

```python
# Set image
state = processor.set_image(image)

# Add positive box (normalized coordinates)
box = [0.5, 0.5, 0.3, 0.4]  # center_x, center_y, width, height
state = processor.add_geometric_prompt(box, label=True, state)

# Add negative box to exclude region
neg_box = [0.2, 0.2, 0.1, 0.1]
state = processor.add_geometric_prompt(neg_box, label=False, state)

# Get refined results
masks = state["masks"]
```

### Example 3: Batch Processing

```python
from pathlib import Path

# Load model once
model = build_sam3_model_arm("weights/efficient_sam3_repvit_s.pt", device="mps")
processor = Sam3ProcessorARM(model, device="mps")

# Process multiple images
image_dir = Path("images/")
for image_path in image_dir.glob("*.jpg"):
    print(f"Processing {image_path.name}")
    
    image = load_image(image_path)
    state = processor.set_image(image)
    state = processor.set_text_prompt("person. car.", state)
    
    print(f"  Found {len(state['scores'])} objects")
```

### Example 4: Custom Visualization

```python
from efficientsam3_arm import visualize_results

# Create custom visualization
fig = visualize_results(
    image,
    masks=state["masks"],
    boxes=state["boxes"],
    scores=state["scores"],
    labels=["person", "car"],  # Optional labels
    score_threshold=0.7,       # Filter low confidence
    show_scores=True,
    save_path="custom_result.png",
    dpi=150,
)
plt.show()
```

## Performance Tips

### 1. Device Selection

```python
# Apple Silicon - Use MPS for best performance
device = "mps"  # 2-3x faster than CPU

# Intel Mac or other ARM - Use CPU
device = "cpu"
```

### 2. Batch Processing

```python
# Process images in batch for better throughput
processor.set_image_batch([image1, image2, image3])
```

### 3. Adjust Resolution

```python
# Lower resolution = faster inference
processor = Sam3ProcessorARM(model, resolution=512, device="mps")
```

### 4. Confidence Threshold

```python
# Higher threshold = fewer but more confident detections
processor.set_confidence_threshold(0.7)
```

## Troubleshooting

### Issue: "No module named 'triton'"

**Solution:** The `efficientsam3_arm` package automatically patches Triton dependencies. Make sure you're using the ARM-specific import:

```python
# ✓ Correct
from efficientsam3_arm import build_sam3_model_arm

# ✗ Incorrect
from sam3.model_builder import build_efficientsam3_image_model
```

### Issue: MPS errors or crashes

**Solution:** Fall back to CPU:

```python
model = build_sam3_model_arm(checkpoint_path, device="cpu")
processor = Sam3ProcessorARM(model, device="cpu")
```

### Issue: Out of memory

**Solutions:**
1. Lower resolution: `resolution=512` instead of `1008`
2. Process images one at a time instead of batching
3. Use CPU instead of MPS

### Issue: Slow inference

**Solutions:**
1. Use MPS on Apple Silicon: `device="mps"`
2. Ensure model is in eval mode (done automatically)
3. Process multiple images to amortize loading overhead

## Model Variants

| Model | Backbone | Size | Parameters | Speed | Accuracy |
|-------|----------|------|------------|-------|----------|
| `efficient_sam3_repvit_s` | RepViT | S | ~20M | Fastest | Good |
| `efficient_sam3_repvit_m` | RepViT | M | ~40M | Fast | Better |
| `efficient_sam3_repvit_l` | RepViT | L | ~60M | Medium | Best |

## Technical Details

### Triton Patching

The package automatically patches Triton-dependent modules:

1. **RMSNorm**: Replaced with PyTorch implementation
2. **NMS**: Falls back to torchvision NMS
3. **Focal Loss**: Training-only, not needed for inference

### Device Compatibility

| Device | Support | Performance |
|--------|---------|-------------|
| Apple M1/M2/M3 (MPS) | ✅ Full | Excellent |
| Apple Intel (CPU) | ✅ Full | Good |
| Linux ARM (CPU) | ✅ Full | Good |
| CUDA (if available) | ✅ Full | Excellent |

## Citation

If you use EfficientSAM3 in your research, please cite:

```bibtex
@article{efficientsam3,
  title={EfficientSAM 3.0: Towards Efficient Segment Anything Model},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This package follows the same license as the original SAM3 project.

## Support

For issues specific to ARM inference:
1. Run `test_arm_setup.py` to verify your setup
2. Check the troubleshooting section above
3. Ensure you're using the latest version of PyTorch with MPS support

For general EfficientSAM3 questions, refer to the main project documentation.
