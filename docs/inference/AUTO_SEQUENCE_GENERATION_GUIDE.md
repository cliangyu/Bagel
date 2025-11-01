# Auto Sequence Generation Guide

This guide explains how to use the new `generate_auto_sequence()` method for automatic interleaved text-image generation.

---

## Overview

The `generate_auto_sequence()` method enables **true automatic interleaved inference** by:
- ✅ Automatically alternating between text and image generation
- ✅ Controlling sequence length via `num_images` parameter
- ✅ No manual modality switching required
- ✅ Supporting both pure generation and editing workflows

**Pattern:** text → image → text → image → text → image ...

---

## Quick Start

### Use Case 1: Pure Text-to-Image Generation

```python
from inferencer import InterleaveInferencer
from scripts.auto_interleaved_demo import AutoInterleaveInferencer

# Initialize
auto_inferencer = AutoInterleaveInferencer(base_inferencer)

# Add initial prompt
auto_inferencer.add_to_context("A magical forest with glowing mushrooms")

# Generate auto sequence
outputs = auto_inferencer.generate_auto_sequence(
    num_images=3,       # Generate 3 images
    think=True,         # Use chain-of-thought
    cfg_text_scale=4.0,
    num_timesteps=50
)

# Returns: [thinking_1, image_1, refinement_1, thinking_2, image_2, refinement_2, thinking_3, image_3]
```

### Use Case 2: Image Editing

```python
from PIL import Image

# Initialize
auto_inferencer = AutoInterleaveInferencer(base_inferencer)

# Add image + editing instruction
original_image = Image.open("photo.jpg")
auto_inferencer.add_to_context(original_image)
auto_inferencer.add_to_context("Make the lighting warmer and add sunset colors")

# Generate editing sequence
outputs = auto_inferencer.generate_auto_sequence(
    num_images=2,        # Generate 2 edited versions
    think=True,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0    # Higher for editing
)

# Returns: [thinking_1, edited_1, refinement_1, thinking_2, edited_2]
```

---

## API Reference

### `generate_auto_sequence()`

```python
def generate_auto_sequence(
    self,
    num_images: int = 1,
    think: bool = False,
    max_think_token_n: int = 500,
    do_sample: bool = True,
    text_temperature: float = 0.7,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 1.5,
    num_timesteps: int = 50,
    enable_taylorseer: bool = False,
    **kwargs
) -> List[Union[str, Image.Image]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_images` | `int` | `1` | **Number of images to generate** (controls sequence length) |
| `think` | `bool` | `False` | Whether to use chain-of-thought before each image |
| `max_think_token_n` | `int` | `500` | Max tokens for thinking/refinement text |
| `do_sample` | `bool` | `True` | Whether to sample during text generation |
| `text_temperature` | `float` | `0.7` | Temperature for text generation |
| `cfg_text_scale` | `float` | `4.0` | Text CFG scale (1.0 = no guidance, 4.0 = strong) |
| `cfg_img_scale` | `float` | `1.5` | Image CFG scale (auto-adjusted: 1.0 for generation, 1.5+ for editing) |
| `num_timesteps` | `int` | `50` | Number of denoising steps |
| `enable_taylorseer` | `bool` | `False` | Use TaylorSeer acceleration |

#### Returns

`List[Union[str, Image.Image]]` - Interleaved list of text and images

---

## How It Works

### Execution Flow

```
Initial Context: ["user text prompt"]
                        ↓
┌───────────────────────────────────────┐
│ Iteration 1                            │
├───────────────────────────────────────┤
│ Last item: str → Generate image        │
│   1. think=True? → Generate thinking   │
│      Context: [prompt, thinking]       │
│   2. Generate image                    │
│      Context: [prompt, thinking, img1] │
│   3. images < num_images? Yes          │
│      → Generate refinement text        │
│      Context: [..., img1, refinement]  │
└───────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────┐
│ Iteration 2                            │
├───────────────────────────────────────┤
│ Last item: str → Generate image        │
│   1. Generate thinking                 │
│   2. Generate image                    │
│   3. Generate refinement (if needed)   │
└───────────────────────────────────────┘
                        ↓
                      ...
```

### Automatic Mode Detection

The method automatically detects generation vs editing mode:

```python
# Detect if input image exists (not in outputs)
has_input_image = any(
    isinstance(item, Image.Image) and item not in outputs
    for item in self.context_list
)

if has_input_image:
    mode = "editing"
    cfg_img_scale = user_specified_value  # e.g., 2.0
else:
    mode = "generation"
    cfg_img_scale = 1.0  # No image guidance
```

---

## Examples

### Example 1: Single Image Generation

```python
auto_inferencer.add_to_context("A sunset over the ocean")
outputs = auto_inferencer.generate_auto_sequence(num_images=1, think=False)

# Returns: [image]
# Context after: ["A sunset over the ocean", image]
```

### Example 2: Multiple Images with Thinking

```python
auto_inferencer.add_to_context("A cyberpunk city at night")
outputs = auto_inferencer.generate_auto_sequence(num_images=2, think=True)

# Returns: [thinking_1, image_1, refinement_1, thinking_2, image_2]
# Context after: [
#   "A cyberpunk city at night",
#   thinking_1,
#   image_1,
#   refinement_1,
#   thinking_2,
#   image_2
# ]
```

### Example 3: Progressive Refinement

```python
auto_inferencer.add_to_context("A portrait of a wise old wizard")
outputs = auto_inferencer.generate_auto_sequence(
    num_images=3,
    think=True,
    max_think_token_n=300,
    cfg_text_scale=5.0,  # Stronger text adherence
    num_timesteps=100    # More steps for quality
)

# Returns:
# [
#   "Let me create a detailed wizard portrait...",  # thinking_1
#   <image_1>,
#   "Now let's enhance the details and add more mystical elements...",  # refinement_1
#   "I'll adjust the lighting and add magical effects...",  # thinking_2
#   <image_2>,
#   "Let's refine the final details and perfect the composition...",  # refinement_2
#   "I'll fine-tune the colors and add finishing touches...",  # thinking_3
#   <image_3>
# ]
```

### Example 4: Image Editing Workflow

```python
# Start with original image
photo = Image.open("portrait.jpg")
auto_inferencer.add_to_context(photo)
auto_inferencer.add_to_context("Make the background blurry and add cinematic lighting")

outputs = auto_inferencer.generate_auto_sequence(
    num_images=2,
    think=True,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0  # Preserve structure
)

# Sequence:
# 1. Thinking: "I'll apply bokeh effect and add dramatic lighting..."
# 2. Image 1: Edited with bokeh + lighting
# 3. Refinement: "Let's enhance the effect further..."
# 4. Thinking: "I'll intensify the cinematic look..."
# 5. Image 2: Further refined version

# Save final result
final_image = outputs[-1]
final_image.save("portrait_edited.png")
```

### Example 5: Multi-Image Editing

```python
# Edit sequence building on previous edits
photo = Image.open("landscape.jpg")
auto_inferencer.add_to_context(photo)
auto_inferencer.add_to_context("Add vibrant sunset colors")

outputs = auto_inferencer.generate_auto_sequence(num_images=3, think=True)

# Each iteration builds on the previous:
# - Image 1: Sunset colors added
# - Image 2: Enhanced based on Image 1
# - Image 3: Final refinement based on Image 2

# The model sees the entire history:
# Context: [original, instruction, think1, img1, refine1, think2, img2, refine2, think3, img3]
```

---

## Comparison: Old vs New

### Before: Manual Modality Switching

```python
# User must manually specify each generation type
auto_inferencer.add_to_context("Draw a cat")
img1 = auto_inferencer.generate_image()  # Manual

auto_inferencer.add_to_context("Make it orange")
img2 = auto_inferencer.generate_image()  # Manual

auto_inferencer.add_to_context("Add stripes")
img3 = auto_inferencer.generate_image()  # Manual

# User must add refinement prompts manually
```

### After: Automatic Sequence

```python
# Single call, automatic sequence
auto_inferencer.add_to_context("Draw a cat")
outputs = auto_inferencer.generate_auto_sequence(num_images=3, think=True)

# Automatically generates:
# [thinking_1, cat_img, refinement_1, thinking_2, orange_cat, refinement_2, thinking_3, striped_cat]

# All refinement text generated by the model automatically!
```

---

## Running the Demo

### Command Line

```bash
# Run auto generation demo
python scripts/auto_interleaved_demo.py \
    --model_path /path/to/BAGEL-7B-MoT \
    --demo auto_generation \
    --gpu_ids 0,1,2,3

# Run editing demo (example only)
python scripts/auto_interleaved_demo.py \
    --model_path /path/to/BAGEL-7B-MoT \
    --demo auto_editing

# Run original manual demo
python scripts/auto_interleaved_demo.py \
    --model_path /path/to/BAGEL-7B-MoT \
    --demo manual

# Run all demos
python scripts/auto_interleaved_demo.py \
    --model_path /path/to/BAGEL-7B-MoT \
    --demo all
```

### Python API

```python
from inferencer import InterleaveInferencer
from scripts.auto_interleaved_demo import AutoInterleaveInferencer

# Load model (see INFERENCE_GUIDE.md for details)
model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(model_path)

# Create base inferencer
base_inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids
)

# Create auto inferencer
auto_inferencer = AutoInterleaveInferencer(base_inferencer)

# Use it!
auto_inferencer.add_to_context("Your prompt here")
outputs = auto_inferencer.generate_auto_sequence(num_images=3, think=True)
```

---

## Best Practices

### 1. Choose `num_images` Based on Use Case

| Use Case | Recommended `num_images` |
|----------|-------------------------|
| Quick generation | 1 |
| Progressive refinement | 2-3 |
| Extensive exploration | 3-5 |
| Editing workflow | 2-3 |

### 2. Use `think=True` for Better Quality

Chain-of-thought improves:
- Planning and coherence
- Progressive refinement
- Understanding of complex prompts

```python
# Better results
outputs = auto_inferencer.generate_auto_sequence(num_images=3, think=True)

# Faster but less refined
outputs = auto_inferencer.generate_auto_sequence(num_images=3, think=False)
```

### 3. Adjust CFG Scales

| Task | `cfg_text_scale` | `cfg_img_scale` |
|------|-----------------|----------------|
| Pure generation | 4.0-5.0 | 1.0 (auto) |
| Precise editing | 4.0 | 2.0-2.5 |
| Creative editing | 3.0 | 1.2-1.5 |
| Diverse outputs | 2.0-3.0 | 1.0 |

### 4. Save Intermediate Results

```python
outputs = auto_inferencer.generate_auto_sequence(num_images=3)

# Save all generated images
image_count = 0
for i, output in enumerate(outputs):
    if isinstance(output, Image.Image):
        output.save(f"output_{image_count:02d}.png")
        image_count += 1
```

---

## Limitations

1. **Linear sequence only**: Always follows text → image → text → image pattern
2. **No backtracking**: Cannot regenerate earlier images in sequence
3. **Context grows unbounded**: Long sequences may hit memory limits
4. **Automatic refinement text**: Model generates refinement prompts (training-dependent quality)

---

## Tips & Tricks

### Clear Context Between Sessions

```python
auto_inferencer.clear_context()
auto_inferencer.add_to_context("New prompt")
outputs = auto_inferencer.generate_auto_sequence(num_images=2)
```

### Extract Specific Outputs

```python
outputs = auto_inferencer.generate_auto_sequence(num_images=3, think=True)

# Get all images
images = [x for x in outputs if isinstance(x, Image.Image)]

# Get all text
texts = [x for x in outputs if isinstance(x, str)]

# Get final image
final_image = images[-1]
```

### Combine with Manual Additions

```python
# Start auto sequence
auto_inferencer.add_to_context("A medieval castle")
outputs = auto_inferencer.generate_auto_sequence(num_images=2, think=True)

# Add manual instruction
auto_inferencer.add_to_context("Now add a dragon flying overhead")

# Continue auto sequence
more_outputs = auto_inferencer.generate_auto_sequence(num_images=1, think=True)
```

---

## Summary

The `generate_auto_sequence()` method provides **true automatic interleaved inference** by:

✅ **Eliminating manual modality switching**
✅ **Automatically alternating text and images**
✅ **Controlling sequence length via `num_images`**
✅ **Supporting both generation and editing**
✅ **Generating refinement text automatically**

This is the missing piece that makes BAGEL's interleaved inference truly automatic!

---

## See Also

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - General inference guide
- [SPECIAL_TOKENS_AND_EXECUTION_PATHS.md](SPECIAL_TOKENS_AND_EXECUTION_PATHS.md) - Deep dive into execution
- [AUTO_INTERLEAVED_ANALYSIS.md](AUTO_INTERLEAVED_ANALYSIS.md) - Analysis of the wrapper
- [INFERENCE_DECISION_TREE.md](INFERENCE_DECISION_TREE.md) - When image generation is triggered
