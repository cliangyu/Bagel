# Auto-Interleaved Text and Image Generation

This document describes the implementation of automatic interleaved text and image generation for the Bagel model, as requested in [issue #99](https://github.com/ByteDance-Seed/Bagel/issues/99).

## Overview

The auto-interleaved generation feature allows Bagel to automatically switch between text and image generation based on special tokens in the output stream. This eliminates the need for manual switching and enables more natural multimodal content creation.

## Key Features

- **Automatic Mode Switching**: Detects `<|vision_start|>` tokens in generated text and automatically switches to image generation
- **Seamless Integration**: Extends the existing `InterleaveInferencer` class without breaking existing functionality
- **Error Handling**: Robust error handling with logging and graceful degradation
- **Flexible Configuration**: Supports all existing generation parameters for both text and image generation

## Architecture

### Core Components

1. **AutoInterleavedInferencer**: Main class that extends `InterleaveInferencer`
   - Monitors token generation for special vision tokens
   - Manages context switching between text and image generation
   - Handles the generation loop with proper state management

2. **Token Detection**: 
   - `vision_start_token_id` (151652): Triggers image generation
   - `vision_end_token_id` (151653): Marks end of image content
   - `eos_token_id`: Ends the generation sequence

3. **Context Management**:
   - Maintains separate contexts for text and image CFG (Classifier-Free Guidance)
   - Updates contexts after each generation step
   - Preserves KV cache for efficient generation

## Usage

### Basic Example

```python
from auto_interleaved_inference import AutoInterleavedInferencer

# Initialize the inferencer
inferencer = AutoInterleavedInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids
)

# Generate interleaved content
prompt = "Create a step-by-step tutorial on how to make a paper airplane."
outputs = inferencer.auto_interleaved_generation(
    prompt,
    max_text_length=200,
    max_interleaved_blocks=5,
    image_shape=(1024, 1024)
)

# Process outputs
for i, output in enumerate(outputs):
    if isinstance(output, str):
        print(f"Text {i}: {output}")
    else:  # PIL Image
        output.save(f"image_{i}.png")
```

### Advanced Configuration

```python
# With thinking/planning enabled
outputs = inferencer.auto_interleaved_generation(
    prompt="Explain photosynthesis with diagrams",
    think=True,                    # Enable planning phase
    max_think_token_n=1000,       # Max tokens for planning
    do_sample=False,              # Deterministic generation
    text_temperature=0.3,         # Text generation temperature
    cfg_text_scale=4.0,          # Text CFG scale
    cfg_img_scale=1.5,           # Image CFG scale
    cfg_interval=[0.4, 1.0],     # CFG application interval
    timestep_shift=3.0,          # Diffusion timestep shift
    num_timesteps=50,            # Number of diffusion steps
    image_shape=(1024, 1024)     # Generated image size
)
```

### Explicit Vision Tokens

You can also include explicit vision tokens in your prompt:

```python
prompt = """Describe three scenes:
1. A sunset over mountains <|vision_start|>
2. A busy city street <|vision_start|>
3. A quiet forest path <|vision_start|>"""

outputs = inferencer.auto_interleaved_generation(prompt)
```

## Implementation Details

### Generation Loop

1. **Text Generation Phase**:
   - Generates tokens one at a time
   - Monitors each token for special vision markers
   - Accumulates text tokens until a special token is encountered

2. **Image Generation Trigger**:
   - When `<|vision_start|>` is detected:
     - Saves accumulated text
     - Updates contexts with generated text
     - Switches to image generation mode
     - Generates image using current context
     - Updates context with generated image
     - Adds `<|vision_end|>` token

3. **Context Preservation**:
   - Maintains continuous context across text and image blocks
   - Ensures generated content influences subsequent generation
   - Supports long-form multimodal content creation

### Error Handling

- Input validation for all parameters
- Graceful handling of generation failures
- Detailed logging for debugging
- Returns partial results on error rather than losing all progress

## Examples

### Recipe Generation
```python
outputs = inferencer.auto_interleaved_generation(
    "Create a recipe for chocolate chip cookies with images showing key steps.",
    max_interleaved_blocks=8
)
```

### Educational Content
```python
outputs = inferencer.auto_interleaved_generation(
    "Explain the water cycle with diagrams for each stage.",
    think=True,
    cfg_text_scale=6.0
)
```

### Story Illustration
```python
outputs = inferencer.auto_interleaved_generation(
    "Write a short story about a robot's adventure in space with illustrations.",
    do_sample=True,
    text_temperature=0.7
)
```

## Testing

Run the test suite:

```bash
python test_auto_interleaved.py
```

## Performance Considerations

- Token-by-token generation may be slower than batch generation
- Image generation is computationally intensive
- Consider using smaller `num_timesteps` for faster generation
- Adjust `max_interleaved_blocks` based on your needs

## Future Improvements

1. **Batch Token Generation**: Generate multiple tokens at once until special tokens
2. **Parallel Processing**: Generate multiple images in parallel when possible
3. **Smart Context Management**: Optimize context updates for better performance
4. **Enhanced Token Detection**: Support additional control tokens
5. **Streaming Output**: Support real-time streaming of generated content

## Troubleshooting

### Common Issues

1. **No Images Generated**:
   - Model may not be generating vision tokens naturally
   - Try including explicit `<|vision_start|>` in prompt
   - Adjust generation parameters (temperature, CFG scales)

2. **Poor Image Quality**:
   - Increase `num_timesteps` for better quality
   - Adjust `cfg_img_scale` and `cfg_text_scale`
   - Try different `cfg_renorm_type` settings

3. **Context Length Errors**:
   - Reduce `max_text_length` or `max_interleaved_blocks`
   - Clear context periodically for very long generations

## References

- [Bagel Paper](https://arxiv.org/abs/2505.14683)
- [Original Issue #99](https://github.com/ByteDance-Seed/Bagel/issues/99)
- [Bagel Model Documentation](https://bagel-ai.org/)