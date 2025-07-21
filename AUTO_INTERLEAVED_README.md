# Auto-Interleaved Generation

Autonomous text and image generation for Bagel models, addressing [Issue #99](https://github.com/ByteDance-Seed/Bagel/issues/99).

## Usage

```bash
python demo_auto_interleaved.py --prompt "Tell a visual story" --do-sample
```

## How It Works

1. Model generates text tokens one by one
2. When model produces `<|vision_start|>` token (151652), system switches to image generation
3. Image is generated using current text as context
4. Generation returns to text mode
5. Process continues until max blocks or end token

## Current Limitation

**BAGEL-7B-MoT was NOT trained on vision start tokens.** From the original author:
> "We didn't train on the `<image>` token, so the model won't naturally generate it during inference."

This means the current model cannot autonomously generate images. The infrastructure is ready for a retrained model.