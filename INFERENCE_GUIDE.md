# BAGEL Inference System Documentation

This document explains how `inferencer.py` implements interleaved multimodal inference, including context management, KV cache updates, and Classifier-Free Guidance (CFG).

---

## Table of Contents

1. [Overview](#overview)
2. [Interleaved Inference Architecture](#interleaved-inference-architecture)
3. [Context Management & KV Cache](#context-management--kv-cache)
4. [Classifier-Free Guidance (CFG)](#classifier-free-guidance-cfg)
5. [Inference Flow Examples](#inference-flow-examples)
6. [CFG Training vs Inference](#cfg-training-vs-inference)

---

## Overview

**File:** `inferencer.py`

The `InterleaveInferencer` class enables autoregressive generation of interleaved text and images by:
1. Maintaining a **generation context** with KV cache
2. Processing inputs (text/images) sequentially
3. Using **separate CFG contexts** for text and image conditioning
4. Generating outputs (text or images) with classifier-free guidance

**Key Principle:** The model builds up context incrementally, caching keys and values from previous tokens to enable efficient generation without recomputing the entire history.

---

## Interleaved Inference Architecture

### Class Structure

```python
class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model                    # Main BAGEL model
        self.vae_model = vae_model            # VAE for image encoding/decoding
        self.tokenizer = tokenizer            # Text tokenizer
        self.vae_transform = vae_transform    # Image transform for VAE (stride=16)
        self.vit_transform = vit_transform    # Image transform for ViT (stride=14)
        self.new_token_ids = new_token_ids    # Special token IDs
```

### Core Methods

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `init_gen_context()` | Initialize empty context | None | `gen_context` dict |
| `update_context_text()` | Add text to context | text string | Updated context |
| `update_context_image()` | Add image to context | PIL Image | Updated context |
| `gen_text()` | Generate text response | context | Text string |
| `gen_image()` | Generate image | context + CFG contexts | PIL Image |
| `interleave_inference()` | Main inference loop | List[str\|Image] | List[str\|Image] |

---

## Context Management & KV Cache

### Generation Context Structure

**File:** `inferencer.py`, lines 31-37

```python
def init_gen_context(self):
    gen_context = {
        'kv_lens': [0],                  # Current KV cache length
        'ropes': [0],                    # Current RoPE position
        'past_key_values': NaiveCache()  # KV cache for all layers
    }
    return gen_context
```

**Key-Value Cache (`NaiveCache`):**
- Stores keys and values for all transformer layers
- Enables efficient generation without recomputing past tokens
- Updated incrementally as new tokens/images are processed

**Position Tracking:**
- `kv_lens`: Total number of cached tokens so far
- `ropes`: Current RoPE (Rotary Position Embedding) position

### Text Context Update

**File:** `inferencer.py`, lines 39-59

```python
@torch.no_grad()
def update_context_text(self, text, gen_context):
    past_key_values = gen_context['past_key_values']
    kv_lens = gen_context['kv_lens']
    ropes = gen_context['ropes']

    # Prepare text tokens
    generation_input, kv_lens, ropes = self.model.prepare_prompts(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        prompts=[text],
        tokenizer=self.tokenizer,
        new_token_ids=self.new_token_ids,
    )

    # Forward pass and update KV cache
    past_key_values = self.model.forward_cache_update_text(
        past_key_values, **generation_input
    )

    # Update context
    gen_context['kv_lens'] = kv_lens
    gen_context['ropes'] = ropes
    gen_context['past_key_values'] = past_key_values

    return gen_context
```

**What happens:**
1. Tokenize text → token IDs
2. Forward pass through LLM
3. **Store keys and values** in KV cache
4. Update position counters

**Key Insight:** Text tokens use **causal attention** (attend to previous tokens only).

### Image Context Update

**File:** `inferencer.py`, lines 61-96

```python
@torch.no_grad()
def update_context_image(self, image, gen_context, vae=True, vit=True):
    assert vae or vit
    past_key_values = gen_context['past_key_values']
    kv_lens = gen_context['kv_lens']
    ropes = gen_context['ropes']

    if vae:
        # Update with VAE (clean) tokens
        generation_input, kv_lens, ropes = self.model.prepare_vae_images(...)
        past_key_values = self.model.forward_cache_update_vae(
            self.vae_model, past_key_values, **generation_input
        )

    if vit:
        # Update with ViT tokens
        generation_input, kv_lens, ropes = self.model.prepare_vit_images(...)
        past_key_values = self.model.forward_cache_update_vit(
            past_key_values, **generation_input
        )

    # Update context
    gen_context['kv_lens'] = kv_lens
    gen_context['ropes'] = ropes
    gen_context['past_key_values'] = past_key_values

    return gen_context
```

**What happens:**
1. Transform image with VAE/ViT transforms
2. **VAE path (if `vae=True`):**
   - Encode image → latent codes
   - Forward through `vae2llm` projection
   - Cache as **clean VAE tokens** (t=0, no noise)
3. **ViT path (if `vit=True`):**
   - Patchify image with ViT encoder
   - Forward through vision transformer
   - Cache as **ViT tokens**

**Key Insight:**
- Image tokens use **full attention** (attend to all previous tokens + within image)
- Clean VAE and ViT provide conditioning for **future generation**
- During inference, we **never store noised VAE** in the cache (only clean representations)

---

## Classifier-Free Guidance (CFG)

### What is CFG?

**Classifier-Free Guidance** improves generation quality by combining:
1. **Conditional prediction:** Model with full context (text + images)
2. **Unconditional predictions:** Model with reduced context (dropped conditioning)

**Formula:**
```
prediction = unconditional + scale * (conditional - unconditional)
```

Where:
- `scale > 1.0`: Amplifies the effect of conditioning
- `scale = 1.0`: No guidance (pure conditional)

### CFG in BAGEL: Two Types

BAGEL uses **dual CFG** with separate scales for text and image conditioning:

1. **Text CFG (`cfg_text_scale`):** Guidance from text prompts
2. **Image CFG (`cfg_img_scale`):** Guidance from input images (for editing tasks)

### CFG Context Management

**File:** `inferencer.py`, lines 229-256

During `interleave_inference()`, three contexts are maintained:

```python
gen_context = self.init_gen_context()        # Full context
cfg_text_context = deepcopy(gen_context)     # Context without last text
cfg_img_context = deepcopy(gen_context)      # Context without images
```

**Context Update Strategy:**

```python
for input_term in input_lists:
    if isinstance(input_term, str):
        # Before adding text, save current state for text CFG
        cfg_text_context = deepcopy(gen_context)

        # Add text to full context
        gen_context = self.update_context_text(input_term, gen_context)

        # Also add text to image CFG context
        cfg_img_context = self.update_context_text(input_term, cfg_img_context)

    elif isinstance(input_term, Image.Image):
        # Add image to full context (VAE + ViT)
        gen_context = self.update_context_image(
            input_term, gen_context, vae=not understanding_output
        )

        # Before adding image, save current state for image CFG
        cfg_text_context = deepcopy(gen_context)

        # Note: cfg_img_context does NOT get the image!
```

**Result:**
- `gen_context`: Has **everything** (all text + all images)
- `cfg_text_context`: Has everything **except the last text prompt**
- `cfg_img_context`: Has all text but **no images**

### CFG During Image Generation

**File:** `inferencer.py`, lines 99-171; `modeling/bagel/bagel.py`, lines 835-891

```python
def gen_image(self, image_shape, gen_context,
              cfg_text_scale=4.0, cfg_img_scale=1.5,
              cfg_text_precontext=None, cfg_img_precontext=None, ...):

    # Prepare latent generation with full context
    generation_input = self.model.prepare_vae_latent(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        image_sizes=[image_shape],
        ...
    )

    # Prepare CFG contexts
    generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_precontext['kv_lens'],  # Without last text
        ...
    )

    generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_precontext['kv_lens'],  # Without any images
        ...
    )

    # Generate with CFG
    unpacked_latent = self.model.generate_image(
        past_key_values=past_key_values,
        cfg_text_past_key_values=cfg_text_past_key_values,
        cfg_img_past_key_values=cfg_img_past_key_values,
        cfg_text_scale=cfg_text_scale,  # e.g., 4.0
        cfg_img_scale=cfg_img_scale,    # e.g., 1.5
        ...
    )
```

**Inside `generate_image()` (flow matching loop):**

**File:** `modeling/bagel/bagel.py`, lines 835-891

```python
# 1. Full conditional prediction
v_t = self._forward_flow(x_t, timestep, full_context)

# 2. Text CFG prediction (if cfg_text_scale > 1.0)
if cfg_text_scale > 1.0:
    cfg_text_v_t = self._forward_flow(x_t, timestep, context_without_last_text)

# 3. Image CFG prediction (if cfg_img_scale > 1.0)
if cfg_img_scale > 1.0:
    cfg_img_v_t = self._forward_flow(x_t, timestep, context_without_images)

# 4. Apply CFG (dual guidance)
if cfg_text_scale > 1.0:
    v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

    if cfg_img_scale > 1.0:
        # Nested CFG: apply image guidance to text-guided result
        v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
    else:
        v_t_ = v_t_text_
```

**Visualization:**

```
Conditional (full)        v_t              [Text + Images]
Text Unconditional        cfg_text_v_t     [Images only, no last text]
Image Unconditional       cfg_img_v_t      [Text only, no images]

Step 1: Text guidance
v_t_text = cfg_text_v_t + 4.0 * (v_t - cfg_text_v_t)
           └─ unconditional ─┘    └─ conditional - unconditional ─┘

Step 2: Image guidance (applied to text-guided result)
v_t_final = cfg_img_v_t + 1.5 * (v_t_text - cfg_img_v_t)
```

### CFG Interval

**File:** `inferencer.py`, lines 108, 154; `modeling/bagel/bagel.py`, lines 701-706

```python
cfg_interval=(0.4, 1.0)  # Apply CFG only for t ∈ [0.4, 1.0]

# During generation loop
if t > cfg_interval[0] and t <= cfg_interval[1]:
    cfg_text_scale_ = cfg_text_scale  # e.g., 4.0
    cfg_img_scale_ = cfg_img_scale    # e.g., 1.5
else:
    cfg_text_scale_ = 1.0  # No guidance
    cfg_img_scale_ = 1.0
```

**Rationale:**
- Early timesteps (t > 0.4): Strong guidance needed for structure
- Late timesteps (t < 0.4): Let model refine details naturally
- Default: `[0.4, 1.0]` applies CFG for first 60% of generation

---

## Inference Flow Examples

### Example 1: Text-to-Image Generation

**Input:**
```python
input_lists = ["A cat sitting on a mat"]
```

**Flow:**

```
Step 1: Initialize contexts
  gen_context        = {kv_lens: [0], ropes: [0], past_key_values: empty}
  cfg_text_context   = copy(gen_context)
  cfg_img_context    = copy(gen_context)

Step 2: Process text "A cat sitting on a mat"
  cfg_text_context   = copy(gen_context)  # Save state before text
  gen_context        = update_context_text("A cat...", gen_context)
  cfg_img_context    = update_context_text("A cat...", cfg_img_context)

  Result:
    gen_context:       [Text: "A cat..."]
    cfg_text_context:  []  (empty)
    cfg_img_context:   [Text: "A cat..."]

Step 3: Generate image
  Use gen_image(gen_context, cfg_text_context, cfg_img_context)

  Forward passes (per denoising step):
    1. Conditional:        [Text: "A cat..."] → v_t
    2. Text unconditional: []                 → cfg_text_v_t
    3. Image unconditional: Not used (no prior images)

  CFG formula:
    v_t_final = cfg_text_v_t + 4.0 * (v_t - cfg_text_v_t)

  Output: Generated image of cat on mat
```

### Example 2: Image Editing

**Input:**
```python
input_lists = [
    original_image,        # PIL Image
    "Make it red"          # Text instruction
]
```

**Flow:**

```
Step 1: Initialize contexts
  gen_context        = {kv_lens: [0], ...}
  cfg_text_context   = copy(gen_context)
  cfg_img_context    = copy(gen_context)

Step 2: Process original image
  gen_context        = update_context_image(original_image, gen_context, vae=True, vit=True)
  cfg_text_context   = copy(gen_context)  # Save state after image

  Result:
    gen_context:       [Clean VAE + ViT tokens]
    cfg_text_context:  [Clean VAE + ViT tokens]
    cfg_img_context:   []  (still empty)

Step 3: Process text "Make it red"
  cfg_text_context   = copy(gen_context)  # Save state before text
  gen_context        = update_context_text("Make it red", gen_context)
  cfg_img_context    = update_context_text("Make it red", cfg_img_context)

  Result:
    gen_context:       [Clean VAE + ViT + Text: "Make it red"]
    cfg_text_context:  [Clean VAE + ViT]  (no text instruction)
    cfg_img_context:   [Text: "Make it red"]  (no image)

Step 4: Generate edited image
  Use gen_image(gen_context, cfg_text_context, cfg_img_context)

  Forward passes (per denoising step):
    1. Conditional:        [Image + Text] → v_t
    2. Text unconditional: [Image only]   → cfg_text_v_t
    3. Image unconditional: [Text only]   → cfg_img_v_t

  CFG formula (nested):
    v_t_text = cfg_text_v_t + 4.0 * (v_t - cfg_text_v_t)
    v_t_final = cfg_img_v_t + 1.5 * (v_t_text - cfg_img_v_t)

  Output: Red version of original image
```

### Example 3: Multi-Turn Image Editing

**Input:**
```python
input_lists = [
    original_image,
    "Make it red",
    # After first generation, second edit:
    generated_red_image,
    "Add a hat"
]
```

**Flow:**

```
[After first generation completes]

Step 5: Add generated image to context
  gen_context = update_context_image(generated_red_image, gen_context, vae=True, vit=True)
  cfg_text_context = copy(gen_context)

  Result:
    gen_context:       [Img0_VAE + Img0_ViT + "Make it red" + Img1_VAE + Img1_ViT]
    cfg_text_context:  [Img0_VAE + Img0_ViT + "Make it red" + Img1_VAE + Img1_ViT]
    cfg_img_context:   [Text: "Make it red"]

Step 6: Process second instruction
  cfg_text_context = copy(gen_context)
  gen_context = update_context_text("Add a hat", gen_context)
  cfg_img_context = update_context_text("Add a hat", cfg_img_context)

  Result:
    gen_context:       [...previous... + "Add a hat"]
    cfg_text_context:  [...previous...] (no "Add a hat")
    cfg_img_context:   ["Make it red" + "Add a hat"]

Step 7: Generate second edit
  Forward passes:
    1. Conditional:        [All images + all text] → v_t
    2. Text unconditional: [All images + first text] → cfg_text_v_t
    3. Image unconditional: [All text only] → cfg_img_v_t

  Output: Red image with hat
```

**Key Insight:** The model builds up a **history of edits** in the KV cache, enabling consistent multi-turn editing.

---

## CFG Training vs Inference

### Training: Random Dropout (Learning Unconditional Distribution)

**File:** `data/dataset_base.py`, lines 323-402

**Purpose:** Train the model to work **with and without** conditioning, enabling CFG at inference time.

**How it works:**

```python
# During training data packing
for item in sequence_plan:
    if item['type'] == 'text':
        if item['enable_cfg'] == 1 and random.random() < text_cond_dropout_prob:
            continue  # Skip this text token (drop it)

    elif item['type'] == 'vit_image':
        if item['enable_cfg'] == 1 and random.random() < vit_cond_dropout_prob:
            continue  # Skip this ViT image

    elif item['type'] == 'vae_image':
        if item['enable_cfg'] == 1 and random.random() < vae_cond_dropout_prob:
            continue  # Skip this clean VAE image
```

**Dropout Probabilities (default):**
- Text: `0.1` (10% dropout)
- ViT: `0.3` (30% dropout)
- Clean VAE: `0.3` (30% dropout)
- **Noised VAE: NEVER dropped** (`enable_cfg=0` always)

**Training Examples:**

```
Original sequence (t2i):
  [Text: "cat"] → [Noised VAE (loss)]

With 10% text dropout:
  90% of time: [Text: "cat"] → [Noised VAE (loss)]
  10% of time: []            → [Noised VAE (loss)]  ← Unconditional
```

```
Original sequence (image editing):
  [Clean VAE] → [ViT] → [Text: "red"] → [Noised VAE (loss)]

With dropout (example):
  80%: [Clean VAE] → [ViT] → [Text: "red"] → [Noised VAE (loss)]
  15%: [ViT] → [Text: "red"] → [Noised VAE (loss)]  ← VAE dropped
  4%:  [Clean VAE] → [Text: "red"] → [Noised VAE (loss)]  ← ViT dropped
  1%:  [Text: "red"] → [Noised VAE (loss)]  ← Both VAE and ViT dropped
```

**Result:** Model learns:
- **Conditional distribution:** P(image | text, input_image)
- **Partially unconditional:** P(image | text), P(image | input_image)
- **Fully unconditional:** P(image)

### Inference: Multiple Forward Passes (Applying Guidance)

**File:** `modeling/bagel/bagel.py`, lines 835-891

**Purpose:** Use the learned conditional and unconditional distributions to **amplify** the effect of conditioning.

**How it works:**

```python
# For each denoising timestep in generation loop:

# 1. Conditional forward (full context)
v_t = model.forward(x_t, timestep, context_with_everything)

# 2. Text unconditional forward (context without last text)
if cfg_text_scale > 1.0:
    cfg_text_v_t = model.forward(x_t, timestep, context_without_text)

# 3. Image unconditional forward (context without images)
if cfg_img_scale > 1.0:
    cfg_img_v_t = model.forward(x_t, timestep, context_without_images)

# 4. Combine predictions with CFG
v_t_text = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
v_t_final = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
```

**No random dropout!** Instead, we **deterministically** create different contexts and run multiple forward passes.

### When to Set CFG Scale to 1 vs Other Values?

#### **CFG Scale = 1.0 (No Guidance)**

**When to use:**
- Quick generation without quality enhancement
- Sampling diverse outputs (e.g., for data generation)
- When conditioning is already very strong
- Debugging unconditional generation

**Effect:**
```python
cfg_scale = 1.0
v_t_final = cfg_v_t + 1.0 * (v_t - cfg_v_t) = v_t  # Pure conditional
```
No amplification of conditioning signal.

#### **CFG Scale > 1.0 (With Guidance)**

**Typical values:**
- **Text CFG:** `3.0 - 5.0` (default: `4.0`)
  - Higher values → stronger adherence to text prompt
  - Too high → oversaturated, artificial-looking images
- **Image CFG:** `1.2 - 2.0` (default: `1.5`)
  - Higher values → closer to input image (for editing)
  - Too high → hard to make significant changes

**When to use:**
- **High text CFG (4.0-5.0):** Complex text prompts, detailed descriptions
- **Moderate text CFG (2.0-3.0):** Simple prompts, natural-looking results
- **Low text CFG (1.2-1.5):** Artistic freedom, diverse outputs
- **High image CFG (1.5-2.0):** Precise editing, preserve input structure
- **Low image CFG (1.0-1.3):** Creative editing, more freedom

**Effect:**
```python
cfg_text_scale = 4.0
cfg_img_scale = 1.5

# Step 1: Text guidance
v_t_text = cfg_text_v_t + 4.0 * (v_t - cfg_text_v_t)
# Amplifies text conditioning by 4×

# Step 2: Image guidance
v_t_final = cfg_img_v_t + 1.5 * (v_t_text - cfg_img_v_t)
# Amplifies image conditioning by 1.5× on top of text guidance
```

#### **CFG Interval (Timestep Range)**

**Default:** `[0.4, 1.0]`

```python
if timestep > 0.4 and timestep <= 1.0:
    # Apply full CFG scales
else:
    # Set scales to 1.0 (no guidance)
```

**Rationale:**
- **Early timesteps (t ∈ [0.7, 1.0]):** Strong structure formation → need strong guidance
- **Middle timesteps (t ∈ [0.4, 0.7]):** Refining structure → moderate guidance
- **Late timesteps (t < 0.4):** Fine details → let model explore naturally

**When to adjust:**
- **Wider interval `[0.2, 1.0]`:** More consistent adherence to conditioning
- **Narrower interval `[0.6, 1.0]`:** More creative freedom in details
- **Full range `[0.0, 1.0]`:** Maximum conditioning adherence (may look artificial)

### Summary Table: Training vs Inference CFG

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Mechanism** | Random dropout of conditioning | Multiple forward passes with different contexts |
| **When applied** | During data packing | During generation loop (each timestep) |
| **Control parameter** | `enable_cfg` in sequence plan | `cfg_text_scale`, `cfg_img_scale` |
| **Dropout rates** | Text: 0.1, ViT: 0.3, VAE: 0.3 | N/A (deterministic context manipulation) |
| **Purpose** | Learn conditional + unconditional | Amplify conditioning effect |
| **File** | `data/dataset_base.py:323-402` | `modeling/bagel/bagel.py:835-891` |
| **Randomness** | Stochastic (different each sample) | Deterministic (same for all timesteps) |
| **Cost** | No extra cost (single forward pass) | 2-3× cost (multiple forward passes) |

---

## Advanced: CFG Renormalization

**File:** `modeling/bagel/bagel.py`, lines 874-891

**Problem:** CFG can increase prediction magnitude, causing unstable generation.

**Solution:** Renormalize CFG-guided prediction to match original magnitude.

**Types:**

### 1. Global Renormalization (Default)

```python
cfg_renorm_type = "global"

v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)

# Renormalize globally (not implemented in shown code, likely in full flow)
```

No per-token normalization, simpler and faster.

### 2. Text Channel Renormalization

```python
cfg_renorm_type = "text_channel"

# Apply text CFG
v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

# Compute norms
norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)

# Scale factor (clamp to avoid shrinking too much)
scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)

# Renormalize
v_t_text = v_t_text_ * scale

# Then apply image CFG
v_t_final = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
```

**Effect:** Keeps prediction magnitude similar to original, preventing oversaturation.

**Parameters:**
- `cfg_renorm_min` (default: `0.0`): Minimum allowed scale factor
  - Higher values → more aggressive renormalization
  - `0.0` allows full renormalization down to zero

---

## Best Practices

### For Text-to-Image Generation:

```python
inferencer.interleave_inference(
    ["A beautiful sunset over mountains"],
    cfg_text_scale=4.0,      # Strong text guidance
    cfg_img_scale=1.0,       # No image guidance (no input image)
    cfg_interval=[0.4, 1.0], # Standard interval
    num_timesteps=50,        # Good quality/speed tradeoff
)
```

### For Image Editing:

```python
inferencer.interleave_inference(
    [input_image, "Make it sunset style"],
    cfg_text_scale=3.0,      # Moderate text guidance
    cfg_img_scale=1.5,       # Preserve input structure
    cfg_interval=[0.4, 1.0],
    num_timesteps=50,
)
```

### For Creative/Diverse Outputs:

```python
inferencer.interleave_inference(
    ["A futuristic cityscape"],
    cfg_text_scale=2.0,      # Lower guidance for diversity
    cfg_img_scale=1.0,
    cfg_interval=[0.6, 1.0], # Less CFG = more freedom
    num_timesteps=30,        # Faster
)
```

### For High-Quality/Detailed:

```python
inferencer.interleave_inference(
    ["A photorealistic portrait of an elderly man"],
    cfg_text_scale=5.0,      # Strong guidance
    cfg_img_scale=1.0,
    cfg_interval=[0.3, 1.0], # Wider CFG interval
    num_timesteps=100,       # More steps
    cfg_renorm_type="text_channel",  # Prevent oversaturation
    cfg_renorm_min=0.5,      # Moderate renormalization
)
```

---

## Conclusion

**Key Takeaways:**

1. **Context Management:** KV cache stores all previous tokens (text + clean VAE + ViT) for efficient generation

2. **Dual CFG:** Separate guidance from text and images enables precise control in editing tasks

3. **Training:** Random dropout (`enable_cfg=1`) teaches model to work with and without conditioning

4. **Inference:** Multiple forward passes with different contexts amplify conditioning effects

5. **CFG Scale:**
   - `1.0` = no guidance (faster, more diverse)
   - `2.0-5.0` = moderate to strong guidance (better quality, adherence to prompts)
   - Text: typically higher (3-5), Image: typically lower (1.2-2.0)

6. **CFG Interval:** `[0.4, 1.0]` applies guidance where it matters most (structure formation)

7. **Never cache noised VAE:** Only clean representations stored for conditioning

This architecture enables BAGEL's powerful interleaved generation and editing capabilities while maintaining consistency across multi-turn interactions.
