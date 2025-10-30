# Diffusion Forcing in BAGEL: Complete Technical Guide

## Table of Contents
1. [What is Diffusion Forcing?](#what-is-diffusion-forcing)
2. [Why is Diffusion Forcing Needed?](#why-is-diffusion-forcing-needed)
3. [BAGEL's Implementation](#bagels-implementation)
4. [Concrete Examples](#concrete-examples)
5. [Code Walkthrough](#code-walkthrough)
6. [Comparison with Other Approaches](#comparison-with-other-approaches)

---

## What is Diffusion Forcing?

**Diffusion forcing** is a training paradigm for sequence diffusion models where **independent noise levels are assigned to different elements** in a sequence, rather than using a single noise level for the entire sequence.

### Traditional vs. Diffusion Forcing

**Traditional Sequence Diffusion:**
- All frames in a video share the same noise timestep `t`
- During training: Sample one `t ~ N(0,1)`, apply to all frames
- Limited flexibility: Only `T` possible noise configurations

**Diffusion Forcing (Original Paper):**
- Each frame gets an independent noise timestep `t_i ~ N(0,1)`
- During training: Sample `t_1, t_2, ..., t_N` independently
- Massive flexibility: `T^N` possible noise configurations for N frames
- Enables **autoregressive generation**: denoise frame `i` conditioned on noisier frames `1...i-1`

### Key Innovation

The critical insight is that by training on sequences with **variable noise patterns**, the model learns to:
1. Generate frames autoregressively (denoise next frame given context)
2. Handle arbitrary context lengths (not limited to training sequence length)
3. Maintain temporal consistency (context frames guide generation)
4. Avoid error accumulation (each frame is properly noised, not generated-then-used)

---

## Why is Diffusion Forcing Needed?

### 1. **Autoregressive Video/Image Generation**

Without diffusion forcing, sequence models face the **train-test mismatch**:
- **Training:** Model sees clean ground-truth frames as context
- **Testing:** Model must use its own generated (imperfect) frames as context
- **Result:** Error accumulation, quality degrades over time

With diffusion forcing:
- **Training:** Model practices denoising with noisy context frames
- **Testing:** Generated frames are naturally "noisy" representations
- **Result:** Consistent quality across long sequences

**Example:**
```
Traditional AR model generating 10-frame video:
Frame 1: Good (conditioned on nothing)
Frame 2: Good (conditioned on clean Frame 1)
Frame 3: Okay (conditioned on clean Frames 1-2)
...
Frame 10: Poor (accumulated errors from Frames 1-9)

Diffusion Forcing model:
Frame 1: Good (denoised from noise)
Frame 2: Good (denoised from noise, conditioned on noisy Frame 1)
Frame 3: Good (denoised from noise, conditioned on noisy Frames 1-2)
...
Frame 10: Good (denoised from noise, conditioned on noisy Frames 1-9)
```

### 2. **Flexible Context Lengths**

Traditional models are limited to fixed context windows from training. Diffusion forcing enables:
- Training on short sequences (e.g., 8 frames)
- Inference on arbitrary lengths (e.g., 100+ frames)
- Dynamic context management (process variable-length histories)

### 3. **Temporal Consistency**

Independent noise levels allow the model to:
- Learn relationships between frames at different noise levels
- Maintain object identity across frames
- Preserve motion continuity
- Handle occlusions and appearance changes

### 4. **Multi-Image Editing Chains**

For image editing workflows (BAGEL's key use case):
```
Original Image → Edit 1 → Edit 2 → Edit 3
```

Each edit can be generated with:
- Independent noise level (prevents error accumulation)
- Full context from previous edits (maintains consistency)
- Flexible conditioning (can skip intermediate edits)

---

## BAGEL's Implementation

BAGEL implements a **modified version** of diffusion forcing that balances flexibility with efficiency:

### Core Mechanism: `split_start` and `split_end`

Instead of per-frame noise, BAGEL uses **per-sequence-segment noise**:

```python
# From dataset_base.py:317-319
split_start = item.get('split_start', True)
if split_start:
    curr_split_len = 0  # Start a new segment
```

```python
# From dataset_base.py:428-431
if item['loss'] == 1:  # This is a noised VAE token
    if split_start:  # NEW SEGMENT → NEW NOISE LEVEL
        timestep = np.random.randn()
else:  # This is a clean VAE token
    timestep = float('-inf')  # Always t=0
```

**Key Points:**
- `split_start=True`: Assigns a **new random noise level** `t ~ N(0,1)`
- `split_start=False`: **Reuses** the previous noise level
- All frames between `split_start` and `split_end` share the same `t`

### Temporal Information: `frame_delta`

For video sequences, BAGEL encodes temporal relationships via position IDs:

```python
# From dataset_base.py:455-458
if 'frame_delta' in item.keys():
    curr_rope_id += item['frame_delta']  # Jump forward by frame spacing
elif item['loss'] == 0:
    curr_rope_id += 1  # Standard increment
```

**Example:**
```
Video frames at indices [0, 5, 10, 15]
frame_delta values:      [5, 5,  5, -]

RoPE positions: [0, 5, 10, 15]
→ Model learns temporal spacing between frames
```

### Attention Masking for Noised Tokens

When `split_start=True` and the token is noised, special attention rules apply:

```python
# From dataset_base.py:450-453
if split_start:
    if item['loss'] == 1 and 'frame_delta' not in item.keys():
        attn_modes.append("noise")  # Invisible to future tokens
    else:
        attn_modes.append("full")   # Normal full attention
```

**"noise" mode** (from data_utils.py:93-97):
```python
if attn_mode == "noise":
    # Block all future tokens from attending TO this noised token
    attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
    # But the noised token CAN attend to itself
    attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
```

**Result:** Noised VAE tokens are invisible to future context (prevents information leakage).

---

## Concrete Examples

### Example 1: Video Generation with Shared Noise

**Scenario:** Generate 4 frames from a video at frame indices [0, 10, 20, 30]

**Data Structure:**
```python
sequence_plan = [
    # Frame 0
    {'type': 'vae_image', 'loss': 1, 'split_start': True, 'split_end': False, 'frame_delta': 10},
    # Frame 10
    {'type': 'vae_image', 'loss': 1, 'split_start': False, 'split_end': False, 'frame_delta': 10},
    # Frame 20
    {'type': 'vae_image', 'loss': 1, 'split_start': False, 'split_end': False, 'frame_delta': 10},
    # Frame 30
    {'type': 'vae_image', 'loss': 1, 'split_start': False, 'split_end': True},
]
```

**Noise Assignment:**
```
split_start=True at Frame 0 → t = -0.73 (sampled once)

All frames use t = -0.73:
Frame 0:  Noised VAE with t=-0.73
Frame 10: Noised VAE with t=-0.73
Frame 20: Noised VAE with t=-0.73
Frame 30: Noised VAE with t=-0.73
```

**RoPE Position IDs:**
```
Frame 0:  RoPE = [0, 1, ..., 255]      (16x16 patches)
Frame 10: RoPE = [10, 11, ..., 265]    (jumped by frame_delta=10)
Frame 20: RoPE = [20, 21, ..., 275]    (jumped by frame_delta=10)
Frame 30: RoPE = [30, 31, ..., 285]    (jumped by frame_delta=10)
```

**Why this design?**
- Frames are temporally related → shared noise preserves consistency
- `frame_delta` encodes temporal spacing → model learns motion
- All frames train together → efficient batch processing

---

### Example 2: Multi-Step Image Editing with Independent Noise

**Scenario:** Three consecutive edits: Original → Red Background → Add Cat → Change Lighting

**Sequence Plan:**
```python
sequence_plan = [
    # Original image (conditioning only, no generation)
    {'type': 'vae_image', 'loss': 0, 'split_start': True, 'split_end': True},   # Clean VAE
    {'type': 'vit_image', 'loss': 0, 'split_start': True, 'split_end': True},   # ViT tokens

    # Instruction: "Change background to red"
    {'type': 'text', 'loss': 0, 'split_start': True, 'split_end': True},

    # Edited image 1 (GENERATE with independent noise)
    {'type': 'vae_image', 'loss': 1, 'split_start': True, 'split_end': True},   # Noised VAE
    {'type': 'vae_image', 'loss': 0, 'split_start': True, 'split_end': True},   # Clean VAE
    {'type': 'vit_image', 'loss': 0, 'split_start': True, 'split_end': True},   # ViT tokens

    # Instruction: "Add a cat in the foreground"
    {'type': 'text', 'loss': 0, 'split_start': True, 'split_end': True},

    # Edited image 2 (GENERATE with independent noise)
    {'type': 'vae_image', 'loss': 1, 'split_start': True, 'split_end': True},   # Noised VAE
    {'type': 'vae_image', 'loss': 0, 'split_start': True, 'split_end': True},   # Clean VAE
    {'type': 'vit_image', 'loss': 0, 'split_start': True, 'split_end': True},   # ViT tokens

    # Instruction: "Make the lighting warmer"
    {'type': 'text', 'loss': 0, 'split_start': True, 'split_end': True},

    # Edited image 3 (GENERATE with independent noise)
    {'type': 'vae_image', 'loss': 1, 'split_start': True, 'split_end': True},   # Noised VAE
]
```

**Noise Assignment:**
```
Original Image:     t = -inf (clean, no noise)
Edit 1 Noised VAE:  t = 0.42  (NEW noise level from split_start)
Edit 1 Clean VAE:   t = -inf
Edit 2 Noised VAE:  t = -1.15 (NEW noise level from split_start)
Edit 2 Clean VAE:   t = -inf
Edit 3 Noised VAE:  t = 0.88  (NEW noise level from split_start)
```

**Training Dynamics:**
```
Training iteration 1:
- Model learns to denoise Edit 1 from t=0.42
- Context: Original image (clean), instruction text
- Loss: MSE on Edit 1's noised VAE

Training iteration 2 (different sample):
- Model learns to denoise Edit 2 from t=-1.15
- Context: Original image (clean), Edit 1 (clean VAE + ViT), instruction text
- Loss: MSE on Edit 2's noised VAE

Training iteration 3 (different sample):
- Model learns to denoise Edit 3 from t=0.88
- Context: Original image, Edit 1, Edit 2 (all clean), instruction text
- Loss: MSE on Edit 3's noised VAE
```

**Why independent noise?**
- Each edit faces a different difficulty level (different `t`)
- Model learns to handle varying noise levels at each step
- Prevents overfitting to single noise schedule
- Enables flexible inference (can start from any noise level)

---

### Example 3: Mixed Context with Diffusion Forcing

**Scenario:** Generate image based on text + 2 reference images + more text

**Sequence Plan:**
```python
sequence_plan = [
    # Text prompt
    {'type': 'text', 'loss': 0, 'split_start': True, 'split_end': True},

    # Reference image 1 (clean)
    {'type': 'vae_image', 'loss': 0, 'split_start': True, 'split_end': True},
    {'type': 'vit_image', 'loss': 0, 'split_start': True, 'split_end': True},

    # More text
    {'type': 'text', 'loss': 0, 'split_start': True, 'split_end': True},

    # Reference image 2 (clean)
    {'type': 'vae_image', 'loss': 0, 'split_start': True, 'split_end': True},
    {'type': 'vit_image', 'loss': 0, 'split_start': True, 'split_end': True},

    # Final instruction
    {'type': 'text', 'loss': 0, 'split_start': True, 'split_end': True},

    # Generated image (noised)
    {'type': 'vae_image', 'loss': 1, 'split_start': True, 'split_end': True},
]
```

**Noise Assignment:**
```
Text 1:              No timestep (text has no VAE representation)
Reference Image 1:   t = -inf (clean conditioning)
Text 2:              No timestep
Reference Image 2:   t = -inf (clean conditioning)
Text 3:              No timestep
Generated Image:     t = -0.35 (NEW noise level from split_start)
```

**Attention Pattern:**
```
                    Text1  Img1  Text2  Img2  Text3  GenImg
Text 1:             [✓     ✓     ✓      ✓     ✓      ✓    ]  Causal
Reference Image 1:  [✓     ✓     ✓      ✓     ✓      ✓    ]  Full
Text 2:             [✓     ✓     ✓      ✓     ✓      ✓    ]  Causal
Reference Image 2:  [✓     ✓     ✓      ✓     ✓      ✓    ]  Full
Text 3:             [✓     ✓     ✓      ✓     ✓      ✓    ]  Causal
Generated Image:    [✓     ✓     ✓      ✓     ✓      ✓    ]  Noise (hidden from future)
```

**Training Goal:**
- Model learns to denoise generated image at t=-0.35
- Can attend to all previous context (text + images)
- Different training samples use different t values
- Generalizes to any noise level at inference

---

## Code Walkthrough

### 1. Setting `split_start` in Datasets

**Video Dataset** (`interleave_t2i_dataset.py:88-129`):
```python
def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
    if need_loss:  # Noised VAE tokens
        for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
            current_sequence_plan = {
                'type': 'vae_image',
                'enable_cfg': 0,
                'loss': 1,
                'split_start': idx == 0,           # ← FIRST FRAME ONLY
                'split_end': idx == len(frames) - 1,  # ← LAST FRAME ONLY
            }
            if idx < len(frame_indexes) - 1:
                # Encode temporal spacing in position IDs
                current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
            data['sequence_plan'].append(current_sequence_plan)
```

**Key Insight:** Only the **first frame** triggers new noise assignment. All subsequent frames in the video share that noise level.

---

**Edit Dataset** (`edit_dataset.py:29-71`):
```python
# Original image (clean only)
data = self._add_image(
    data,
    original_image,
    need_loss=False,  # ← No noised VAE → no noise assignment
    need_vae=True,    # ← Clean VAE for conditioning
    need_vit=True,
)

# ... instruction text ...

# Edited image (all three tokens)
data = self._add_image(
    data,
    edited_image,
    need_loss=True,   # ← Noised VAE → gets NEW noise level (split_start defaults to True)
    need_vae=True,    # ← Clean VAE for next edit
    need_vit=True,
)
```

**Key Insight:** Each `_add_image()` call with `need_loss=True` creates a new noised VAE token, which by default has `split_start=True` (from `interleave_t2i_dataset.py:42-86`).

---

### 2. Noise Assignment in `pack_sequence()`

**Location:** `dataset_base.py:428-431`

```python
if item['loss'] == 1:  # This is a noised VAE token
    if split_start:
        timestep = np.random.randn()  # ← NEW INDEPENDENT NOISE LEVEL
    # else: timestep remains from previous token (same segment)
else:  # Clean VAE token
    timestep = float('-inf')  # ← Always t=0
```

**Timestep Storage:**
```python
sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
```

Each VAE token (256 tokens for 16x16 image) stores the same timestep value.

---

### 3. Attention Mode Assignment

**Location:** `dataset_base.py:448-453`

```python
if split_start:
    if item['loss'] == 1 and 'frame_delta' not in item.keys():
        attn_modes.append("noise")  # ← Noised VAE without frame_delta
    else:
        attn_modes.append("full")   # ← Clean VAE or video frames
```

**Why the distinction?**
- **Image editing:** Noised VAE should be hidden (`"noise"` mode)
- **Video frames:** Noised VAE can be visible (`"full"` mode, has `frame_delta`)
- Allows model to learn different interaction patterns

---

### 4. Position ID Handling with `frame_delta`

**Location:** `dataset_base.py:454-458`

```python
sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
if 'frame_delta' in item.keys():
    curr_rope_id += item['frame_delta']  # ← Large jump for video
elif item['loss'] == 0:  # Clean VAE
    curr_rope_id += 1  # ← Small increment
# else: noised VAE without frame_delta keeps same RoPE ID
```

**Example RoPE progression:**
```
Text tokens [0-50]:        RoPE IDs = [0, 1, 2, ..., 50]
ViT image [51-306]:        RoPE IDs = [51, 51, 51, ..., 51]  (same position)
VAE image [307-562]:       RoPE IDs = [51, 51, 51, ..., 51]  (same position)
Text tokens [563-600]:     RoPE IDs = [52, 53, 54, ..., 89]
Video frame 1 [601-856]:   RoPE IDs = [90, 90, ..., 90]
Video frame 2 [857-1112]:  RoPE IDs = [100, 100, ..., 100]  (jumped by frame_delta=10)
```

---

### 5. FlexAttention with Diffusion Forcing

**Location:** `data_utils.py:13-40`

```python
def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def full_and_noise_mask(b, h, q_idx, kv_idx):
        # Tokens in same segment (same seq_id) can attend to each other
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & \
               (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        # Remove attention TO noised tokens from different segments
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))
```

**How it works:**
```
split_lens =   [50,  256, 256, 40,  256]
attn_modes = ['causal', 'full', 'noise', 'causal', 'noise']

full_and_noise_seq_id: [-1, -1, ..., 1, 1, ..., 2, 2, ..., -1, -1, ..., 4, 4, ...]
                        (causal=-1, full/noise=segment_id)

noise_seq_id:          [-1, -1, ..., -1, -1, ..., 2, 2, ..., -1, -1, ..., 4, 4, ...]
                        (only noise segments get id)

Result:
- Causal tokens: Normal causal attention
- Full tokens: Bidirectional attention within segment
- Noise tokens: Bidirectional within segment, BLOCKED from outside
```

---

## Comparison with Other Approaches

### Standard Diffusion Models

**Approach:** All tokens share the same timestep `t`

```python
# Traditional approach
t = np.random.randn()  # Sample once
timesteps = [t] * num_frames  # All frames use same t
```

**Pros:**
- Simple and efficient
- Easy to implement
- Good for synchronized outputs (e.g., single image)

**Cons:**
- Cannot do autoregressive generation
- Limited to fixed-length sequences
- Train-test mismatch for sequential tasks

---

### Full Diffusion Forcing (Original Paper)

**Approach:** Every frame gets independent `t_i`

```python
# Full diffusion forcing
timesteps = [np.random.randn() for _ in range(num_frames)]
# Each frame has completely independent noise level
```

**Pros:**
- Maximum flexibility (T^N possible configurations)
- Best for autoregressive generation
- Can extend to infinite sequences
- Handles variable-length contexts

**Cons:**
- High training cost (many noise combinations)
- Slower inference (must generate frame-by-frame)
- May lose temporal consistency within short sequences

---

### BAGEL's Hybrid Approach

**Approach:** Independent noise per **segment**, shared within segment

```python
# BAGEL's approach
timesteps = []
for segment in segments:
    t = np.random.randn()  # Sample once per segment
    timesteps.extend([t] * segment_length)
```

**Pros:**
- Balances efficiency and flexibility
- Maintains temporal consistency within segments
- Enables multi-step editing chains
- Efficient batching (process segments together)
- Flexible segment boundaries (video vs. editing)

**Cons:**
- Not as flexible as per-frame noise
- Segments must be defined at data loading time
- Cannot split segments during inference

---

### Progressive Autoregressive Video Diffusion

**Approach:** Progressively increasing noise levels

```python
# Progressive approach
timesteps = [0.1, 0.3, 0.5, 0.7, 0.9, ...]  # Increasing noise
# Later frames are noisier
```

**Pros:**
- Structured noise progression
- Good for future prediction (later = more uncertain)
- Can generate autoregressively

**Cons:**
- Fixed noise schedule
- Cannot handle non-sequential tasks (e.g., editing)
- Less flexible than independent noise

---

## Summary Table

| Approach | Noise Assignment | Autoregressive? | Best For | Training Cost |
|----------|------------------|-----------------|----------|---------------|
| **Standard Diffusion** | Single `t` for all | ❌ No | Single images, T2I | Low |
| **Full Diffusion Forcing** | Independent `t_i` per frame | ✅ Yes | Long videos, flexible generation | High |
| **BAGEL (Hybrid)** | Independent `t` per segment | ✅ Yes | Multi-step editing, short videos | Medium |
| **Progressive AR** | Increasing `t_i` schedule | ✅ Yes | Future prediction, video | Medium |

---

## Key Takeaways

1. **Diffusion forcing enables autoregressive generation** by training models to denoise with noisy context.

2. **BAGEL uses segment-level diffusion forcing** via `split_start`/`split_end` flags, balancing flexibility and efficiency.

3. **Independent noise per edit** prevents error accumulation in multi-step image editing chains.

4. **Video frames share noise** within a sequence for temporal consistency, but different video clips get different noise levels.

5. **`frame_delta` encodes temporal information** in position IDs, helping the model learn motion and spacing.

6. **Attention masking** (`"noise"` mode) prevents information leakage from noised tokens to future context.

7. **Training diversity** comes from random `t ~ N(0,1)` at each `split_start`, exposing the model to all noise levels.

8. **Inference flexibility** allows starting from any noise level, enabling both direct generation and iterative refinement.

---

## References

- **Diffusion Forcing Paper:** Chen et al., "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion", NeurIPS 2024
- **BAGEL Paper:** Deng et al., "Emerging Properties in Unified Multimodal Pretraining", arXiv:2505.14683
- **Progressive AR Video:** Xie et al., "Progressive Autoregressive Video Diffusion Models", arXiv:2410.08151
- **FlexAttention:** PyTorch's flexible attention masking API

---

## Code References

- **Noise assignment:** `data/dataset_base.py:428-431`
- **`split_start`/`split_end` handling:** `data/dataset_base.py:317-319, 460-462`
- **`frame_delta` usage:** `data/dataset_base.py:455-458`
- **Video helper method:** `data/interleave_datasets/interleave_t2i_dataset.py:88-129`
- **Attention mode assignment:** `data/dataset_base.py:448-453`
- **Attention mask creation:** `data/data_utils.py:13-40, 72-103`
