# `split_start` and `split_end`: Complete Control Flow

## Overview

The `split_start` and `split_end` flags control **attention segmentation** in BAGEL's training pipeline. They determine:

1. ✅ **Timestep assignment** - When to sample new noise levels
2. ✅ **Attention segment boundaries** - Which tokens attend to each other
3. ✅ **Attention mode** - How tokens within a segment interact (noise vs full)

## Control Flow in `dataset_base.py`

### 1. Reset Split Length Counter (Lines 317-319)

```python
split_start = item.get('split_start', True)  # Default: True
if split_start:
    curr_split_len = 0  # Start counting a new segment
```

**Effect:** Marks the beginning of a new attention segment.

---

### 2. Assign Timestep for Noised VAE (Lines 428-431)

```python
if item['loss'] == 1:  # Noised VAE token
    if split_start:
        timestep = np.random.randn()  # NEW random noise level
    # else: reuse previous timestep (same segment)
else:
    timestep = float('-inf')  # Clean tokens always t=0
```

**Effect:**
- `split_start=True` → Sample NEW independent noise level `t ~ N(0,1)`
- `split_start=False` → Reuse the timestep from previous token in same segment
- This implements **segment-level diffusion forcing**

---

### 3. Determine Attention Mode (Lines 449-453)

```python
if split_start:
    if item['loss'] == 1 and 'frame_delta' not in item.keys():
        attn_modes.append("noise")  # Invisible to future tokens
    else:
        attn_modes.append("full")   # Bidirectional attention
```

**Effect:** Controls how tokens in this segment interact with others:

| Condition | Attention Mode | Behavior |
|-----------|---------------|----------|
| Noised VAE without `frame_delta` | `"noise"` | Hidden from future context (image editing) |
| Noised VAE with `frame_delta` | `"full"` | Visible to all (video frames) |
| Clean VAE or ViT | `"full"` | Visible to all |
| Text | `"causal"` | Normal causal masking |

**Why the distinction?**
- **Image editing:** Noised VAE should be invisible to future edits (prevents seeing the noised target)
- **Video frames:** Noised VAE can be visible (temporal context, all frames share noise)

---

### 4. End Split Segment (Lines 460-462)

```python
if item.get('split_end', True):  # Default: True
    split_lens.append(curr_split_len)  # Record segment length
    sample_lens += curr_split_len
```

**Effect:** Marks the end of attention segment and records its length for mask creation.

---

## Visual Examples

### Example 1: Text-to-Image (Default Behavior)

**Sequence Plan:**
```python
[
    {'type': 'text', 'loss': 0},         # No split flags set
    {'type': 'vae_image', 'loss': 1},    # No split flags set
]
```

**With Defaults Applied:**
```python
[
    {'type': 'text', 'loss': 0, 'split_start': True, 'split_end': True},
    {'type': 'vae_image', 'loss': 1, 'split_start': True, 'split_end': True},
]
```

**Effects:**

| Element | split_start | Timestep | Attention Mode | Split Length |
|---------|-------------|----------|----------------|--------------|
| Text | ✅ True (default) | N/A | "causal" | 50 tokens |
| Noised VAE | ✅ True (default) | `t = 0.73` (NEW) | "noise" | 256 tokens |

**Result:**
- Text: Segment 1, causal attention, length 50
- Noised VAE: Segment 2, noise mode (invisible to future), length 256, independent noise

---

### Example 2: Multi-Step Image Editing

**Sequence Plan:**
```python
[
    {'type': 'vae_image', 'loss': 0},  # Original clean VAE
    {'type': 'vit_image', 'loss': 0},  # Original ViT
    {'type': 'text', 'loss': 0},       # Instruction
    {'type': 'vae_image', 'loss': 1},  # Edit 1 noised VAE
    {'type': 'vae_image', 'loss': 0},  # Edit 1 clean VAE
    {'type': 'vit_image', 'loss': 0},  # Edit 1 ViT
    {'type': 'text', 'loss': 0},       # Instruction
    {'type': 'vae_image', 'loss': 1},  # Edit 2 noised VAE
]
```

**All default to split_start=True, split_end=True**

**Effects:**

| Element | split_start | Timestep | Attention Mode | Segment |
|---------|-------------|----------|----------------|---------|
| Original Clean VAE | ✅ True | `t = -inf` | "full" | Seg 1 |
| Original ViT | ✅ True | N/A | "full" | Seg 2 |
| Text 1 | ✅ True | N/A | "causal" | Seg 3 |
| Edit 1 Noised VAE | ✅ True | `t = 0.42` (NEW) | "noise" | Seg 4 |
| Edit 1 Clean VAE | ✅ True | `t = -inf` | "full" | Seg 5 |
| Edit 1 ViT | ✅ True | N/A | "full" | Seg 6 |
| Text 2 | ✅ True | N/A | "causal" | Seg 7 |
| Edit 2 Noised VAE | ✅ True | `t = -1.15` (NEW) | "noise" | Seg 8 |

**Result:**
- Each noised VAE gets **independent noise level** (diffusion forcing)
- Each noised VAE is in **"noise" mode** (invisible to future context)
- Each element is its **own attention segment** (all split_start/end = True)

---

### Example 3: Video with Grouped Frames

**Sequence Plan (explicitly set by `_add_video()`):**
```python
[
    {'type': 'vae_image', 'loss': 1, 'split_start': True,  'split_end': False, 'frame_delta': 5},
    {'type': 'vae_image', 'loss': 1, 'split_start': False, 'split_end': False, 'frame_delta': 5},
    {'type': 'vae_image', 'loss': 1, 'split_start': False, 'split_end': False, 'frame_delta': 5},
    {'type': 'vae_image', 'loss': 1, 'split_start': False, 'split_end': True},
]
```

**Effects:**

| Element | split_start | Timestep | Attention Mode | Segment |
|---------|-------------|----------|----------------|---------|
| Frame 0 | ✅ True | `t = -0.73` (NEW) | "full" (has frame_delta) | Seg 1 start |
| Frame 1 | ❌ False | `t = -0.73` (REUSED) | N/A (not split_start) | Seg 1 continue |
| Frame 2 | ❌ False | `t = -0.73` (REUSED) | N/A | Seg 1 continue |
| Frame 3 | ❌ False | `t = -0.73` (REUSED) | N/A | Seg 1 end |

**Result:**
- All frames share **one noise level** `t = -0.73`
- All frames are in **one attention segment** (length = 4 × 256 = 1024 tokens)
- Attention mode is **"full"** (because has frame_delta)
- RoPE positions: [0, 5, 10, 15] (encoded via frame_delta)

---

## Attention Mode Details

### "causal" Mode (Text)

**Applied to:** Text tokens

**Behavior:**
```python
# From data_utils.py:91-92
if attn_mode == "causal":
    attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s)).tril()
```

**Attention Pattern:**
```
Token:  0  1  2  3
0:     [✓  ✗  ✗  ✗]
1:     [✓  ✓  ✗  ✗]
2:     [✓  ✓  ✓  ✗]
3:     [✓  ✓  ✓  ✓]
```

Each token can only attend to itself and previous tokens (autoregressive).

---

### "full" Mode (Clean VAE, ViT, Video Frames)

**Applied to:**
- Clean VAE tokens
- ViT tokens
- Noised VAE tokens with `frame_delta` (video)

**Behavior:**
```python
# From data_utils.py:93-95
else:  # 'full' or 'noise'
    attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
attention_mask[csum:csum + s, :csum] = 1  # Attend to all past
```

**Attention Pattern:**
```
Token:  0  1  2  3
0:     [✓  ✓  ✓  ✓]  + all past tokens
1:     [✓  ✓  ✓  ✓]  + all past tokens
2:     [✓  ✓  ✓  ✓]  + all past tokens
3:     [✓  ✓  ✓  ✓]  + all past tokens
```

Bidirectional attention within segment, can attend to all previous segments.

---

### "noise" Mode (Noised VAE for Editing)

**Applied to:** Noised VAE tokens without `frame_delta` (image editing)

**Behavior:**
```python
# Phase 1: Create full attention within segment
attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))

# Phase 2: BLOCK attention TO these tokens from others
# From data_utils.py:96-98
if attn_mode == "noise":
    attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
    attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
```

**Attention Pattern:**
```
Context tokens → [✗  ✗  ✗  ✗]  Cannot see noised VAE
Noised VAE:
0:             [✓  ✓  ✓  ✓]  Can see itself
1:             [✓  ✓  ✓  ✓]
2:             [✓  ✓  ✓  ✓]
3:             [✓  ✓  ✓  ✓]
Future tokens → [✗  ✗  ✗  ✗]  Cannot see noised VAE
```

**Key Property:** Noised VAE tokens are **isolated islands** - they can attend to themselves, but no other tokens can attend to them.

**Why?** During training, the noised VAE is the **generation target**. Future context (like the next edit's clean VAE) shouldn't be able to "peek" at the noisy target.

---

## Summary Table

| Flag | Default | Controls | Effect |
|------|---------|----------|--------|
| `split_start` | `True` | 1. Segment start<br>2. Timestep sampling<br>3. Attention mode | - Resets split counter<br>- Samples new `t ~ N(0,1)` for noised VAE<br>- Determines "noise" vs "full" mode |
| `split_end` | `True` | Segment end | Records segment length for attention mask |

---

## Key Insights

1. **Default behavior = independent segments**
   - Everything gets split_start=True and split_end=True by default
   - Each element is its own attention segment
   - Each noised VAE gets independent noise (diffusion forcing)

2. **Video explicitly overrides to group frames**
   - Only first frame has split_start=True
   - Only last frame has split_end=True
   - All frames share one timestep (temporal consistency)

3. **Attention mode depends on context**
   - Image editing: Noised VAE gets "noise" mode (hidden)
   - Video: Noised VAE gets "full" mode (visible)
   - Distinction via `frame_delta` presence

4. **split_start does NOT directly control RoPE**
   - RoPE position IDs are managed separately
   - Text: Sequential increment
   - Images: All tokens in image share one position
   - Video: Positions jump by `frame_delta`

5. **Segments enable efficient batching**
   - Multiple related tokens processed together
   - Attention masks computed per segment
   - FlexAttention optimizes sparse patterns

---

## Code References

- **split_start handling:** `data/dataset_base.py:317-319`
- **Timestep assignment:** `data/dataset_base.py:428-431`
- **Attention mode:** `data/dataset_base.py:449-453`
- **split_end handling:** `data/dataset_base.py:460-462`
- **Attention mask creation:** `data/data_utils.py:72-103`
- **Video grouping:** `data/interleave_datasets/interleave_t2i_dataset.py:88-129`
- **Default behavior:** All dataset classes that use `_add_image()` and `_add_text()`
