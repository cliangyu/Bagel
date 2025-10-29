# BAGEL Training Implementation Details

This document maps the paper's description to the actual code implementation, focusing on the interleaved multimodal generation system.

---

## Three-Token System for Each Image

**Paper Description:**
> During training, an interleaved multimodal generation sample may contain multiple images. For each image, we prepare three sets of visual tokens:
> - **Noised VAE tokens**: VAE latents corrupted with diffusion noise, used exclusively for Rectified-Flow training; the MSE loss is computed on this set.
> - **Clean VAE tokens**: the original (noise-free) latents, which serve as conditioning when generating subsequent image or text tokens.
> - **ViT tokens**: obtained from the SigLIP encoder, which help to unify the input format across interleaved generation and understanding data and, empirically, to boost interleaved-generation quality.

### Implementation

**File:** `data/interleave_datasets/edit_dataset.py` (lines 29-72)

The `_add_image()` method supports creating all three token types:

```python
def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
    """
    need_loss: Add noised VAE tokens with MSE loss
    need_vae:  Add clean VAE tokens (conditioning only, no loss)
    need_vit:  Add ViT tokens (conditioning only, no loss)
    """
```

**Example: Multi-step editing sequence**

From `edit_dataset.py:29-35` (Original image):
```python
data = self._add_image(
    data,
    original_image,
    need_loss=False,  # No noised VAE (not generating this)
    need_vae=True,    # ✓ Clean VAE tokens for conditioning
    need_vit=True,    # ✓ ViT tokens for understanding
)
```

From `edit_dataset.py:57-63` (Intermediate edited image):
```python
data = self._add_image(
    data,
    edited_image,
    need_loss=True,   # ✓ Noised VAE tokens with MSE loss
    need_vae=True,    # ✓ Clean VAE tokens for next image
    need_vit=True,    # ✓ ViT tokens for understanding
)
```

From `edit_dataset.py:65-71` (Final edited image):
```python
data = self._add_image(
    data,
    final_image,
    need_loss=True,   # ✓ Noised VAE tokens with MSE loss
    need_vae=False,   # No need for clean VAE (end of sequence)
    need_vit=False,   # No need for ViT (end of sequence)
)
```

**Token Sequence Example:**
```
[Clean VAE₀] → [ViT₀] → [Text] → [Noised VAE₁] → [Clean VAE₁] → [ViT₁] → [Text] → [Noised VAE₂]
     ↑            ↑                      ↑              ↑             ↑                    ↑
  Condition    Condition            MSE Loss      Condition     Condition           MSE Loss
```

---

## Timestep Assignment and Noise Scheduling

**Paper Description:**
> t is the noise timestep and t=0 means no noise.

### Implementation

**File:** `data/dataset_base.py` (lines 426-433)

```python
if item['loss'] == 1:  # Noised VAE tokens (need MSE loss)
    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
    if split_start:
        timestep = np.random.randn()  # Random noise level for diffusion forcing
    else:
        timestep = <inherited from previous split>  # Same noise level within group
else:  # Clean VAE tokens (no loss)
    timestep = float('-inf')  # Special marker for t=0 (no noise)

sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
```

**Key Points:**
1. **Clean VAE tokens:** `timestep = float('-inf')` → Indicates t=0 (no noise)
2. **Noised VAE tokens:** `timestep = np.random.randn()` → Random Gaussian noise level
3. **Grouping:** `split_start=True` assigns new timestep; `split_start=False` inherits timestep

---

## Attention Patterns

**Paper Description:**
> (a) During interleaved image-text generation, each image attends exclusively to the clean (noise-free) VAE and ViT tokens of preceding images (if present).
> (b) For interleaved multi-image or video clip generation, we adopt the diffusion forcing strategy, conditioning each image on noisy representations of preceding images.

### Implementation

**File:** `data/dataset_base.py` (lines 449-458)

```python
# Determine attention mode for VAE image tokens
if split_start:
    if item['loss'] == 1 and 'frame_delta' not in item.keys():
        attn_modes.append("noise")  # Case (a): Noised VAE with restricted attention
    else:
        attn_modes.append("full")   # Case (b): Clean VAE / video frames with full attention
```

**File:** `data/data_utils.py` (lines 72-103) - Attention mask generation

```python
def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    attn_modes: ['causal', 'full', 'noise'] for each split

    'causal': Text tokens use causal attention (attend to past only)
    'full':   Image tokens use full attention within image + all past tokens
    'noise':  Noised VAE tokens - OTHER tokens CANNOT attend to them
    """

    # Phase 1: Build base attention (causal or full)
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s)).tril()
            attention_mask[csum:csum + s, :csum] = 1  # Attend to all past
        else:  # 'full' or 'noise'
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1  # Attend to all past

    # Phase 2: Block attention TO noised VAE tokens
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
            # Result: Only the noised VAE tokens themselves can attend to each other
            #         All other tokens (including future images) CANNOT attend to noised VAE
```

**Attention Pattern Visualization:**

```
Sequence: [Text₁] [Clean VAE₀] [ViT₀] [Text₂] [Noised VAE₁] [Text₃] [Clean VAE₁] [ViT₁]

Query Token       | Can Attend To
------------------|--------------------------------------------------------
Text₁             | Text₁ (causal)
Clean VAE₀        | Text₁, Clean VAE₀ (full attention)
ViT₀              | Text₁, Clean VAE₀, ViT₀ (full attention)
Text₂             | Text₁, Clean VAE₀, ViT₀, Text₂ (causal)
Noised VAE₁       | Text₁, Clean VAE₀, ViT₀, Text₂, Noised VAE₁ (full, but blocked from others)
Text₃             | Text₁, Clean VAE₀, ViT₀, Text₂, Text₃ (CANNOT see Noised VAE₁)
Clean VAE₁        | Text₁, Clean VAE₀, ViT₀, Text₂, Text₃, Clean VAE₁ (CANNOT see Noised VAE₁)
ViT₁              | Text₁, Clean VAE₀, ViT₀, Text₂, Text₃, Clean VAE₁, ViT₁ (CANNOT see Noised VAE₁)
```

**Key Point:** Noised VAE tokens are "isolated" - subsequent tokens act as if they don't exist.

---

## Diffusion Forcing: Independent Noise Levels per Image

**Paper Description:**
> For interleaved multi-image generation, we adopt the diffusion forcing strategy, which adds independent noise levels to different images and conditions each image on noisy representations of preceding images.

### Implementation

**File:** `data/dataset_base.py` (lines 428-429)

```python
if split_start:
    timestep = np.random.randn()  # NEW independent noise level
else:
    timestep = <same as previous>  # Same noise level within group
```

**Example: Three consecutive images in a video**

```python
# Image 1
{'type': 'vae_image', 'loss': 1, 'split_start': True,  'split_end': False}  # t₁ = 0.73
# Image 2
{'type': 'vae_image', 'loss': 1, 'split_start': False, 'split_end': False}  # t₂ = 0.73 (same group)
# Image 3
{'type': 'vae_image', 'loss': 1, 'split_start': True,  'split_end': True}   # t₃ = -0.42 (new noise)
```

**Result:**
- Images 1-2: Same noise level t=0.73 (grouped)
- Image 3: Different noise level t=-0.42 (diffusion forcing)

---

## Random Grouping for Generation Consistency

**Paper Description:**
> Additionally, to enhance generation consistency, we randomly group consecutive images following [17] and apply full attention within each group. The noise level is the same inside each group.

### Implementation

**File:** `data/data_utils.py` (lines 106-115)

```python
def split_integer_exp_decay(S, ng_sample_decay=1.0):
    """
    Randomly splits S images into N groups with exponential decay preference.

    Args:
        S: Total number of images
        ng_sample_decay: Decay factor (1.0 = uniform, <1.0 = prefer fewer groups)

    Returns:
        result: [group1_size, group2_size, ..., groupN_size]
        cumsum: [0, group1_end, group2_end, ..., S]

    Example:
        S=5 images → may return [2, 3] meaning:
            Group 1: Images 0-1 (same noise level, full attention)
            Group 2: Images 2-4 (same noise level, full attention)
    """
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)  # Uniform: 1 to S groups
    else:
        # Exponential decay: prefer fewer groups
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]

    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum
```

**Implementation in Video Dataset:**

**File:** `data/interleave_datasets/interleave_t2i_dataset.py` (lines 92-108)

```python
def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
    # Groups are determined BEFORE calling this method
    # Then each frame gets:
    for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
        current_sequence_plan = {
            'type': 'vae_image',
            'loss': 1 if need_loss else 0,
            'split_start': idx == 0,              # First frame starts new split
            'split_end': idx == len(frames) - 1,  # Last frame ends split
            'frame_delta': frame_indexes[idx + 1] - frame_idx  # Temporal info
        }
```

**Grouping Effect:**

Without grouping (each image independent):
```
[VAE₁ t=0.5] → [VAE₂ t=-0.2] → [VAE₃ t=0.8] → [VAE₄ t=-0.4]
    noise        noise            noise           noise
```

With grouping (2 groups: [0-1], [2-3]):
```
[VAE₁ t=0.5] → [VAE₂ t=0.5] → [VAE₃ t=-0.2] → [VAE₄ t=-0.2]
    noise        FULL             noise           FULL
                 (same noise)                     (same noise)
```

**Benefit:** Images in the same group have consistent noise levels, improving temporal coherence in video generation.

**Note:** `split_integer_exp_decay` is defined but not actively used in the current codebase. Grouping is instead controlled by `split_start` and `split_end` flags in the sequence plan.

---

## Classifier-Free Guidance (CFG) Dropout

**Paper Description:**
> To enable classifier-free guidance in interleaved inference, we randomly drop text, ViT, and clean VAE tokens with probabilities 0.1, 0.5, and 0.1, respectively.

### Implementation

**File:** `train/pretrain_unified_navit.py` (lines 161-172)

```python
@dataclass
class ModelArguments:
    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,  # Note: Paper says 0.5, default is 0.3
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,  # Note: Paper says 0.1, default is 0.3
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
```

**File:** `data/dataset_base.py` (lines 323-402) - Actual dropout logic

```python
# Text dropout
if item['type'] == 'text':
    text_ids = text_ids_list.pop(0)
    if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
        continue  # Skip this text element (dropped for CFG)
    # ... otherwise add to sequence

# ViT dropout
elif item['type'] == 'vit_image':
    image_tensor = image_tensor_list.pop(0)
    if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
        curr_rope_id += 1
        continue  # Skip this ViT image (dropped for CFG)
    # ... otherwise add to sequence

# Clean VAE dropout
elif item['type'] == 'vae_image':
    image_tensor = image_tensor_list.pop(0)
    if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
        curr_rope_id += 1
        continue  # Skip this VAE image (dropped for CFG)
    # ... otherwise add to sequence
```

**CFG Dropout Matrix:**

| Token Type | Paper Probability | Default Config | Controlled by `enable_cfg` |
|------------|-------------------|----------------|----------------------------|
| Text | 0.1 | 0.1 | `text_cond_dropout_prob` |
| ViT | 0.5 | 0.3 | `vit_cond_dropout_prob` |
| Clean VAE | 0.1 | 0.3 | `vae_cond_dropout_prob` |
| **Noised VAE** | **Never dropped** | **N/A** | **`enable_cfg=0` always** |

**Key Point:** Only tokens with `enable_cfg=1` can be dropped. Noised VAE tokens (for generation) always have `enable_cfg=0`.

**Example Sequence with Dropout:**

Original:
```
[Text₁] [Clean VAE₀] [ViT₀] [Text₂] [Noised VAE₁]
```

After CFG dropout (with probabilities applied):
```
[Text₁] [✗ Clean VAE₀ dropped] [ViT₀] [✗ Text₂ dropped] [Noised VAE₁]
```

Result: Model must generate Noised VAE₁ with less conditioning (only Text₁ and ViT₀ remain).

---

## FlexAttention Implementation

**Paper Description:**
> We implement the generalized causal attention with PyTorch FlexAttention, achieving a ~2× speed-up over naive scaled-dot-product attention.

### Implementation

**File:** `data/data_utils.py` (lines 13-40)

```python
from torch.nn.attention.flex_attention import or_masks, and_masks

def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    """
    Creates FlexAttention-compatible mask combining:
    - Causal masking for text
    - Full attention for images
    - Noise masking for noised VAE tokens
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        # Full attention within same sequence ID
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & \
               (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        # Prevent attending to noised VAE tokens from different sequences
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        # Separate samples in packed batch
        return document_id[q_idx] == document_id[kv_idx]

    # Combine all masks
    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)
```

**File:** `data/dataset_base.py` (lines 467-473) - Selection of attention backend

```python
# Prepare attention mask
if not self.use_flex:
    # Traditional attention: Create explicit mask matrix
    sequence_status['nested_attention_masks'].append(
        prepare_attention_mask_per_sample(split_lens, attn_modes)
    )
else:
    # FlexAttention: Store split info for dynamic mask generation
    sequence_status['split_lens'].extend(split_lens)
    sequence_status['attn_modes'].extend(attn_modes)
```

**Key Benefit:** FlexAttention computes attention masks on-the-fly during forward pass, avoiding large pre-computed mask matrices.

---

## Inference KV Cache Management

**Paper Description:**
> During inference, the generalized causal structure allows us to cache key-value (KV) pairs of the generated multimodal context and thus accelerate multimodal decoding. Only the KV pairs of clean VAE tokens and ViT tokens are stored; once an image is fully generated, the corresponding noised VAE tokens in the context are replaced by their clean counterparts.

### Implementation Location

**Note:** The current codebase focuses on training. Inference KV cache management is likely implemented in the model forward pass, not in the data loading pipeline.

**Expected Implementation (in model code, not dataset):**

```python
# Pseudo-code for inference
def generate_interleaved():
    kv_cache = []

    for step in generation_steps:
        if step.type == 'text':
            hidden = model.forward_text(step.tokens, kv_cache)
            kv_cache.append(hidden)  # Cache text KV

        elif step.type == 'vit_image':
            hidden = model.forward_vit(step.image, kv_cache)
            kv_cache.append(hidden)  # Cache ViT KV

        elif step.type == 'generate_image':
            # Generate with noised VAE tokens (not cached)
            noised_latents = model.diffusion_forward(kv_cache)

            # Convert to clean latents
            clean_latents = vae.encode(generated_image)
            hidden_clean = model.forward_vae(clean_latents, kv_cache)
            kv_cache.append(hidden_clean)  # Cache ONLY clean VAE KV

            # Noised VAE KV is discarded (not added to cache)
```

**Key Point:** This ensures that future generation steps only see clean representations of past images, matching the training attention pattern.

---

## Summary: Paper → Code Mapping

| Paper Concept | Implementation File | Key Lines | Function/Method |
|---------------|---------------------|-----------|-----------------|
| Three-token system | `edit_dataset.py` | 36-63 | `_add_image(need_loss, need_vae, need_vit)` |
| Timestep assignment (t=0) | `dataset_base.py` | 428-433 | `float('-inf')` for clean, `np.random.randn()` for noised |
| Attention patterns | `data_utils.py` | 72-103 | `prepare_attention_mask_per_sample()` |
| Diffusion forcing | `dataset_base.py` | 428-429 | `split_start=True` → new timestep |
| Random grouping | `data_utils.py` | 106-115 | `split_integer_exp_decay()` (defined but not actively used) |
| CFG dropout (0.1, 0.5, 0.1) | `dataset_base.py` | 323-402 | Dropout in sequence packing |
| FlexAttention | `data_utils.py` | 13-40 | `create_sparse_mask()` |
| KV cache management | Model code (not in dataset) | N/A | Inference-only feature |

---

## Dataset Type Usage Matrix

| Feature | `t2i_pretrain` | `unified_edit` | `vlm_sft` |
|---------|----------------|----------------|-----------|
| Noised VAE tokens | ✓ (generated image) | ✓ (edited images) | ✗ |
| Clean VAE tokens | ✗ | ✓ (conditioning) | ✗ |
| ViT tokens | ✗ | ✓ (understanding) | ✓ (all images) |
| Text CFG dropout | ✓ (0.1) | ✓ (0.1) | ✓ (0.1) |
| VAE CFG dropout | ✗ (generated, not dropped) | ✓ (0.3 for clean) | ✗ |
| ViT CFG dropout | ✗ | ✓ (0.3) | ✓ (0.3) |
| Attention mode | Causal text → Noise image | Causal text + Full/Noise images | Causal text → Full images |
| Diffusion forcing | Single image (N/A) | Multi-step editing (✓) | Not used |
| Random grouping | Not applicable | Video editing (potential) | Video understanding (frames) |

---

## Key Takeaways

1. **Three-token system enables flexible conditioning:**
   - Noised VAE: Training target (MSE loss)
   - Clean VAE: Past image conditioning for generation
   - ViT: Understanding/unified representation

2. **Attention isolation protects generation:**
   - Noised VAE tokens are "invisible" to future tokens
   - Ensures model learns to condition on clean representations only

3. **Diffusion forcing improves multi-image coherence:**
   - Each image gets independent noise level (`split_start=True`)
   - Or shares noise within groups (`split_start=False`)

4. **CFG dropout enables guidance at inference:**
   - Training with random dropout → learned unconditional distribution
   - Inference with controlled dropout → classifier-free guidance

5. **FlexAttention optimizes complex attention patterns:**
   - Combines causal, full, and noise masking efficiently
   - ~2× speedup over naive attention
