# Flow Matching and VAE Analysis in BAGEL

This document provides a comprehensive analysis of how flow matching training/inference and VAE encoding/decoding work in the BAGEL codebase.

---

## Table of Contents
1. [Flow Matching Training](#flow-matching-training)
2. [Flow Matching Inference](#flow-matching-inference)
3. [Classifier-Free Guidance (CFG)](#classifier-free-guidance-cfg)
4. [Key Architectural Components](#key-architectural-components)
5. [VAE Training and Inference](#vae-training-and-inference)
6. [Data Flow Summary](#data-flow-summary)
7. [Hyperparameters Reference](#hyperparameters-reference)

---

## Flow Matching Training

### Core Algorithm
**Location:** `modeling/bagel/bagel.py:181-229`

Flow matching in BAGEL uses **velocity prediction** to learn the trajectory between clean images and noise.

#### 1. Interpolation Path

The training creates a linear path between clean latent images and Gaussian noise:

```python
# Lines 190-193
noise = torch.randn_like(packed_latent_clean)
packed_timesteps = torch.sigmoid(packed_timesteps)  # Normalize timesteps
packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
```

**Formula:** `x_t = (1 - t) * x_0 + t * x_1`
- `x_0`: Clean latent image
- `x_1`: Gaussian noise
- `t`: Timestep (normalized between 0 and 1)

#### 2. Timestep Sampling

**Location:** `data/dataset_base.py:429-433`

```python
if item['loss'] == 1:
    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
    if split_start:
        timestep = np.random.randn()  # Standard normal distribution
else:
    timestep = float('-inf')

sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
```

**Key Details:**
- Timesteps sampled from standard normal: `np.random.randn()`
- Each image gets 1 random timestep for all its latent tokens
- Non-loss images get timestep of `-inf` (ignored in loss)
- Timestep shifting applied: `t_shifted = shift * t / (1 + (shift - 1) * t)`

#### 3. Velocity Target

**Location:** `modeling/bagel/bagel.py:220`

```python
target = noise - packed_latent_clean  # v_t = x_1 - x_0
```

The model learns to predict the **velocity field** pointing from data to noise:
- `v_t = dx_t/dt = x_1 - x_0`
- This represents the direction and magnitude to move from clean data toward noise

#### 4. Loss Function

**Location:** `modeling/bagel/bagel.py:219-222`

```python
packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
target = noise - packed_latent_clean
has_mse = packed_timesteps > 0
mse = (packed_mse_preds - target[has_mse]) ** 2
```

**Simple MSE Loss:** `L = ||predicted_velocity - target_velocity||²`

#### 5. Training Configuration

**Location:** `train/pretrain_unified_navit.py`

```python
timestep_shift: float = 1.0      # Line 331-332: Controls timestep distribution scaling
mse_weight: float = 1.0          # Line 334-336: Loss weight for generation branch
ce_weight: float = 1.0           # Line 338-340: Loss weight for understanding branch
```

---

## Flow Matching Inference

### ODE Solving Pipeline
**Location:** `modeling/bagel/bagel.py:644-754`

Inference solves an Ordinary Differential Equation (ODE) to generate images by following the learned velocity field from noise to data.

#### 1. Initial State

Start with pure Gaussian noise: `x_t ~ N(0, I)` at `t=1`

#### 2. Timestep Schedule

**Location:** `modeling/bagel/bagel.py:693-696`

```python
timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
dts = timesteps[:-1] - timesteps[1:]
timesteps = timesteps[:-1]
```

**Parameters:**
- `num_timesteps`: Number of denoising steps (default 24-50)
- `timestep_shift`: Adjusts timestep distribution (default 3.0 for inference)

#### 3. Euler Integration Loop

**Location:** `modeling/bagel/bagel.py:700-746`

```python
for i, t in enumerate(timesteps):
    # Predict velocity at current state
    v_t = self._forward_flow(
        packed_sequence=packed_sequence,
        timestep=t,
        ...
    )

    # Integration step: move along velocity field
    x_t = x_t - v_t.to(x_t.device) * dts[i]
```

**Simple Euler Method:**
- `x_{t+Δt} = x_t + v_t * Δt`
- Moves from `t=1` (noise) toward `t=0` (data)

#### 4. Velocity Prediction

**Location:** `modeling/bagel/bagel.py:757-907` (`_forward_flow()` method)

The model predicts velocity fields conditioned on:
- **Current latent state** `x_t`
- **Timestep embedding** (sinusoidal + MLP)
- **Text conditioning** (prompt embeddings)
- **Image conditioning** (ViT features)

```python
# Lines 801-806: Embed current state with timestep information
packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
packed_timestep_embeds = self.time_embedder(timestep)
x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
packed_sequence[packed_vae_token_indexes] = x_t

# Line 832: Extract velocity prediction
v_t = self.llm2vae(output.packed_query_sequence)
v_t = v_t[packed_vae_token_indexes]
```

**Pipeline:**
1. Project latent to LLM hidden space: `vae2llm(x_t)`
2. Add timestep and position embeddings
3. Run through LLM backbone with conditioning
4. Project back to latent space: `llm2vae(hidden)`
5. Extract predicted velocity

---

## Classifier-Free Guidance (CFG)

### Dual-Guidance Strategy
**Location:** `modeling/bagel/bagel.py:835-906`

CFG enables stronger conditioning control by running **three parallel forward passes**:

#### 1. Three Forward Passes

```python
# 1. Full conditioning (text + image)
v_t = model(x_t, t, text=prompt, image=reference)

# 2. Text-only conditioning (if cfg_text_scale > 1.0)
cfg_text_v_t = model(x_t, t, text=prompt, image=None)

# 3. Image-only conditioning (if cfg_img_scale > 1.0)
cfg_img_v_t = model(x_t, t, text=None, image=reference)
```

#### 2. Guidance Combination

**Location:** `modeling/bagel/bagel.py:885-902`

```python
# Text guidance
v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

# Image guidance (if enabled)
if cfg_img_scale > 1.0:
    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
else:
    v_t_ = v_t_text_

# Renormalization to prevent magnitude explosion
scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
v_t = v_t_ * scale
```

**Interpretation:**
- `cfg_text_scale > 1.0`: Amplifies text prompt influence
- `cfg_img_scale > 1.0`: Amplifies image conditioning influence
- Renormalization keeps velocity magnitude stable

#### 3. CFG Parameters

**Location:** `README.md:93-104`

```python
cfg_text_scale: float = 4.0      # Range: 1.0-8.0 (text guidance strength)
cfg_img_scale: float = 1.5       # Range: 1.0-2.0 (image preservation)
cfg_interval: Tuple = (0.4, 1.0) # When to apply guidance during denoising
cfg_renorm_type: str = "global"  # "global" | "channel" | "text_channel"
cfg_renorm_min: float = 0.0      # Minimum scale clamp value
```

**CFG Interval:** Controls when guidance is active during sampling
- `[0.4, 1.0]`: Apply CFG from timestep 1.0 down to 0.4
- Outside this range, use unconditional velocity

---

## Key Architectural Components

### 1. TimestepEmbedder

**Location:** `modeling/bagel/modeling_utils.py:74-110`

Embeds continuous timesteps into the model's hidden dimension.

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Sinusoidal positional embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
```

**Process:**
1. Sinusoidal encoding (like Transformer positional encoding)
2. MLP projection to hidden dimension
3. Added to latent embeddings during forward pass

### 2. VAE-LLM Projections

**Location:** `modeling/bagel/bagel.py:68-78`

Projects between VAE latent space and LLM hidden space.

```python
self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)  # Line 76
self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)  # Line 77

# Initialize llm2vae to zero (line 98)
nn.init.constant_(self.llm2vae.weight, 0)
nn.init.constant_(self.llm2vae.bias, 0)
```

**Dimensions:**
- `patch_latent_dim`: 2 × 2 × 16 = 64 (patch of VAE latents)
- `hidden_size`: LLM hidden dimension (e.g., 4096)

**Zero Initialization:** `llm2vae` starts at zero, allowing the model to gradually learn to generate images.

### 3. Latent Position Embedding

**Location:** `modeling/bagel/modeling_utils.py:127-144`

2D sinusoidal positional embeddings for latent patches (from DiT).

```python
class LatentPositionalEmbedding(nn.Module):
    """2D positional embeddings for spatial latent patches"""
    ...
```

Encodes spatial position of each latent patch in the image grid.

### 4. Latent Patchification

**Location:** `modeling/bagel/bagel.py:182-197`

Converts VAE latents into discrete tokens for the LLM.

```python
# Extract and reshape latent tokens from VAE output
for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
    latent = latent[:, :h * p, :w * p].reshape(...)
    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
    packed_latent.append(latent)

packed_latent_clean = torch.cat(packed_latent, dim=0)
```

**Process:**
1. VAE latents: `[C, H/8, W/8]` where `C=16`
2. Split into 2×2 patches
3. Flatten each patch: `2 × 2 × 16 = 64`
4. Each patch becomes one "visual token"

---

## VAE Training and Inference

### VAE Model Details

**Location:** `modeling/autoencoder.py:290-326`

**Model:** Flux VAE (from Black Forest Labs)

**Configuration:**
```python
resolution: 256
in_channels: 3
out_ch: 3
z_channels: 16              # Latent channels
ch: 128                     # Base channel count
ch_mult: [1, 2, 4, 4]       # Channel multipliers per stage
downsample_factor: 8        # Spatial downsampling
scale_factor: 0.3611        # Latent normalization
shift_factor: 0.1159        # Latent centering
```

**Architecture:**
- **Encoder:** 3-channel RGB → 16-channel latent (8× downsampled)
- **Decoder:** 16-channel latent → 3-channel RGB (8× upsampled)
- **DiagonalGaussian:** Samples from learned mean/variance

### VAE Initialization and Loading

**Location:** `train/pretrain_unified_navit.py:500-504`

```python
if training_args.visual_gen:
    vae_model, vae_config = load_ae(
        local_path=os.path.join(model_args.model_path, "ae.safetensors")
        if training_args.finetune_from_hf else model_args.vae_path
    )
```

**Default Path:** `flux/vae/ae.safetensors`

### VAE Freezing (Critical!)

**Location:** `train/pretrain_unified_navit.py:541-543`

```python
if training_args.freeze_vae and training_args.visual_gen:
    for param in vae_model.parameters():
        param.requires_grad = False
```

**Status:** ✅ **VAE IS FROZEN BY DEFAULT**
- `freeze_vae: bool = field(default=True, ...)`
- Set to evaluation mode: `vae_model.to(device).eval()`
- **No gradient updates through the VAE**

### VAE Encoder - Training Usage

**Location:** `train/pretrain_unified_navit.py:683-686`

```python
with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
    if training_args.visual_gen:
        with torch.no_grad():  # ← CRITICAL: No gradient computation
            data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
```

**When:** Before each training iteration
**Context:** `torch.no_grad()` - no gradients computed
**Input:** `padded_images` - batch of RGB images `[B, 3, H, W]`
**Output:** `padded_latent` - latent codes `[B, 16, H/8, W/8]`

**VAE Encoder Forward:**
```python
# modeling/autoencoder.py:315-318
def encode(self, x: Tensor) -> Tensor:
    z = self.reg(self.encoder(x))              # DiagonalGaussian sampling
    z = self.scale_factor * (z - self.shift_factor)
    return z
```

### VAE Decoder - Training Usage

**During Training:** ❌ **NOT CALLED**

The decoder is not used during training:
- Flow matching learns to predict latent trajectories
- Loss computed directly in latent space
- Decoder reconstruction only happens at inference

### VAE Encoder - Inference Usage

**Location:** `modeling/bagel/bagel.py:512`

```python
padded_latent = vae_model.encode(padded_images)
```

**Context:** Encoding reference/condition images during generation
**File:** `inferencer.py:70-79` - Interactive generation interface

### VAE Decoder - Inference Usage

**Location:** `inferencer.py:174-185`

```python
def decode_image(self, latent, image_shape):
    H, W = image_shape
    h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

    # Reshape latent patches back to spatial grid
    latent = latent.reshape(1, h, w, self.model.latent_patch_size,
                           self.model.latent_patch_size, self.model.latent_channel)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, self.model.latent_channel,
                           h * self.model.latent_patch_size,
                           w * self.model.latent_patch_size)

    # Decode to pixel space
    image = self.vae_model.decode(latent)  # ← DECODER CALL

    # Post-process
    image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
    image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
    return image
```

**When:** After flow matching sampling completes
**Input:** Final latent code from ODE solver
**Output:** RGB image in pixel space

**VAE Decoder Forward:**
```python
# modeling/autoencoder.py:320-322
def decode(self, z: Tensor) -> Tensor:
    z = z / self.scale_factor + self.shift_factor
    return self.decoder(z)
```

### Image Encoding/Decoding Process

**Encoding Pipeline:**
```
RGB Image [3 × H × W]
    ↓ (VAE Encoder - 8× downsample)
Latent [16 × H/8 × W/8]
    ↓ (DiagonalGaussian sampling)
Latent codes [16 × H/8 × W/8]
    ↓ (Apply scaling)
z * 0.3611 - 0.1159
```

**Decoding Pipeline:**
```
Latent codes [16 × H/8 × W/8]
    ↓ (Reverse scaling)
z / 0.3611 + 0.1159
    ↓ (VAE Decoder - 8× upsample)
RGB Image [3 × H × W]
```

### Gradient Flow Summary

| Component | Training | Inference | Trainable |
|-----------|----------|-----------|-----------|
| **VAE Encoder** | ✓ (no_grad) | ✓ | ❌ No |
| **VAE Decoder** | ❌ Not called | ✓ | ❌ No |
| **DiagonalGaussian** | ✓ (no_grad) | ✓ | ❌ No |
| **vae2llm Projection** | ✓ (with_grad) | ✓ | ✅ Yes |
| **llm2vae Projection** | ✓ (with_grad) | ✓ | ✅ Yes |
| **LLM Backbone** | ✓ (with_grad) | ✓ | ✅ Yes |

**Key Insight:** The VAE acts as a **frozen tokenizer** during training, converting images to latent space for the LLM to predict flow trajectories.

---

## Data Flow Summary

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING DATA FLOW                       │
└─────────────────────────────────────────────────────────────┘

Raw Images (batch)
    ↓
padded_images tensor [B, 3, H, W]
    ↓ [torch.no_grad()]
┌───────────────────┐
│  VAE.encode()     │ ← FROZEN, NO GRAD
└───────────────────┘
    ↓
padded_latent [B, 16, H/8, W/8]
    ↓
Patchify into tokens [B, num_patches, 64]
    ↓
Sample timestep t ~ N(0,1), apply sigmoid & shift
    ↓
Interpolate: x_t = (1-t)*x_0 + t*noise
    ↓ [torch.grad()]
┌───────────────────┐
│  vae2llm()        │ ← TRAINABLE
└───────────────────┘
    ↓
Add timestep + position embeddings
    ↓
┌───────────────────┐
│  LLM Forward      │ ← TRAINABLE
└───────────────────┘
    ↓
┌───────────────────┐
│  llm2vae()        │ ← TRAINABLE
└───────────────────┘
    ↓
Predicted velocity v_t
    ↓
MSE Loss: ||v_t - (noise - x_0)||²
    ↓
Backward pass (only through LLM & projections)
```

### Inference Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE DATA FLOW                       │
└─────────────────────────────────────────────────────────────┘

Reference Images (optional)
    ↓
┌───────────────────┐
│  VAE.encode()     │ ← Condition encoding
└───────────────────┘
    ↓
condition_latents [B, 16, H/8, W/8]


Pure Noise x_t ~ N(0,I) at t=1
    ↓
┌─────────────────────────────────────────────────────────────┐
│              Iterative ODE Solving (Euler Method)            │
│                                                              │
│  for t in [1.0, 0.96, 0.92, ..., 0.04, 0.0]:              │
│      ┌───────────────────┐                                  │
│      │  vae2llm(x_t)     │                                  │
│      └───────────────────┘                                  │
│             ↓                                                │
│      Add timestep + position embeddings                     │
│             ↓                                                │
│      ┌───────────────────┐                                  │
│      │  LLM Forward      │ ← With text/image conditioning   │
│      └───────────────────┘                                  │
│             ↓                                                │
│      ┌───────────────────┐                                  │
│      │  llm2vae()        │                                  │
│      └───────────────────┘                                  │
│             ↓                                                │
│      Predict velocity v_t                                   │
│             ↓                                                │
│      (Optional: Apply CFG with 3 forward passes)            │
│             ↓                                                │
│      x_{t-Δt} = x_t - v_t * Δt                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    ↓
Final latent x_0 [B, 16, H/8, W/8]
    ↓
Unpatchify to spatial grid
    ↓
┌───────────────────┐
│  VAE.decode()     │
└───────────────────┘
    ↓
Generated Image [B, 3, H, W]
    ↓
Post-process: denormalize, clamp, convert to uint8
    ↓
Final RGB Image
```

---

## Hyperparameters Reference

### Training Hyperparameters

**Location:** `train/pretrain_unified_navit.py`

| Parameter | Default | Range | Description | Line |
|-----------|---------|-------|-------------|------|
| `timestep_shift` | 1.0 | 1.0-3.0 | Timestep distribution scaling | 331 |
| `mse_weight` | 1.0 | 0.0-10.0 | Generation loss weight | 334 |
| `ce_weight` | 1.0 | 0.0-10.0 | Understanding loss weight | 338 |
| `freeze_vae` | True | bool | Freeze VAE parameters | 102 |
| `vae_cond_dropout_prob` | 0.3 | 0.0-1.0 | Latent conditioning dropout | 122 |
| `max_latent_size` | 32 | int | Max latent grid patches | 120 |
| `latent_patch_size` | 2 | int | Spatial VAE pixels per patch | 121 |

### Inference Hyperparameters

**Location:** `inferencer.py`

| Parameter | Default | Range | Description | Line |
|-----------|---------|-------|-------------|------|
| `num_timesteps` | 50 | 20-100 | ODE solver steps | 112 |
| `timestep_shift` | 3.0 | 1.0-5.0 | Inference timestep scaling | 113 |
| `cfg_text_scale` | 4.0 | 1.0-8.0 | Text guidance strength | 103 |
| `cfg_img_scale` | 1.5 | 1.0-2.0 | Image guidance strength | 104 |
| `cfg_interval` | [0.4, 1.0] | [0.0-1.0] | CFG application range | 108 |
| `cfg_renorm_type` | "global" | str | CFG renormalization method | - |
| `cfg_renorm_min` | 0.0 | 0.0-1.0 | Min renormalization scale | - |

### VAE Configuration

**Location:** `modeling/autoencoder.py`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `z_channels` | 16 | Latent feature channels |
| `downsample_factor` | 8 | Spatial downsampling ratio |
| `scale_factor` | 0.3611 | Latent normalization scale |
| `shift_factor` | 0.1159 | Latent centering shift |
| `in_channels` | 3 | Input RGB channels |
| `resolution` | 256 | Base resolution |

---

## Key Files Reference

| Component | File | Lines |
|-----------|------|-------|
| **Flow Matching Training** | `modeling/bagel/bagel.py` | 101-229 |
| **Flow Matching Inference** | `modeling/bagel/bagel.py` | 644-754 |
| **Velocity Prediction** | `modeling/bagel/bagel.py` | 757-907 |
| **CFG Implementation** | `modeling/bagel/bagel.py` | 835-906 |
| **TimestepEmbedder** | `modeling/bagel/modeling_utils.py` | 74-110 |
| **LatentPositionalEmbedding** | `modeling/bagel/modeling_utils.py` | 127-144 |
| **VAE Model** | `modeling/autoencoder.py` | 290-326 |
| **VAE Loading** | `modeling/autoencoder.py` | 339-360 |
| **Training Loop** | `train/pretrain_unified_navit.py` | 658-872 |
| **VAE Freezing** | `train/pretrain_unified_navit.py` | 541-543 |
| **VAE Encoding (Training)** | `train/pretrain_unified_navit.py` | 686 |
| **Timestep Sampling** | `data/dataset_base.py` | 429-433 |
| **Inferencer** | `inferencer.py` | 98-185 |

---

## Summary

### Flow Matching
- **Method:** Velocity prediction flow matching
- **Training:** Learns to predict velocity field `v_t = x_1 - x_0` between clean data and noise
- **Inference:** Solves ODE using Euler method with 24-50 steps
- **Integration:** Simple forward Euler: `x_{t-Δt} = x_t - v_t * Δt`

### VAE Usage
- **Status:** Frozen during training (no gradients)
- **Training:** Encoder converts images to latent space before each iteration
- **Inference:** Encoder for conditions, decoder for final pixel generation
- **Role:** Acts as a frozen tokenizer/detokenizer for the LLM

### Key Insights
1. BAGEL operates entirely in VAE latent space during training
2. Flow matching learns latent-space trajectories, not pixel-space
3. The LLM predicts velocity fields conditioned on text and images
4. CFG provides strong control over text and image conditioning
5. VAE decoder only runs once at the end of generation

---

*Last updated: 2025-10-30*
