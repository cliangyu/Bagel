# BAGEL Dataset Types Reference

Complete enumeration of all dataset types, their data formats, and sequence plans.

---

## Dataset Type 1: `t2i_pretrain` (Text-to-Image)

**Implementation:** `data/t2i_dataset.py` (lines 17-129)

**Registry Key:** `'t2i_pretrain'` → `T2IIterableDataset`

### Raw Data Format

**Storage:** Parquet files with row groups

**Schema:**
```python
{
    'image': bytes,           # Binary image data (JPEG/PNG encoded)
    'captions': str,          # JSON string: {"0": "caption1", "1": "caption2", ...}
}
```

**Example Parquet Row:**
```python
{
    'image': b'\xff\xd8\xff\xe0...',  # JPEG bytes
    'captions': '{"0": "A cat sitting on a mat", "1": "Feline resting on floor covering"}'
}
```

**Configuration (data/configs/example.yaml):**
```yaml
t2i_pretrain:
  dataset_names:
  - t2i
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
```

### Processing Pipeline (lines 68-123)

1. Load image from bytes → PIL RGB
2. Transform image → tensor `[C, H, W]`
3. Parse captions JSON → randomly select one caption
4. Tokenize caption → `text_ids`
5. Build sequence plan

### Sequence Plan Structure

```python
{
    'sequence_plan': [
        {
            'type': 'text',                    # Text caption
            'enable_cfg': 1,                   # 100% chance to drop for CFG
            'loss': 0,                         # No loss on text
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'vae_image',               # Image to generate
            'enable_cfg': 0,                   # Never drop
            'loss': 1,                         # Compute MSE loss
            'special_token_loss': 0,
            'special_token_label': None,
        }
    ],
    'text_ids_list': [
        [128000, 32, 8415, 11961, 389, 264, 5634]  # Tokenized caption
    ],
    'image_tensor_list': [
        torch.Tensor([3, 1024, 1024])         # Image tensor
    ],
    'num_tokens': 2407,  # 7 text tokens + 64*64/16^2 = 2400 image tokens
}
```

**Sequence Visualization:**
```
[Caption Text (CFG)] → [Generated Image (Loss)]
```

---

## Dataset Type 2: `unified_edit` (Multi-Step Image Editing)

**Implementation:** `data/interleave_datasets/edit_dataset.py` (lines 19-72)

**Registry Key:** `'unified_edit'` → `UnifiedEditIterableDataset`

### Raw Data Format

**Storage:** Parquet files with row groups

**Schema:**
```python
{
    'image_list': List[bytes],        # List of binary images (editing trajectory)
    'instruction_list': List[List[str]]  # List of instruction options per step
}
```

**Example Parquet Row:**
```python
{
    'image_list': [
        b'\xff\xd8\xff\xe0...',  # Step 0: Original image
        b'\xff\xd8\xff\xe0...',  # Step 1: After first edit
        b'\xff\xd8\xff\xe0...',  # Step 2: After second edit
        b'\xff\xd8\xff\xe0...',  # Step 3: After third edit
    ],
    'instruction_list': [
        ["make it red", "change color to red", "turn it red"],           # 0→1 instructions
        ["add a hat", "put a hat on it", "give it headwear"],           # 1→2 instructions
        ["make it smile", "add a smile", "make the face happy"],        # 2→3 instructions
    ]
}
```

**Configuration (data/configs/example.yaml):**
```yaml
unified_edit:
  dataset_names:
  - unified_edit
  image_transform_args:
    max_image_size: 1024
    min_image_size: 512
```

### Processing Pipeline (lines 21-72)

1. **Random sampling:** Select start_idx and end_idx from trajectory (1-3 steps)
2. **Two modes (50% probability each):**
   - **Sequential mode:** Add text-image pairs for each step
   - **Concatenated mode:** Merge multiple instructions → single final image

### Sequence Plan - Sequential Mode Example

**Sample:** Start=0, End=2 (two editing steps)

```python
{
    'sequence_plan': [
        {
            'type': 'vae_image',               # Original image (no loss)
            'enable_cfg': 1,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'vit_image',               # ViT encoding of original
            'enable_cfg': 1,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'text',                    # First instruction
            'enable_cfg': 0,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'vae_image',               # First edited result (with loss)
            'enable_cfg': 0,
            'loss': 1,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'vit_image',               # ViT encoding of first edit
            'enable_cfg': 0,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'text',                    # Second instruction
            'enable_cfg': 0,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'vae_image',               # Final edited result (with loss, no vit)
            'enable_cfg': 0,
            'loss': 1,
            'special_token_loss': 0,
            'special_token_label': None,
        },
    ],
    'text_ids_list': [
        [128000, 1304, 433, 2579],              # "make it red"
        [128000, 1995, 264, 9072],              # "add a hat"
    ],
    'image_tensor_list': [
        torch.Tensor([3, 1024, 768]),           # Original (VAE)
        torch.Tensor([3, 1024, 768]),           # Original (ViT)
        torch.Tensor([3, 1024, 768]),           # First edit (VAE with loss)
        torch.Tensor([3, 1024, 768]),           # First edit (ViT)
        torch.Tensor([3, 1024, 768]),           # Final edit (VAE with loss)
    ],
}
```

**Sequence Visualization (Sequential):**
```
[Original VAE] → [Original ViT] → [Text: "make it red"] → [Edit1 VAE (Loss)] → [Edit1 ViT]
→ [Text: "add a hat"] → [Edit2 VAE (Loss)]
```

### Sequence Plan - Concatenated Mode Example

**Sample:** Concatenate all instructions → single output

```python
{
    'sequence_plan': [
        {
            'type': 'vae_image',               # Original image
            'enable_cfg': 1,
            'loss': 0,
        },
        {
            'type': 'vit_image',               # ViT encoding of original
            'enable_cfg': 1,
            'loss': 0,
        },
        {
            'type': 'text',                    # Concatenated instructions
            'enable_cfg': 0,
            'loss': 0,
        },
        {
            'type': 'vae_image',               # Final result only (with loss)
            'enable_cfg': 0,
            'loss': 1,
        },
    ],
    'text_ids_list': [
        [128000, 1304, 433, 2579, 13, 1995, 264, 9072, 13]  # "make it red. add a hat."
    ],
    'image_tensor_list': [
        torch.Tensor([3, 1024, 768]),           # Original (VAE)
        torch.Tensor([3, 1024, 768]),           # Original (ViT)
        torch.Tensor([3, 1024, 768]),           # Final result (VAE with loss)
    ],
}
```

**Sequence Visualization (Concatenated):**
```
[Original VAE] → [Original ViT] → [Text: "make it red. add a hat."] → [Final VAE (Loss)]
```

---

## Dataset Type 3: `vlm_sft` (Vision-Language Multi-Turn)

**Implementation:** `data/vlm_dataset.py` (lines 20-196)

**Registry Key:** `'vlm_sft'` → `SftJSONLIterableDataset`

### Raw Data Format

**Storage:** JSONL file + separate image/video directory

**JSONL Schema:**
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Question with <image> placeholder"
    },
    {
      "from": "gpt",
      "value": "Assistant response"
    }
  ],
  "image": ["img1.jpg", "img2.jpg"] or "single.jpg",  // Optional
  "video": "video.mp4"  // Optional (replaces <video> with <image> tokens)
}
```

**Example 1: Single Image QA**
```json
{
  "conversations": [
    {"from": "human", "value": "What's in this <image>?"},
    {"from": "gpt", "value": "The image shows a cat sitting on a windowsill."}
  ],
  "image": "cat_001.jpg"
}
```

**Example 2: Multi-Turn with Multiple Images**
```json
{
  "conversations": [
    {"from": "human", "value": "Compare these two images: <image> and <image>"},
    {"from": "gpt", "value": "The first image shows a mountain landscape, while the second depicts an ocean scene."},
    {"from": "human", "value": "Which one is more peaceful?"},
    {"from": "gpt", "value": "The ocean scene appears more tranquil with calm waters."}
  ],
  "image": ["mountain.jpg", "ocean.jpg"]
}
```

**Example 3: Video Understanding**
```json
{
  "conversations": [
    {"from": "human", "value": "Describe what happens in this <video>"},
    {"from": "gpt", "value": "A person walks across the street and waves to someone."}
  ],
  "video": "street_scene.mp4"
}
```
*Note: Video is sampled to N frames, and `<video>` is replaced with N `<image>` tokens*

**Configuration (data/configs/example.yaml):**
```yaml
vlm_sft:
  dataset_names:
  - llava_ov
  image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 490
```

### Processing Pipeline (lines 117-192)

1. Parse JSONL line
2. Load images/video frames
3. Transform images → tensors
4. Call `change_format()` to create element list (lines 67-94):
   - Split `<image>` placeholders
   - Mark human text as `has_loss=0`
   - Mark GPT text as `has_loss=1`
5. Build sequence plan from elements

### Sequence Plan - Single Image QA Example

**Input:**
```json
{
  "conversations": [
    {"from": "human", "value": "What's in this <image>?"},
    {"from": "gpt", "value": "A cat on a windowsill."}
  ],
  "image": "cat.jpg"
}
```

**Sequence Plan:**
```python
{
    'sequence_plan': [
        {
            'type': 'text',                    # "What's in this"
            'enable_cfg': 0,
            'loss': 0,                         # No loss on human question
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'vit_image',               # Image content
            'enable_cfg': 0,
            'loss': 0,                         # No loss on image
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'text',                    # "?"
            'enable_cfg': 0,
            'loss': 0,                         # No loss on question end
            'special_token_loss': 0,
            'special_token_label': None,
        },
        {
            'type': 'text',                    # "A cat on a windowsill."
            'enable_cfg': 0,
            'loss': 1,                         # Loss on GPT response
            'special_token_loss': 0,
            'special_token_label': None,
        },
    ],
    'text_ids_list': [
        [128000, 3923, 596, 304, 420],         # "What's in this"
        [30],                                   # "?"
        [128000, 32, 8415, 389, 264, 3321, 28149],  # "A cat on a windowsill."
    ],
    'image_tensor_list': [
        torch.Tensor([3, 980, 980])            # ViT image
    ],
}
```

**Sequence Visualization:**
```
[Text: "What's in this"] → [Image] → [Text: "?"] → [Text: "A cat..." (Loss)]
```

### Sequence Plan - Multi-Turn with Two Images

**Input:**
```json
{
  "conversations": [
    {"from": "human", "value": "Compare <image> and <image>"},
    {"from": "gpt", "value": "First is a mountain, second is ocean."},
    {"from": "human", "value": "Which is peaceful?"},
    {"from": "gpt", "value": "The ocean scene."}
  ],
  "image": ["mountain.jpg", "ocean.jpg"]
}
```

**Sequence Plan:**
```python
{
    'sequence_plan': [
        {
            'type': 'text',                    # "Compare"
            'enable_cfg': 0,
            'loss': 0,
        },
        {
            'type': 'vit_image',               # First image
            'enable_cfg': 0,
            'loss': 0,
        },
        {
            'type': 'text',                    # "and"
            'enable_cfg': 0,
            'loss': 0,
        },
        {
            'type': 'vit_image',               # Second image
            'enable_cfg': 0,
            'loss': 0,
        },
        {
            'type': 'text',                    # Assistant response 1
            'enable_cfg': 0,
            'loss': 1,                         # Loss on first response
        },
        {
            'type': 'text',                    # Second human question
            'enable_cfg': 0,
            'loss': 0,
        },
        {
            'type': 'text',                    # Assistant response 2
            'enable_cfg': 0,
            'loss': 1,                         # Loss on second response
        },
    ],
    'text_ids_list': [
        [128000, 28715],                       # "Compare"
        [323],                                  # "and"
        [128000, 5451, 374, 264, 16700, 11, 2132, 374, 18435],  # Response 1
        [128000, 23956, 374, 26733],           # Question 2
        [128000, 791, 18435, 6237],            # Response 2
    ],
    'image_tensor_list': [
        torch.Tensor([3, 980, 735]),           # Mountain image
        torch.Tensor([3, 980, 1225]),          # Ocean image
    ],
}
```

**Sequence Visualization:**
```
[Text: "Compare"] → [Image1] → [Text: "and"] → [Image2] → [Text: Response1 (Loss)]
→ [Text: "Which..."] → [Text: Response2 (Loss)]
```

### Sequence Plan - Video Understanding

**Input:**
```json
{
  "conversations": [
    {"from": "human", "value": "What happens in <video>?"},
    {"from": "gpt", "value": "A person walks and waves."}
  ],
  "video": "action.mp4"
}
```

**After frame sampling (e.g., 8 frames):**
```python
# <video> → '<image><image><image><image><image><image><image><image>'
```

**Sequence Plan:**
```python
{
    'sequence_plan': [
        {
            'type': 'text',                    # "What happens in"
            'enable_cfg': 0,
            'loss': 0,
        },
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 1
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 2
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 3
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 4
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 5
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 6
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 7
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # Frame 8
        {
            'type': 'text',                    # "?"
            'enable_cfg': 0,
            'loss': 0,
        },
        {
            'type': 'text',                    # Response
            'enable_cfg': 0,
            'loss': 1,                         # Loss on answer
        },
    ],
    'text_ids_list': [
        [128000, 3923, 8741, 304],             # "What happens in"
        [30],                                   # "?"
        [128000, 32, 1732, 23291, 323, 17738],  # "A person walks and waves."
    ],
    'image_tensor_list': [
        torch.Tensor([3, 980, 980]),           # Frame 1
        torch.Tensor([3, 980, 980]),           # Frame 2
        # ... (8 frames total)
    ],
}
```

**Sequence Visualization:**
```
[Text: "What..."] → [Frame1] → [Frame2] → ... → [Frame8] → [Text: "?"] → [Text: Response (Loss)]
```

---

## Summary Table

| Dataset Type | Use Case | Image Types | Loss On | Typical Sequence Pattern |
|--------------|----------|-------------|---------|--------------------------|
| `t2i_pretrain` | Text-to-image generation | VAE (latents) | Generated image | `[Caption] → [Image↓]` |
| `unified_edit` | Multi-step image editing | VAE + ViT | Edited images | `[Img₀] → [Text] → [Img₁↓] → [Text] → [Img₂↓]` |
| `vlm_sft` | Vision-language understanding | ViT (patches) | GPT responses | `[Text] → [Image] → [Text] → [Text↓]` |

**Legend:**
- `↓` = Loss computed on this element
- `VAE` = Latent diffusion targets (MSE loss)
- `ViT` = Vision transformer patches (no loss, just conditioning)
- `Text` with loss = Cross-entropy loss
- `Text` without loss = No gradient (conditioning only)

---

## Key Sequence Plan Fields

| Field | Values | Meaning |
|-------|--------|---------|
| `type` | `'text'`, `'vae_image'`, `'vit_image'` | Element modality |
| `enable_cfg` | `0` or `1` | CFG dropout probability (0=never drop, 1=always drop) |
| `loss` | `0` or `1` | Whether to compute loss on this element |
| `special_token_loss` | `0` or `1` | Whether to include special tokens in loss |
| `special_token_label` | `None` or token_id | Label for special token |
| `split_start` | `True`/`False` | (Video) Marks first frame of sequence |
| `split_end` | `True`/`False` | (Video) Marks last frame of sequence |
| `frame_delta` | int | (Video) Frame index difference to next frame |

---

## Notes on Implementation

1. **VAE vs ViT Images:**
   - `vae_image`: For diffusion training (MSE loss), uses latent space
   - `vit_image`: For understanding (no loss), uses patch embeddings

2. **CFG (Classifier-Free Guidance):**
   - `enable_cfg=1`: Element is dropped with certain probability during training
   - `enable_cfg=0`: Element is always kept

3. **Loss Masking:**
   - Only elements with `loss=1` contribute to gradient updates
   - Allows flexible control over what the model learns

4. **Multi-Modal Packing:**
   - All sequences are packed into fixed-length batches by `PackedDataset`
   - See `data/dataset_base.py:306-475` for packing logic
