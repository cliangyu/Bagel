# BAGEL Dataset Types Reference

Complete enumeration of all dataset types, their data formats, sequence plans, and detailed implementation.

**See also:** `IMPLEMENTATION_DETAILS.md` for the three-token system, attention patterns, and training mechanics.

---

## Dataset Type 1: `t2i_pretrain` (Text-to-Image)

**Implementation:** `data/t2i_dataset.py` (lines 17-129)

**Registry Key:** `'t2i_pretrain'` → `T2IIterableDataset`

**Inherits from:** `DistributedIterableDataset` (handles GPU sharding and worker distribution)

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
    image_stride: 16              # VAE downsampling stride
    max_image_size: 1024          # Maximum dimension
    min_image_size: 512           # Minimum dimension
  is_mandatory: true              # Always included in training
  weight: 1                       # Sampling weight
```

### Implementation Details: `__iter__` Method (lines 53-128)

The iterator loops over parquet files → row groups → rows:

#### **Line 55-62: Parquet File Iteration**
```python
for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
    fs = init_arrow_pf_fs(parquet_file_path)
    with fs.open_input_file(parquet_file_path) as f:
        fr = pq.ParquetFile(f)
        row_group_ids = list(range(fr.num_row_groups))
        row_group_ids_ = row_group_ids[row_group_start_id:]
```
- Supports resumption via `parquet_start_id`, `row_group_start_id`, `row_start_id`
- Uses Arrow filesystem for S3/local compatibility

#### **Line 66-76: Image Loading and Transform**
```python
for row_idx, row in df.iterrows():
    try:
        image_byte = row['image']
        image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
    except Exception as e:
        print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
        continue
    image_tensor = self.transform(image)
    height, width = image_tensor.shape[1:]
    num_tokens += width * height // transform_stride ** 2
```
- Converts RGBA → RGB with white background
- Transforms to tensor `[3, H, W]` with random resizing between [min, max]
- Computes token count: `(H/16) * (W/16)` patches

#### **Line 78-90: Caption Parsing**
```python
try:
    caption_dict = row['captions']
    caption_dict = json.loads(caption_dict)
except Exception as e:
    print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
    continue

caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
if len(caps_token) == 0:
    caption_token = self.tokenizer.encode(' ')
else:
    caption_token = random.choice(caps_token)  # Random caption selection
```
- Parses JSON caption dict: `{"0": "caption1", "1": "caption2", ...}`
- Tokenizes all captions
- **Randomly selects one** at each epoch for data augmentation

#### **Line 92-110: Sequence Plan Construction**
```python
sequence_plan, text_ids_list = [], []
text_ids = caption_token
num_tokens += len(caption_token)
text_ids_list.append(text_ids)

# Text element: Caption (can be dropped for CFG)
sequence_plan.append({
    'type': 'text',
    'enable_cfg': 1,              # 10% chance to drop during training
    'loss': 0,                    # No gradient on caption
    'special_token_loss': 0,
    'special_token_label': None,
})

# VAE image element: Generated image (always has MSE loss)
sequence_plan.append({
    'type': 'vae_image',
    'enable_cfg': 0,              # Never dropped (generation target)
    'loss': 1,                    # MSE loss computed
    'special_token_loss': 0,
    'special_token_label': None,
})
```

**Key Implementation Note:**
- **Only noised VAE tokens** are created (no clean VAE or ViT)
- This is a simple text→image generation task without past image conditioning
- The model learns: `P(image | text caption)`

#### **Line 112-123: Yield Sample**
```python
sample = dict(
    image_tensor_list=[image_tensor],
    text_ids_list=text_ids_list,
    num_tokens=num_tokens,
    sequence_plan=sequence_plan,
    data_indexes={
        "data_indexes": [parquet_idx, row_group_id, row_idx],
        "worker_id": worker_id,
        "dataset_name": self.dataset_name,
    }
)
yield sample
```

### Sequence Plan Output Example

```python
{
    'sequence_plan': [
        {
            'type': 'text',                    # Text caption
            'enable_cfg': 1,                   # CFG dropout eligible
            'loss': 0,                         # No loss on text
        },
        {
            'type': 'vae_image',               # Image to generate
            'enable_cfg': 0,                   # Never drop
            'loss': 1,                         # MSE loss on noised VAE
        }
    ],
    'text_ids_list': [
        [128000, 32, 8415, 11961, 389, 264, 5634]  # "A cat sitting on a mat"
    ],
    'image_tensor_list': [
        torch.Tensor([3, 1024, 1024])         # Image tensor
    ],
    'num_tokens': 2407,  # 7 text + 2400 image tokens (64×64 patches @ stride 16)
}
```

**Token Flow During Training:**
```
Text Tokens → Noised VAE Tokens
     ↓              ↓
  No loss      MSE loss (predict noise)
```

**Sequence Visualization:**
```
[Caption Text (CFG)] → [Noised VAE (Loss)]
```

---

## Dataset Type 2: `unified_edit` (Multi-Step Image Editing)

**Implementation:** `data/interleave_datasets/edit_dataset.py` (lines 19-72)

**Registry Key:** `'unified_edit'` → `UnifiedEditIterableDataset`

**Inherits from:**
- `InterleavedBaseIterableDataset` (provides `_add_text`, `_add_image`, `_add_video` helpers)
- `ParquetStandardIterableDataset` (handles parquet iteration with `parse_row` template method)

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
  - seedxedit_multi
  image_transform_args:
    image_stride: 16              # VAE downsampling stride
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:       # Separate transform for ViT
    image_stride: 14              # ViT patch size
    max_image_size: 518
    min_image_size: 224
  weight: 1
```

### Implementation Details: `parse_row` Method (lines 21-72)

This method is called for each parquet row by the parent `ParquetStandardIterableDataset.__iter__`.

#### **Line 22-26: Random Trajectory Sampling**
```python
def parse_row(self, row):
    image_num = len(row["image_list"])
    # Randomly choose start and end, return [0, 1] when only two images
    start_idx = random.choice(range(image_num - 1))
    max_end = min(start_idx + 3, image_num)          # At most 3 steps
    end_idx = random.choice(range(start_idx + 1, max_end))
```
- Selects a **random sub-sequence** of 1-3 editing steps from the full trajectory
- Ensures `end_idx > start_idx` (at least one transformation)
- Data augmentation: Different training epochs see different sub-trajectories

#### **Line 28-35: Original Image (Three-Token System)**
```python
data = self._init_data()
data = self._add_image(
    data,
    pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
    need_loss=False,   # ✗ No noised VAE (not generating this image)
    need_vae=True,     # ✓ Clean VAE tokens for conditioning future images
    need_vit=True,     # ✓ ViT tokens for understanding
)
```

**Three-Token System Implementation:**
- `need_loss=False` → No noised VAE tokens (we're not training to generate the original image)
- `need_vae=True` → Creates **clean VAE tokens** (t=0, no noise) as conditioning for next image
- `need_vit=True` → Creates **ViT tokens** for visual understanding

This creates **2 sequence plan entries**:
1. `{'type': 'vae_image', 'loss': 0, 'enable_cfg': 1}` (clean VAE)
2. `{'type': 'vit_image', 'loss': 0, 'enable_cfg': 1}` (ViT)

#### **Line 37-51: Mode Selection - Concatenated Instructions (50% probability)**
```python
if end_idx - start_idx > 1 and random.random() < 0.5:  # Concat mode
    if end_idx == image_num - 1:
        end_idx -= 1  # Avoid edge case

    # Concatenate all instructions
    instruction = ""
    for idx in range(start_idx + 1, end_idx + 1):
        instruction += random.choice(row["instruction_list"][idx-1]) + ". "

    data = self._add_text(data, instruction.rstrip(), need_loss=False)
    data = self._add_image(
        data,
        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
        need_loss=True,    # ✓ Noised VAE with MSE loss
        need_vae=False,    # ✗ No clean VAE (end of sequence)
        need_vit=False,    # ✗ No ViT (end of sequence)
    )
```

**Concatenated Mode Result:**
```
[Clean VAE₀] → [ViT₀] → [Text: "make it red. add a hat."] → [Noised VAE_final]
                                                                    ↑
                                                                MSE Loss
```

#### **Line 52-71: Mode Selection - Sequential Instructions (50% probability)**
```python
else:  # Sequential mode
    for idx in range(start_idx + 1, end_idx + 1):
        instruction = random.choice(row["instruction_list"][idx-1])
        data = self._add_text(data, instruction, need_loss=False)

        if idx != end_idx:  # Intermediate image
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                need_loss=True,    # ✓ Noised VAE (MSE loss)
                need_vae=True,     # ✓ Clean VAE (condition next image)
                need_vit=True,     # ✓ ViT (understanding)
            )
        else:  # Final image
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                need_loss=True,    # ✓ Noised VAE (MSE loss)
                need_vae=False,    # ✗ No clean VAE (no future images)
                need_vit=False,    # ✗ No ViT (no future images)
            )
```

**Sequential Mode Result (2 steps):**
```
[Clean VAE₀] → [ViT₀] → [Text₁] → [Noised VAE₁] → [Clean VAE₁] → [ViT₁] → [Text₂] → [Noised VAE₂]
                                          ↑                                                    ↑
                                      MSE Loss                                            MSE Loss
```

**Key Implementation Insight:**
- **Intermediate images** need all three token types (noised for loss, clean+ViT for conditioning)
- **Final image** only needs noised tokens (nothing to condition after it)
- This implements the paper's strategy: "subsequent tokens attend to clean VAE and ViT of preceding images"

### Sequence Plan Output Examples

#### **Sequential Mode (2 editing steps):**

```python
{
    'sequence_plan': [
        # Original image conditioning
        {'type': 'vae_image', 'enable_cfg': 1, 'loss': 0},  # Clean VAE₀
        {'type': 'vit_image', 'enable_cfg': 1, 'loss': 0},  # ViT₀

        # First edit
        {'type': 'text', 'enable_cfg': 0, 'loss': 0},       # Instruction: "make it red"
        {'type': 'vae_image', 'enable_cfg': 0, 'loss': 1},  # Noised VAE₁ (MSE loss)
        {'type': 'vae_image', 'enable_cfg': 0, 'loss': 0},  # Clean VAE₁
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0},  # ViT₁

        # Second edit
        {'type': 'text', 'enable_cfg': 0, 'loss': 0},       # Instruction: "add a hat"
        {'type': 'vae_image', 'enable_cfg': 0, 'loss': 1},  # Noised VAE₂ (MSE loss)
    ],
    'text_ids_list': [
        [128000, 1304, 433, 2579],    # "make it red"
        [128000, 1995, 264, 9072],    # "add a hat"
    ],
    'image_tensor_list': [
        torch.Tensor([3, 1024, 768]),  # Original (VAE)
        torch.Tensor([3, 518, 518]),   # Original (ViT, different resolution)
        torch.Tensor([3, 1024, 768]),  # Edit 1 noised
        torch.Tensor([3, 1024, 768]),  # Edit 1 clean
        torch.Tensor([3, 518, 518]),   # Edit 1 ViT
        torch.Tensor([3, 1024, 768]),  # Edit 2 noised
    ],
}
```

**Token Flow:**
```
Clean VAE₀ + ViT₀ → Text₁ → Noised VAE₁ + Clean VAE₁ + ViT₁ → Text₂ → Noised VAE₂
     ↓                ↓            ↓              ↓         ↓       ↓          ↓
 Condition       Condition    MSE Loss       Condition Condition Condition MSE Loss
```

#### **Concatenated Mode:**

```python
{
    'sequence_plan': [
        {'type': 'vae_image', 'enable_cfg': 1, 'loss': 0},  # Clean VAE₀
        {'type': 'vit_image', 'enable_cfg': 1, 'loss': 0},  # ViT₀
        {'type': 'text', 'enable_cfg': 0, 'loss': 0},       # "make it red. add a hat."
        {'type': 'vae_image', 'enable_cfg': 0, 'loss': 1},  # Noised VAE_final (MSE loss)
    ],
    'text_ids_list': [
        [128000, 1304, 433, 2579, 13, 1995, 264, 9072, 13]  # Concatenated instructions
    ],
    'image_tensor_list': [
        torch.Tensor([3, 1024, 768]),  # Original (VAE)
        torch.Tensor([3, 518, 518]),   # Original (ViT)
        torch.Tensor([3, 1024, 768]),  # Final result noised
    ],
}
```

**Sequence Visualization:**
```
Sequential:    [Clean VAE₀] [ViT₀] [Text₁] [Noised VAE₁] [Clean VAE₁] [ViT₁] [Text₂] [Noised VAE₂]
Concatenated:  [Clean VAE₀] [ViT₀] [Text: "step1. step2."] [Noised VAE_final]
```

---

## Dataset Type 3: `vlm_sft` (Vision-Language Multi-Turn)

**Implementation:** `data/vlm_dataset.py` (lines 20-196)

**Registry Key:** `'vlm_sft'` → `SftJSONLIterableDataset`

**Inherits from:** `DistributedIterableDataset`

### Raw Data Format

**Storage:** JSONL file + separate image/video directory

**JSONL Schema:**
```json
{
  "conversations": [
    {"from": "human", "value": "Question with <image> placeholder"},
    {"from": "gpt", "value": "Assistant response"}
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

**Configuration (data/configs/example.yaml):**
```yaml
vlm_sft:
  dataset_names:
  - llava_ov
  image_transform_args:
    image_stride: 14              # ViT patch size
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040         # Maximum total pixels per image
  frame_sampler_args:
    max_num_frames: 12            # Maximum frames per video
    min_num_frames: 8             # Minimum frames per video
  shuffle_lines: True             # Shuffle JSONL lines
  weight: 1
```

### Implementation Details: `__iter__` Method (lines 96-195)

#### **Line 110-141: Load Images/Videos**
```python
while True:
    for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
        num_tokens = 0
        image_tensor_list = []
        text_ids_list = []
        sequence_plan = []

        try:
            data_item = json.loads(data)
            raw_images = None

            # Load images
            if 'image' in data_item:
                if type(data_item['image']) == list:
                    raw_images = [
                        pil_img2rgb(Image.open(os.path.join(image_dir, image)))
                        for image in data_item['image']
                    ]
                else:
                    raw_images = [
                        pil_img2rgb(Image.open(os.path.join(image_dir, data_item['image'])))
                    ]

            # Load video → frame sampling
            elif 'video' in data_item:
                raw_images = self.frame_sampler(os.path.join(image_dir, data_item['video']))
                # Replace <video> with N×<image>
                special_tokens = '<image>' * len(raw_images)
                for item in data_item['conversations']:
                    if '<video>' in item['value']:
                        item['value'] = item['value'].replace('<video>', special_tokens)
                        break
```

**Key Video Processing:**
- `FrameSampler` samples 8-12 frames uniformly from video
- `<video>` token is replaced with N consecutive `<image>` tokens
- Each frame becomes a separate ViT image in the sequence

#### **Line 143-149: Transform Images**
```python
if raw_images:
    for raw_image in raw_images:
        image_tensor = self.transform(raw_image, img_num=len(raw_images))
        image_tensor_list.append(image_tensor)
        height, width = image_tensor.shape[1:]
        num_tokens += width * height // transform_stride ** 2
```

- `transform(img_num=...)` allows different resizing strategies for multi-image inputs
- Adaptive sizing based on total pixel budget

#### **Line 150: Parse Conversations → Elements**
```python
elements = self.change_format(data_item, len(image_tensor_list))
```

**`change_format` Method (lines 67-94):** Converts conversations to element list

```python
def change_format(self, data, num_images):
    elements = []
    for conversation in data['conversations']:
        if conversation['from'] == 'human':
            if '<image>' not in conversation['value']:
                # Pure text question
                elements.append({'type': 'text', 'has_loss': 0, 'text': conversation['value']})
            else:
                # Split by <image> placeholder
                text_list = conversation['value'].split('<image>')
                for idx, text in enumerate(text_list):
                    if text.strip() != '':
                        elements.append({'type': 'text', 'has_loss': 0, 'text': text.strip()})
                    if (idx != len(text_list) - 1) and (idx < num_images):
                        elements.append({'type': 'image'})  # Insert image placeholder

        elif conversation['from'] == 'gpt':
            # GPT response has loss
            elements.append({'type': 'text', 'has_loss': 1, 'text': conversation['value']})

    return elements
```

**Example Transformation:**
```python
# Input:
{"from": "human", "value": "Compare <image> and <image>"}
{"from": "gpt", "value": "First is mountain, second is ocean"}

# Output elements:
[
    {'type': 'text', 'has_loss': 0, 'text': 'Compare'},
    {'type': 'image'},
    {'type': 'text', 'has_loss': 0, 'text': 'and'},
    {'type': 'image'},
    {'type': 'text', 'has_loss': 1, 'text': 'First is mountain, second is ocean'},
]
```

#### **Line 152-176: Build Sequence Plan**
```python
for item in elements:
    if item['type'] == 'text':
        text_data = item['text']
        text_ids = self.tokenizer.encode(text_data)
        if len(text_ids) > 0:
            text_ids_list.append(text_ids)
            num_tokens += len(text_ids)
            current_plan = {
                'type': 'text',
                'enable_cfg': 0,           # Text never dropped in VLM (only generation tasks)
                'loss': item['has_loss'],  # 0 for human, 1 for GPT
                'special_token_loss': 0,
                'special_token_label': None,
            }
            sequence_plan.append(current_plan)

    elif item['type'] == 'image':
        current_plan = {
            'type': 'vit_image',           # Only ViT tokens (understanding, not generation)
            'enable_cfg': 0,               # Never dropped
            'loss': 0,                     # No loss on images (only on text responses)
            'special_token_loss': 0,
            'special_token_label': None,
        }
        sequence_plan.append(current_plan)
```

**Key Implementation Notes:**
- **Only ViT tokens** are created (no VAE tokens)
- VLM is for **understanding**, not **generation**
- Loss is computed only on GPT text responses
- `enable_cfg=0` for all tokens (CFG not used in understanding-only tasks)

#### **Line 177-192: Validation and Yield**
```python
has_loss = [item['loss'] for item in sequence_plan]
if sum(has_loss) == 0:
    print(f'No loss defined, skipped.')
    continue  # Skip samples with no GPT responses

yield dict(
    image_tensor_list=image_tensor_list,
    text_ids_list=text_ids_list,
    sequence_plan=sequence_plan,
    num_tokens=num_tokens,
    data_indexes={
        "data_indexes": row_idx,
        "worker_id": worker_id,
        "dataset_name": self.dataset_name,
    }
)
```

### Sequence Plan Output Examples

#### **Single Image QA:**

```python
{
    'sequence_plan': [
        {'type': 'text', 'enable_cfg': 0, 'loss': 0},      # "What's in this"
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Image
        {'type': 'text', 'enable_cfg': 0, 'loss': 0},      # "?"
        {'type': 'text', 'enable_cfg': 0, 'loss': 1},      # "A cat..." (CE loss)
    ],
    'text_ids_list': [
        [128000, 3923, 596, 304, 420],                     # "What's in this"
        [30],                                               # "?"
        [128000, 32, 8415, 389, 264, 3321, 28149],         # "A cat on a windowsill."
    ],
    'image_tensor_list': [
        torch.Tensor([3, 980, 980])                        # ViT image
    ],
}
```

**Token Flow:**
```
Text₁ → ViT Image → Text₂ → Text₃
  ↓         ↓         ↓        ↓
 None     None      None   CE Loss
```

#### **Video Understanding (8 frames):**

```python
{
    'sequence_plan': [
        {'type': 'text', 'enable_cfg': 0, 'loss': 0},      # "What happens in"
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 1
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 2
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 3
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 4
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 5
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 6
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 7
        {'type': 'vit_image', 'enable_cfg': 0, 'loss': 0}, # Frame 8
        {'type': 'text', 'enable_cfg': 0, 'loss': 0},      # "?"
        {'type': 'text', 'enable_cfg': 0, 'loss': 1},      # "A person..." (CE loss)
    ],
    'text_ids_list': [
        [128000, 3923, 8741, 304],                         # "What happens in"
        [30],                                               # "?"
        [128000, 32, 1732, 23291, 323, 17738],             # "A person walks and waves."
    ],
    'image_tensor_list': [
        torch.Tensor([3, 980, 980]),  # Frame 1
        torch.Tensor([3, 980, 980]),  # Frame 2
        # ... (8 frames total)
    ],
}
```

**Sequence Visualization:**
```
[Text] → [Frame1] → [Frame2] → ... → [Frame8] → [Text: "?"] → [Text: Response (Loss)]
```

---

## Comprehensive Comparison Table

| Aspect | `t2i_pretrain` | `unified_edit` | `vlm_sft` |
|--------|----------------|----------------|-----------|
| **Use Case** | Text-to-image generation | Multi-step image editing | Vision-language understanding |
| **Noised VAE** | ✓ (generated image) | ✓ (each edited image) | ✗ |
| **Clean VAE** | ✗ | ✓ (conditioning for next edit) | ✗ |
| **ViT Tokens** | ✗ | ✓ (understanding + conditioning) | ✓ (all images) |
| **Text Loss (CE)** | ✗ | ✗ | ✓ (GPT responses) |
| **Image Loss (MSE)** | ✓ (generated image) | ✓ (edited images) | ✗ |
| **CFG Dropout** | Text: 0.1 | Text: 0.1, Clean VAE: 0.3, ViT: 0.3 | None (enable_cfg=0) |
| **Attention Pattern** | Causal text → Noise image | Causal text + Full/Noise images | Causal text + Full images |
| **Inherits From** | `DistributedIterableDataset` | `InterleavedBaseIterableDataset` + `ParquetStandardIterableDataset` | `DistributedIterableDataset` |
| **Key Method** | `__iter__` (lines 53-128) | `parse_row` (lines 21-72) | `__iter__` + `change_format` (lines 67-94) |
| **Data Format** | Parquet: image bytes + caption JSON | Parquet: image list + instruction list | JSONL + image/video files |
| **Typical Length** | ~2400 tokens | ~5000-8000 tokens | ~500-2000 tokens |

---

## Key Sequence Plan Fields Reference

| Field | Values | Meaning |
|-------|--------|---------|
| `type` | `'text'`, `'vae_image'`, `'vit_image'` | Element modality |
| `enable_cfg` | `0` or `1` | CFG dropout eligibility (0=never drop, 1=drop with probability) |
| `loss` | `0` or `1` | Whether to compute loss on this element |
| `special_token_loss` | `0` or `1` | Whether to include special tokens like `<|im_end|>` in loss |
| `special_token_label` | `None` or token_id | Label for special token |
| `split_start` | `True`/`False` | (Video) Marks first frame/image of a group (new timestep) |
| `split_end` | `True`/`False` | (Video) Marks last frame/image of a group |
| `frame_delta` | int | (Video) Temporal distance to next frame |

---

## Helper Methods from `InterleavedBaseIterableDataset`

**File:** `data/interleave_datasets/interleave_t2i_dataset.py` (lines 21-86)

### `_add_text(data, text, need_loss, enable_cfg=True)`

```python
def _add_text(self, data, text, need_loss, enable_cfg=True):
    text_ids = self.tokenizer.encode(text)
    data['num_tokens'] += len(text_ids)
    data['text_ids_list'].append(text_ids)
    data['sequence_plan'].append({
        'type': 'text',
        'enable_cfg': int(enable_cfg),
        'loss': int(need_loss),
        'special_token_loss': 0,
        'special_token_label': None,
    })
    return data
```

### `_add_image(data, image, need_loss, need_vae, need_vit, enable_cfg=True)`

```python
def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
    """
    Implements the three-token system:
    - need_loss: Noised VAE tokens (MSE loss)
    - need_vae:  Clean VAE tokens (conditioning, no loss)
    - need_vit:  ViT tokens (understanding, no loss)
    """
    assert need_loss or need_vae or need_vit

    if need_loss:  # Noised VAE
        data['sequence_plan'].append({
            'type': 'vae_image',
            'enable_cfg': 0,  # Never drop noised VAE (generation target)
            'loss': 1,
            'special_token_loss': 0,
            'special_token_label': None,
        })
        image_tensor = self.transform(image)
        height, width = image_tensor.shape[1:]
        data['num_tokens'] += width * height // self.transform.stride ** 2
        data['image_tensor_list'].append(image_tensor)

    if need_vae:  # Clean VAE
        data['sequence_plan'].append({
            'type': 'vae_image',
            'enable_cfg': int(enable_cfg),  # Can be dropped with vae_cond_dropout_prob
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        })
        image_tensor = self.transform(image)
        height, width = image_tensor.shape[1:]
        data['num_tokens'] += width * height // self.transform.stride ** 2
        data['image_tensor_list'].append(image_tensor.clone())

    if need_vit:  # ViT tokens
        data['sequence_plan'].append({
            'type': 'vit_image',
            'enable_cfg': int(enable_cfg),  # Can be dropped with vit_cond_dropout_prob
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        })
        vit_image_tensor = self.vit_transform(image)
        height, width = vit_image_tensor.shape[1:]
        data['num_tokens'] += width * height // self.vit_transform.stride ** 2
        data['image_tensor_list'].append(vit_image_tensor)

    return data
```

**Key Insight:** A single image can generate up to **3 different token sequences** depending on the flags.

### `_add_video(data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True)`

```python
def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
    """
    Adds video frames with temporal metadata for diffusion forcing and grouping.
    """
    for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
        current_sequence_plan = {
            'type': 'vae_image',
            'enable_cfg': 0 if need_loss else int(enable_cfg),
            'loss': 1 if need_loss else 0,
            'special_token_loss': 0,
            'special_token_label': None,
            'split_start': idx == 0,              # First frame starts new group
            'split_end': idx == len(frames) - 1,  # Last frame ends group
        }
        if idx < len(frame_indexes) - 1:
            current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx

        data['sequence_plan'].append(current_sequence_plan)
        image_tensor = self.transform(image)
        data['image_tensor_list'].append(image_tensor)
        # ... token counting ...

    return data
```

**Used for:** Video-to-video generation with temporal consistency (diffusion forcing strategy).

---

## Advanced Topics

### 1. **Timestep Assignment** (see `IMPLEMENTATION_DETAILS.md`)

- **Clean VAE:** `timestep = float('-inf')` → t=0 (no noise)
- **Noised VAE:** `timestep = np.random.randn()` → Random Gaussian noise
- **Grouping:** `split_start=True` assigns new timestep; `split_start=False` inherits

### 2. **Attention Masking** (see `IMPLEMENTATION_DETAILS.md`)

- **Causal:** Text tokens attend to past only
- **Full:** Image tokens attend to all tokens within image + all past
- **Noise:** Noised VAE tokens are "invisible" to future tokens

### 3. **CFG Dropout** (see `IMPLEMENTATION_DETAILS.md`)

**Dropout Probabilities:**
- Text: 0.1 (paper) / 0.1 (default)
- ViT: 0.5 (paper) / 0.3 (default)
- Clean VAE: 0.1 (paper) / 0.3 (default)
- **Noised VAE: Never dropped** (always `enable_cfg=0`)

### 4. **Multi-Modal Packing** (`data/dataset_base.py:306-475`)

All sequences are packed into fixed-length batches (~32K tokens) by `PackedDataset`, which:
- Accumulates samples until `expected_num_tokens` is reached
- Creates attention masks via `prepare_attention_mask_per_sample` or FlexAttention
- Assigns timesteps to VAE tokens
- Tracks loss indexes for CE and MSE

---

## Quick Reference: Which Dataset for Which Task?

| Task | Dataset Type | Key Features |
|------|--------------|--------------|
| Text → Image generation | `t2i_pretrain` | Simple caption → image, MSE loss only |
| Image editing (one step) | `unified_edit` (concatenated mode) | Original + instruction → edited |
| Image editing (multi-step) | `unified_edit` (sequential mode) | Chained edits with intermediate results |
| Visual question answering | `vlm_sft` | Image → question → answer, CE loss on answer |
| Video understanding | `vlm_sft` (with video) | Video frames → question → answer |
| Video generation | Custom (using `_add_video`) | Not in current dataset registry |

---

For detailed implementation of attention patterns, diffusion forcing, and the three-token system, see **`IMPLEMENTATION_DETAILS.md`**.
