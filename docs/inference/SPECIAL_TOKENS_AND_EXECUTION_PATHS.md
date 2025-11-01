# Special Tokens and Inference Execution Paths

This document provides a deep dive into how special tokens work and traces the exact execution paths through the BAGEL inference code for different scenarios.

---

## Table of Contents

1. [Special Tokens Overview](#special-tokens-overview)
2. [Token Roles in Inference](#token-roles-in-inference)
3. [Execution Path Traces](#execution-path-traces)
4. [Token Sequence Examples](#token-sequence-examples)
5. [Critical Implementation Details](#critical-implementation-details)

---

## Special Tokens Overview

BAGEL uses **four special tokens** to structure interleaved text-image sequences. These are defined in `data/data_utils.py:130-165`.

### Token Definitions

```python
def add_special_tokens(tokenizer):
    new_tokens = [
        '<|im_start|>',      # Text/conversation start marker
        '<|im_end|>',        # Text/conversation end marker
        '<|vision_start|>',  # Image start marker
        '<|vision_end|>',    # Image end marker
    ]

    tokenizer.add_tokens(new_tokens)

    new_token_ids = {
        'bos_token_id': tokenizer.convert_tokens_to_ids('<|im_start|>'),
        'eos_token_id': tokenizer.convert_tokens_to_ids('<|im_end|>'),
        'start_of_image': tokenizer.convert_tokens_to_ids('<|vision_start|>'),
        'end_of_image': tokenizer.convert_tokens_to_ids('<|vision_end|>'),
    }

    return tokenizer, new_token_ids, num_new_tokens
```

### Token Summary Table

| Token | Alias | Purpose | Auto-Added |
|-------|-------|---------|------------|
| `<|im_start|>` | `bos_token_id` | Marks start of text/response | ✅ Yes |
| `<|im_end|>` | `eos_token_id` | Marks end of text/response, stops generation | ✅ Yes |
| `<|vision_start|>` | `start_of_image` | Marks beginning of image tokens | ✅ Yes |
| `<|vision_end|>` | `end_of_image` | Marks end of image tokens | ✅ Yes |

**Important**: All tokens are **automatically added** by the model's `prepare_*` methods. Users never need to add them manually.

---

## Token Roles in Inference

### 1. `<|im_start|>` (BOS Token)

**Where Added:**

**In text prompts** (`modeling/bagel/bagel.py:246`):
```python
def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
    text_ids = tokenizer.encode(prompt)  # User text: [123, 456, 789]
    text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
    # Result: [<|im_start|>, 123, 456, 789, <|im_end|>]
```

**In text generation** (`modeling/bagel/bagel.py:916`):
```python
def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
    # First token of generated text is always <|im_start|>
    packed_start_tokens.append(new_token_ids['bos_token_id'])
```

**Role:**
- Signals model to enter text mode
- First token in every text sequence
- Provides context for text generation

---

### 2. `<|im_end|>` (EOS Token)

**Where Used:**

**As stop condition** (`modeling/bagel/bagel.py:996`):
```python
def generate_text(self, ..., end_token_id: int = None):
    while step < max_length:
        # Generate next token
        curr_tokens = torch.argmax(pred_logits, dim=-1)

        # Stop when <|im_end|> is generated
        if end_token_id is not None and curr_tokens[0] == end_token_id:
            break
```

**In post-processing** (`inferencer.py:204`):
```python
def gen_text(self, gen_context, ...):
    unpacked_latent = self.model.generate_text(
        end_token_id=self.new_token_ids['eos_token_id'],  # Pass stop token
    )
    output = self.tokenizer.decode(unpacked_latent[:,0])
    # Raw: "<|im_start|>The answer is...<|im_end|>"

    output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
    # Clean: "The answer is..."
    return output
```

**Role:**
- **Critical**: Only token that stops text generation
- Marks end of text sequence
- Removed during post-processing

---

### 3. `<|vision_start|>` (Start of Image)

**Where Added:**

**Before ViT tokens** (`modeling/bagel/bagel.py:312-334`):
```python
def prepare_vit_images(self, ...):
    packed_text_ids.append(new_token_ids['start_of_image'])  # <|vision_start|>

    # Process ViT tokens
    vit_tokens = patchify(image_tensor, self.vit_patch_size)
    # num_img_tokens ViT tokens added here

    packed_text_ids.append(new_token_ids['end_of_image'])    # <|vision_end|>
```

**Before VAE tokens** (`modeling/bagel/bagel.py:431-456`):
```python
def prepare_vae_images(self, ...):
    packed_text_ids.append(new_token_ids['start_of_image'])  # <|vision_start|>

    # Process VAE tokens (clean, no noise during inference)
    # num_img_tokens VAE tokens added here

    packed_text_ids.append(new_token_ids['end_of_image'])    # <|vision_end|>
```

**Before generated image** (`modeling/bagel/bagel.py:563-586`):
```python
def prepare_vae_latent(self, ...):
    # Marks where image will be generated
    packed_text_ids.append(new_token_ids['start_of_image'])  # <|vision_start|>

    # Image tokens will be generated via flow matching
    # Random noise initialized here

    packed_text_ids.append(new_token_ids['end_of_image'])    # <|vision_end|>
```

**Role:**
- Marks boundary where image tokens begin
- Signals model to switch to vision mode
- Essential for interleaved sequences

---

### 4. `<|vision_end|>` (End of Image)

**Role:**
- Marks boundary where image tokens end
- Signals model to return to text mode
- Always paired with `<|vision_start|>`

---

## Execution Path Traces

This section traces the **exact code execution** for different inference scenarios.

### Scenario 1: Text-to-Image Generation (No Think)

**User Code:**
```python
inferencer(text="A cat sitting on a mat", cfg_text_scale=4.0, cfg_img_scale=1.0)
```

**Execution Path:**

#### Step 1: `__call__()` Entry Point (`inferencer.py:288-313`)

```python
def __call__(self, image=None, text="A cat sitting on a mat", **kargs):
    output_dict = {'image': None, 'text': None}

    input_list = []
    if image is not None:
        input_list.append(image)  # ✗ Skipped (image is None)
    if text is not None:
        input_list.append(text)   # ✓ input_list = ["A cat sitting on a mat"]

    output_list = self.interleave_inference(input_list, **kargs)
```

#### Step 2: `interleave_inference()` Initialization (`inferencer.py:208-231`)

```python
def interleave_inference(
    self,
    input_lists=["A cat sitting on a mat"],
    think=False,                    # Default
    understanding_output=False,     # Default (we want image)
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    # ... other params
):
    output_list = []

    # Initialize THREE contexts
    gen_context = self.init_gen_context()
    # gen_context = {
    #     'kv_lens': [0],
    #     'ropes': [0],
    #     'past_key_values': NaiveCache()  # Empty
    # }

    cfg_text_context = deepcopy(gen_context)  # Copy of empty
    cfg_img_context = deepcopy(gen_context)   # Copy of empty
```

#### Step 3: Process Input Loop (`inferencer.py:242-256`)

```python
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        if think:  # ✗ False, skip system prompt
            pass

        for input_term in input_lists:  # ["A cat sitting on a mat"]
            if isinstance(input_term, str):  # ✓ True
                # SAVE state BEFORE adding text (for CFG)
                cfg_text_context = deepcopy(gen_context)
                # cfg_text_context = {kv_lens: [0], ...} (empty)

                # ADD TEXT to main context
                gen_context = self.update_context_text(input_term, gen_context)
                # Inside update_context_text():
                #   1. Tokenize: "A cat sitting on a mat" → [tok1, tok2, ...]
                #   2. Wrap: [<|im_start|>, tok1, tok2, ..., <|im_end|>]
                #   3. Forward pass through LLM
                #   4. Cache keys/values
                # gen_context now = {kv_lens: [N], ropes: [N], past_kv: <cached>}

                # ALSO add text to img CFG context
                cfg_img_context = self.update_context_text(input_term, cfg_img_context)
```

**After loop:**
```
gen_context:      [<|im_start|> A cat sitting on a mat <|im_end|>]
cfg_text_context: [] (empty - saved before text)
cfg_img_context:  [<|im_start|> A cat sitting on a mat <|im_end|>]
```

#### Step 4: Generate Output (`inferencer.py:258-284`)

```python
        if understanding_output:  # ✗ False (we want image, not text)
            pass  # Skip

        else:  # ✓ Image generation
            if think:  # ✗ False, skip thinking
                pass

            # GENERATE IMAGE
            img = self.gen_image(
                image_shapes=(1024, 1024),
                gen_context,                    # Has text prompt
                cfg_text_precontext=cfg_text_context,  # Empty (no text)
                cfg_img_precontext=cfg_img_context,    # Has text

                cfg_text_scale=4.0,
                cfg_img_scale=1.0,
                num_timesteps=50,
                # ... other params
            )
            # Inside gen_image():
            #   1. prepare_vae_latent(): Add [<|vision_start|>] + noise + [<|vision_end|>]
            #   2. Flow matching denoising (50 steps):
            #      Each step:
            #        a. Forward with full context (text prompt) → v_t
            #        b. Forward with empty context (no text) → cfg_text_v_t
            #        c. Apply CFG: v_t_final = cfg_text_v_t + 4.0 * (v_t - cfg_text_v_t)
            #   3. decode_image(): VAE decode → PIL Image

            output_list.append(img)

    return output_list  # [<PIL.Image>]
```

#### Step 5: Return to `__call__()` (`inferencer.py:308-313`)

```python
    for i in output_list:  # [<PIL.Image>]
        if isinstance(i, Image.Image):  # ✓ True
            output_dict['image'] = i
        elif isinstance(i, str):
            output_dict['text'] = i  # Not executed

    return output_dict  # {'image': <PIL.Image>, 'text': None}
```

---

### Scenario 2: Text-to-Image with Think

**User Code:**
```python
inferencer(text="a car made of small cars", think=True, ...)
```

**Key Differences from Scenario 1:**

#### Step 3a: Add System Prompt (`inferencer.py:234-240`)

```python
    if think:  # ✓ True NOW
        if understanding_output:  # ✗ False
            system_prompt = VLM_THINK_SYSTEM_PROMPT
        else:  # ✓ Image generation
            system_prompt = GEN_THINK_SYSTEM_PROMPT
            # "You should first think about the planning process in the mind
            #  and then generate the image. The planning process is enclosed
            #  within <think> </think> tags..."

        gen_context = self.update_context_text(system_prompt, gen_context)
        cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
```

#### Step 3b: Process User Prompt

```python
    for input_term in ["a car made of small cars"]:
        if isinstance(input_term, str):
            cfg_text_context = deepcopy(gen_context)  # Save WITH system prompt
            gen_context = self.update_context_text(input_term, gen_context)
            cfg_img_context = self.update_context_text(input_term, cfg_img_context)
```

**After loop:**
```
gen_context:      [System prompt] + [User prompt]
cfg_text_context: [System prompt] (saved before user prompt)
cfg_img_context:  [System prompt] + [User prompt]
```

#### Step 4a: Generate Thinking Text FIRST (`inferencer.py:262-266`)

```python
        if think:  # ✓ True
            gen_text = self.gen_text(gen_context, max_length=1000, ...)
            # Inside gen_text():
            #   1. prepare_start_tokens(): Add <|im_start|>
            #   2. Autoregressive loop:
            #      - Generate: <think>To create an image of a car made
            #        of small cars, I should show a large car silhouette...
            #      - Continue until </think> or <|im_end|>
            #   3. Return text
            # gen_text = "<think>planning process...</think>"

            # ADD THINKING TO CONTEXT
            gen_context = self.update_context_text(gen_text, gen_context)
            output_list.append(gen_text)
```

**After thinking:**
```
gen_context:      [System] + [User] + [Thinking text]
cfg_text_context: [System] (unchanged)
cfg_img_context:  [System] + [User] (unchanged)
```

#### Step 4b: Generate Image Conditioned on Thinking

```python
            img = self.gen_image(
                gen_context,  # NOW includes thinking!
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                ...
            )
            # Image is generated with thinking as additional context
            output_list.append(img)
```

**Final Output:**
```python
return {'image': <PIL.Image>, 'text': "<think>planning...</think>"}
```

---

### Scenario 3: Image Editing

**User Code:**
```python
image = Image.open('woman.jpg')
inferencer(image=image, text="She boards a modern subway...", ...)
```

**Execution Path:**

#### Step 1: Build Input List

```python
def __call__(self, image=<PIL.Image>, text="She boards...", **kargs):
    input_list = []
    if image is not None:  # ✓ True
        input_list.append(image)
    if text is not None:  # ✓ True
        input_list.append(text)

    # input_list = [<PIL.Image>, "She boards a modern subway..."]
```

#### Step 2: Process Inputs - FIRST Iteration (Image)

```python
    for input_term in input_list:  # First: <PIL.Image>
        if isinstance(input_term, str):
            pass  # ✗ Skip

        elif isinstance(input_term, Image.Image):  # ✓ True
            # Resize to VAE resolution
            input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
            # input_term now (1024, 1024)

            # ADD IMAGE TO CONTEXT
            gen_context = self.update_context_image(
                input_term,
                gen_context,
                vae=not understanding_output  # vae=True (editing needs VAE)
            )
            # Inside update_context_image(vae=True, vit=True):
            #   if vae:  # ✓ True
            #     1. Encode image with VAE → clean latent codes (no noise!)
            #     2. Add [<|vision_start|>] + VAE_tokens + [<|vision_end|>]
            #     3. Forward pass, cache keys/values
            #
            #   if vit:  # ✓ True
            #     1. Encode image with SigLIP ViT
            #     2. Add [<|vision_start|>] + ViT_tokens + [<|vision_end|>]
            #     3. Forward pass, cache keys/values

            image_shapes = input_term.size[::-1]  # (1024, 1024)

            # SAVE context AFTER image (for text CFG)
            cfg_text_context = deepcopy(gen_context)
```

**After first iteration:**
```
gen_context:      [<|vision_start|> VAE_tokens <|vision_end|>]
                  [<|vision_start|> ViT_tokens <|vision_end|>]
cfg_text_context: [Same as gen_context]
cfg_img_context:  [] (empty)
```

#### Step 3: Process Inputs - SECOND Iteration (Text)

```python
    for input_term in input_list:  # Second: "She boards a modern subway..."
        if isinstance(input_term, str):  # ✓ True
            # SAVE before adding text
            cfg_text_context = deepcopy(gen_context)
            # cfg_text_context = [VAE + ViT tokens]

            # ADD TEXT
            gen_context = self.update_context_text(
                "She boards a modern subway...",
                gen_context
            )
            # Adds: [<|im_start|> She boards... <|im_end|>]

            # ALSO add to img CFG context
            cfg_img_context = self.update_context_text(
                "She boards a modern subway...",
                cfg_img_context
            )
```

**After second iteration:**
```
gen_context:      [VAE_tokens] + [ViT_tokens] + [<|im_start|> Text <|im_end|>]
cfg_text_context: [VAE_tokens] + [ViT_tokens] (no text instruction)
cfg_img_context:  [<|im_start|> Text <|im_end|>] (no image)
```

#### Step 4: Generate Edited Image

```python
        img = self.gen_image(
            image_shapes=(1024, 1024),
            gen_context,                    # Image + Text
            cfg_text_precontext=cfg_text_context,  # Image only
            cfg_img_precontext=cfg_img_context,    # Text only

            cfg_text_scale=4.0,
            cfg_img_scale=2.0,  # ← Higher for editing!
            ...
        )
        # Inside gen_image() denoising loop:
        #   Each step does THREE forward passes:
        #     1. Conditional:        [Image + Text] → v_t
        #     2. Text unconditional: [Image only]   → cfg_text_v_t
        #     3. Image unconditional:[Text only]    → cfg_img_v_t
        #
        #   Nested CFG:
        #     v_t_text = cfg_text_v_t + 4.0 * (v_t - cfg_text_v_t)
        #     v_t_final = cfg_img_v_t + 2.0 * (v_t_text - cfg_img_v_t)

        output_list.append(img)
```

---

### Scenario 4: Image Understanding

**User Code:**
```python
image = Image.open('meme.jpg')
inferencer(image=image, text="Explain this meme", understanding_output=True)
```

**Key Difference: `understanding_output=True`**

#### Step 1: Process Image - ViT ONLY!

```python
    elif isinstance(input_term, Image.Image):
        input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))

        gen_context = self.update_context_image(
            input_term,
            gen_context,
            vae=not understanding_output  # vae=False! ← KEY DIFFERENCE
        )
        # Inside update_context_image(vae=False, vit=True):
        #   if vae:  # ✗ FALSE - SKIP VAE encoding!
        #       pass  # Not executed
        #
        #   if vit:  # ✓ True
        #       - Encode with SigLIP ViT only
        #       - Add [<|vision_start|>] + ViT_tokens + [<|vision_end|>]
        #       - Cache ViT tokens
```

**Why skip VAE?** VAE is only needed for image **generation**. For understanding (text output), we only need semantic ViT features.

**After processing image:**
```
gen_context:      [<|vision_start|> ViT_tokens <|vision_end|>] (NO VAE!)
cfg_text_context: [Same]
cfg_img_context:  []
```

#### Step 2: Process Text Prompt

```python
    if isinstance(input_term, str):
        cfg_text_context = deepcopy(gen_context)  # [ViT]
        gen_context = self.update_context_text("Explain this meme", gen_context)
        cfg_img_context = self.update_context_text("Explain this meme", cfg_img_context)
```

**After text:**
```
gen_context:      [ViT_tokens] + [<|im_start|> Explain... <|im_end|>]
cfg_text_context: [ViT_tokens] (no question)
cfg_img_context:  [<|im_start|> Explain... <|im_end|>] (no image)
```

#### Step 3: Generate Text Response

```python
        if understanding_output:  # ✓ TRUE!
            gen_text = self.gen_text(gen_context, max_length=1000, ...)
            # Inside gen_text():
            #   1. prepare_start_tokens(): Add <|im_start|>
            #   2. Autoregressive generation:
            #      - Forward pass with [ViT + Question] context
            #      - Generate: "The humor in this meme comes from..."
            #      - Stop at <|im_end|>
            #   3. Post-process: Remove <|im_start|> and <|im_end|>
            # gen_text = "The humor in this meme comes from..."

            output_list.append(gen_text)

        else:  # NOT executed
            pass  # Image generation skipped

    return output_list  # [<text_response>]
```

**Final Output:**
```python
return {'image': None, 'text': "The humor in this meme comes from..."}
```

---

## Token Sequence Examples

### Example 1: Simple Text-to-Image

**Input:** `"A cat"`

**Sequence in KV Cache:**
```
┌─────────────────────────────────────────┐
│ Input Processing                         │
├─────────────────────────────────────────┤
│ <|im_start|> A cat <|im_end|>           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Image Generation                         │
├─────────────────────────────────────────┤
│ <|vision_start|>                        │
│   IMG_TOK_1                             │
│   IMG_TOK_2                             │
│   ...                                    │
│   IMG_TOK_N                             │
│ <|vision_end|>                          │
└─────────────────────────────────────────┘
```

### Example 2: Image Editing

**Input:** `original_image + "Make it red"`

**Sequence in KV Cache:**
```
┌─────────────────────────────────────────┐
│ Input Image (VAE)                        │
├─────────────────────────────────────────┤
│ <|vision_start|>                        │
│   VAE_TOK_1 (clean, t=0)                │
│   VAE_TOK_2                             │
│   ...                                    │
│ <|vision_end|>                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Input Image (ViT)                        │
├─────────────────────────────────────────┤
│ <|vision_start|>                        │
│   VIT_TOK_1                             │
│   VIT_TOK_2                             │
│   ...                                    │
│ <|vision_end|>                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Edit Instruction                         │
├─────────────────────────────────────────┤
│ <|im_start|> Make it red <|im_end|>     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Generated Edited Image                   │
├─────────────────────────────────────────┤
│ <|vision_start|>                        │
│   GEN_IMG_TOK_1                         │
│   GEN_IMG_TOK_2                         │
│   ...                                    │
│ <|vision_end|>                          │
└─────────────────────────────────────────┘
```

### Example 3: Multi-Turn Conversation

**Turn 1:** `"Draw a cat"` → Image
**Turn 2:** `"Make it orange"` → Edited image
**Turn 3:** `"Describe it"` → Text response

**Full Sequence:**
```
┌──────────────────────────────────────────────────────┐
│ Turn 1: User Request                                  │
├──────────────────────────────────────────────────────┤
│ <|im_start|> Draw a cat <|im_end|>                   │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Turn 1: AI Response (Image)                           │
├──────────────────────────────────────────────────────┤
│ <|vision_start|> [IMAGE_1_TOKENS] <|vision_end|>     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Turn 2: User Request                                  │
├──────────────────────────────────────────────────────┤
│ <|im_start|> Make it orange <|im_end|>               │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Turn 2: AI Response (Edited Image)                    │
├──────────────────────────────────────────────────────┤
│ <|vision_start|> [IMAGE_2_TOKENS] <|vision_end|>     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Turn 3: User Request                                  │
├──────────────────────────────────────────────────────┤
│ <|im_start|> Describe it <|im_end|>                  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Turn 3: AI Response (Text)                            │
├──────────────────────────────────────────────────────┤
│ <|im_start|> This is an orange cat... <|im_end|>     │
└──────────────────────────────────────────────────────┘
```

**Note:** In `auto_interleaved_demo.py`, the wrapper accumulates this entire sequence and re-passes it on each call. This is **inefficient** but simple.

---

## Critical Implementation Details

### 1. Automatic Token Wrapping

**All tokens are added automatically** by `prepare_*` methods:

```python
# Text: prepare_prompts() adds <|im_start|> and <|im_end|>
text_ids = [bos_token_id] + tokenizer.encode(prompt) + [eos_token_id]

# Images: prepare_vit_images()/prepare_vae_images() add vision markers
packed_text_ids.append(start_of_image)
# ... image tokens ...
packed_text_ids.append(end_of_image)
```

**Users never manually add these tokens!**

### 2. VAE vs ViT for Input Images

| Mode | VAE | ViT | Reason |
|------|-----|-----|--------|
| **Image Generation** | ✅ | ✅ | VAE provides conditioning for generation |
| **Image Editing** | ✅ | ✅ | VAE provides structure, ViT provides semantics |
| **Understanding** | ❌ | ✅ | Only need semantic understanding, not generation |

```python
# inferencer.py:250
gen_context = self.update_context_image(
    input_term,
    gen_context,
    vae=not understanding_output  # Skip VAE if understanding
)
```

### 3. Clean VAE vs Noised VAE

**During Inference:**
- Only **clean VAE** (t=0, no noise) is cached
- Noised VAE is **never stored** in KV cache
- Clean VAE tokens provide conditioning for generation

**During Training:**
- Both clean (t=0) and noised (t>0) VAE are used
- Noised tokens have MSE loss for learning diffusion
- Clean tokens provide conditioning (like inference)

### 4. CFG Context Snapshots

Three contexts are maintained with strategic snapshots:

```python
gen_context = {all inputs}          # Full context
cfg_text_context = {without last text}  # Saved BEFORE text added
cfg_img_context = {without images}      # Never gets images
```

**When snapshots are taken:**

```python
for input in inputs:
    if is_text:
        cfg_text_context = deepcopy(gen_context)  # Save BEFORE text
        gen_context.add_text(input)
        cfg_img_context.add_text(input)

    elif is_image:
        gen_context.add_image(input)
        cfg_text_context = deepcopy(gen_context)  # Save AFTER image
        # cfg_img_context does NOT get image
```

### 5. Text Generation Stop Condition

**Only `<|im_end|>` stops generation:**

```python
# modeling/bagel/bagel.py:996
if end_token_id is not None and curr_tokens[0] == end_token_id:
    break  # Stop autoregressive loop
```

No other mechanism stops text generation (besides max_length).

### 6. Post-Processing Removes Markers

```python
# Raw output: "<|im_start|>The answer is 42<|im_end|>"
output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
# Clean output: "The answer is 42"
```

Users receive clean text without special tokens.

### 7. Position Tracking

Special tokens **consume positions** in the sequence:

```python
# Text: [<|im_start|>] [tok1] [tok2] [<|im_end|>]
# Positions: 0            1      2      3

# Image: [<|vision_start|>] [img_tok1] ... [img_tokN] [<|vision_end|>]
# Positions: 0                1             N           N+1
```

But RoPE position increments by only 1 for entire image:

```python
# inferencer.py:343, 465
new_rope.append(curr_position_id + 1)  # Only +1, not +num_img_tokens
```

This is because images use 2D position embeddings, not RoPE.

---

## Summary: Token Usage Matrix

| Scenario | `<|im_start|>` | `<|im_end|>` | `<|vision_start|>` | `<|vision_end|>` |
|----------|----------------|--------------|-------------------|------------------|
| **T2I** | Before/after text prompt | Before/after text prompt | Before/after generated image | Before/after generated image |
| **Editing** | Before/after instruction | Before/after instruction | Before/after input VAE, ViT, output image | Before/after input VAE, ViT, output image |
| **Understanding** | Before/after question & answer | Before/after question & answer | Before/after ViT tokens | Before/after ViT tokens |
| **Think Mode** | Around system prompt, user prompt, thinking, response | Around system prompt, user prompt, thinking, response | Same as above | Same as above |

---

## Key Takeaways

1. **All tokens are automatic** - users never manually add them
2. **Four tokens, two purposes** - text markers (`<|im_*|>`) and vision markers (`<|vision_*|>`)
3. **`<|im_end|>` is critical** - only token that stops text generation
4. **VAE conditional** - only used when generating/editing images, skipped for understanding
5. **CFG contexts** - strategic snapshots enable dual text+image guidance
6. **Clean representations** - only clean (t=0) VAE cached during inference
7. **Post-processing** - special tokens removed from final outputs
8. **Position embedding** - images increment RoPE by 1, not by token count

These special tokens are the **glue** that enables BAGEL's unified autoregressive architecture to seamlessly mix text and images in a single sequence!
