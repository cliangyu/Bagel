# Checkpoint Conversion Guide

This guide explains how to convert sharded FSDP checkpoints to complete HuggingFace format directories using the enhanced checkpoint management system.

## ⚠️ CRITICAL WARNING ⚠️

**NEVER REPLACE THE BASE MODEL `/data/users/leonlc/BAGEL-7B-MoT`**

Always create new model directories with unique names. The base model should remain untouched and serve as the reference for all conversions.

## Enhanced Checkpoint System

The checkpoint system now supports automatic format detection and flexible loading:

### Supported Formats
- **Sharded checkpoints**: Memory-efficient FSDP shards with metadata
- **HuggingFace format**: Standard safetensors/pytorch_model.bin files
- **Legacy checkpoints**: Backward compatibility with old formats

### Format Detection
The system automatically detects checkpoint format by examining:
- `metadata.json` for sharded checkpoints (contains `checkpoint_type: "sharded"`)
- Standard HF files (`model.safetensors`, `config.json`, etc.)
- Direct safetensors files

## Conversion Process

The conversion script creates a complete HuggingFace model directory with all necessary files:
- `ema.safetensors` - EMA model weights  
- `model.safetensors` - Main model weights (if needed)
- `ae.safetensors` - VAE weights (copied from base model)
- All config files, tokenizer files, etc.

## ShardedCheckpointManager Features

The enhanced `ShardedCheckpointManager` provides:

### Loading Capabilities
- **`load_sharded_checkpoint()`**: Loads sharded FSDP checkpoints with selective loading
  - `load_model=False` to skip main model loading
  - `load_ema=False` to skip EMA model loading
  - Automatic fallback for legacy formats
- **`load_hf_checkpoint()`**: Loads HuggingFace format checkpoints
  - `load_ema_from_main=True` to initialize EMA from main model
  - Flexible file detection (safetensors, pytorch_model.bin, model.pt)
- **`detect_checkpoint_format()`**: Automatic format detection

### Saving Capabilities  
- **`save_sharded_checkpoint()`**: Memory-efficient sharded saving
  - Prevents OOM during checkpoint creation
  - Supports both PyTorch DCP and fallback modes
  - EMA metadata tracking with improved logging

### Setup Environment

```bash
export PATH=/home/leonlc/.conda/envs/grpo/bin:$PATH
export PYTHONPATH=/data/users/leonlc/fsdp_bagel/Bagel:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2,3,4
```

### Convert to Complete HuggingFace Directory

```bash
# Convert both models (default and recommended)
torchrun --nproc_per_node=4 --master_port=12357 \
    train/convert_to_hf_format.py \
    --checkpoint_dir "/path/to/sharded/checkpoint" \
    --output_dir "/path/to/output/NEW_MODEL_DIR" \
    --base_model_path "/data/users/leonlc/BAGEL-7B-MoT"

# Or specify which models to convert:
# Convert EMA model only
torchrun --nproc_per_node=4 --master_port=12357 \
    train/convert_to_hf_format.py \
    --checkpoint_dir "/path/to/sharded/checkpoint" \
    --output_dir "/path/to/output/NEW_MODEL_DIR" \
    --base_model_path "/data/users/leonlc/BAGEL-7B-MoT" \
    --models ema

# Convert main model only
torchrun --nproc_per_node=4 --master_port=12357 \
    train/convert_to_hf_format.py \
    --checkpoint_dir "/path/to/sharded/checkpoint" \
    --output_dir "/path/to/output/NEW_MODEL_DIR" \
    --base_model_path "/data/users/leonlc/BAGEL-7B-MoT" \
    --models main
```

## Example Usage

For checkpoint at `/data/users/leonlc/bagel_output/aligned_20250830_083509/checkpoints/0000006`:

```bash
# Convert both models to complete HF directory (default behavior)
torchrun --nproc_per_node=4 --master_port=12357 \
    train/convert_to_hf_format.py \
    --checkpoint_dir "/data/users/leonlc/bagel_output/aligned_20250830_083509/checkpoints/0000006" \
    --output_dir "/data/users/leonlc/bagel_output/aligned_20250830_083509/hf_0000006" \
    --base_model_path "/data/users/leonlc/BAGEL-7B-MoT"
```

This creates a directory with the same structure as `/data/users/leonlc/BAGEL-7B-MoT`:
```
hf_0000006/
├── model.safetensors        # Main model weights (bf16)
├── ema.safetensors          # EMA model weights (bf16)
├── ae.safetensors           # VAE weights (copied from base)
├── config.json              # Model config
├── llm_config.json          # LLM config  
├── vit_config.json          # Vision config
├── tokenizer_config.json    # Tokenizer config
├── tokenizer.json           # Tokenizer
├── vocab.json               # Vocabulary
├── merges.txt               # BPE merges
└── model.safetensors.index.json  # Model index
```

### Enhanced Conversion Features

The improved conversion script now provides:
- **Perfect training alignment**: Uses identical model creation logic as training script
- **NavIT architecture matching**: Applies `convert_conv2d_to_linear()` exactly like training
- **Flexible model selection**: `--models` argument (main/ema/both, defaults to both)
- **Improved argument naming**: `--base_model_path` for clarity  
- **Sequential loading**: Avoids OOM by loading and saving models one at a time
- **Automatic structure matching**: Detects qk_norm layers and preserves exact tensor structure
- **Memory management**: Includes memory clearing and CUDA cache cleanup between conversions
- **Verification**: Shows NavIT format verification and tensor comparisons

## Notes

- The enhanced checkpoint system provides automatic format detection and flexible loading
- Sharded checkpoints include metadata for better reliability and debugging
- Model weights are saved in bfloat16 (smaller than the old FP32 checkpoints)  
- Always matches original model structure for perfect compatibility
- Use `--models both` (default) to convert both models in one run
- Use `--models ema` to convert only EMA model (for inference)
- Use `--models main` to convert only main model
- The script automatically handles memory management to prevent OOM issues
- Model creation now perfectly matches training script architecture
- The system supports PyTorch Distributed Checkpoint API when available, with fallback for older versions

### Troubleshooting

If conversion fails:
1. Check checkpoint format with `ShardedCheckpointManager.detect_checkpoint_format()`
2. Ensure the checkpoint directory contains valid metadata.json for sharded format
3. Verify CUDA_VISIBLE_DEVICES matches the number of processes in torchrun