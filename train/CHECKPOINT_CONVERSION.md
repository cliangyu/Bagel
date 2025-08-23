# BAGEL Checkpoint Conversion Guide

This guide explains how to convert BAGEL sharded FSDP checkpoints to HuggingFace format.

## Quick Start

Convert a sharded checkpoint to HuggingFace format:

```bash
# Setup environment
export PATH=/home/leonlc/.conda/envs/grpo/bin:$PATH
export PYTHONPATH=/data/users/leonlc/fsdp_bagel/Bagel:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Convert main model
torchrun --nproc_per_node=4 --master_port=12356 \
    train/convert_to_hf_format.py \
    --checkpoint_dir "/path/to/sharded/checkpoint" \
    --output_path "/path/to/output/model.safetensors" \
    --model_path "/data/users/leonlc/BAGEL-7B-MoT" \
    --match_original

# Convert EMA model
torchrun --nproc_per_node=4 --master_port=12357 \
    train/convert_to_hf_format.py \
    --checkpoint_dir "/path/to/sharded/checkpoint" \
    --output_path "/path/to/output/ema.safetensors" \
    --model_path "/data/users/leonlc/BAGEL-7B-MoT" \
    --match_original \
    --save_ema
```

## Key Features

### ✅ What Works
- **Sharded FSDP Checkpoint Loading**: Supports both model and EMA weights
- **HuggingFace Compatibility**: Output matches original BAGEL tensor structure exactly
- **Complete Architecture**: Language model (28 layers) + Vision model (27 layers) + VAE
- **Distributed Conversion**: Uses 4 GPUs for fast processing

### 📁 Output Files
- `model.safetensors` (55GB): Main trained model weights
- `ema.safetensors` (55GB): EMA model weights  
- Copy over config files: `config.json`, `llm_config.json`, `vit_config.json`, tokenizer files

### 🔧 Options
- `--match_original`: Match original HF model structure exactly (recommended)
- `--save_ema`: Convert EMA weights instead of main model weights

## Implementation Details

### Key Files
- `convert_to_hf_format.py`: Main conversion script
- `fsdp_checkpoint_manager.py`: Handles sharded checkpoint loading with EMA metadata fix
- `pretrain_unified_navit.py`: Training script with automatic checkpoint format detection

### Tensor Structure
- **Total Tensors**: 1,239 (matches original HF model)
- **Language Model**: 788 tensors (Qwen2 + MoE + q/k norms)
- **Vision Model**: 421 tensors (SigLIP, 27 layers)
- **Connectors**: 14 tensors (multimodal bridges)
- **Embeddings**: 2 tensors (positional embeddings)

## Troubleshooting

### Common Issues
1. **Missing .metadata files**: Fixed automatically by `ShardedCheckpointManager`
2. **Tensor count mismatch**: Use `--match_original` flag
3. **CUDA OOM**: Reduce batch size or use fewer GPUs

### Verification
Check tensor count matches original:
```python
from safetensors import safe_open
with safe_open('model.safetensors', framework='pt') as f:
    print(f"Converted tensors: {len(f.keys())}")  # Should be 1,239
```

## Example Usage

See working example at:
- Input: `/data/users/leonlc/bagel_output/aligned_20250822_233613/checkpoints/0000002`
- Output: Compatible with `transformers.AutoModel.from_pretrained()`