# BAGEL Setup Scripts

Scripts to help set up your BAGEL development and training environment.

## Quick Start

### 1. Configure Environment Variables

Copy the example environment file and customize it:

```bash
cp scripts/setup/example.env .env
```

Edit `.env` to match your system:
- Update `BAGEL_DATA_DIR` to point to your data storage location
- Update `BAGEL_MODEL_DIR` to point to your model weights
- Adjust `CUDA_VISIBLE_DEVICES` for your GPU setup

### 2. Create Data Directory Structure

Run the setup script to create the external data directories:

```bash
# Option 1: Use BAGEL_DATA_DIR from .env
source .env
./scripts/setup/setup_data_dirs.sh

# Option 2: Specify directory directly
./scripts/setup/setup_data_dirs.sh /path/to/your/data
```

This creates:
```
bagel-data/
├── datasets/      # Production datasets
│   ├── t2i/       # Text-to-image parquet files
│   ├── editing/   # Image editing datasets
│   └── vlm/       # Vision-language model datasets
├── samples/       # Example/test data
├── outputs/       # Training outputs and checkpoints
└── test_images/   # Test images for inference
```

### 3. Load Environment for Each Session

Before running training or inference:

```bash
source .env
```

Or add to your `~/.bashrc` or `~/.zshrc`:
```bash
if [ -f /path/to/Bagel/.env ]; then
    source /path/to/Bagel/.env
fi
```

## Files

- **example.env** - Template environment configuration file
- **setup_data_dirs.sh** - Script to create external data directory structure
- **README.md** - This file

## Data Management Strategy

BAGEL uses an **external data directory** approach:

- **Code repository** (`Bagel/`): Only code, configs, and documentation
- **Data directory** (`bagel-data/`): All datasets, outputs, and large files
- **Model weights**: Stored separately, referenced via environment variables

### Benefits:

1. **Clean repository**: No large files in git
2. **Shared storage**: Multiple users can share the same data directory
3. **Flexible paths**: Easy to adapt to different systems
4. **Version control**: Only track code changes, not data

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `BAGEL_DATA_DIR` | Root directory for all data | `/mnt/shared/bagel-data` |
| `BAGEL_MODEL_PATH` | Pretrained BAGEL model | `/models/BAGEL-7B-MoT` |
| `BAGEL_LLM_PATH` | Language model checkpoint | `/models/Qwen2.5-0.5B-Instruct` |
| `BAGEL_VAE_PATH` | VAE checkpoint | `/models/flux/vae/ae.safetensors` |
| `BAGEL_VIT_PATH` | Vision transformer | `/models/siglip-...` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0,1,2,3` |

## Troubleshooting

### Data paths not found
- Ensure `.env` is sourced: `source .env`
- Check `echo $BAGEL_DATA_DIR` shows correct path
- Verify data exists in the expected locations

### Permission errors
- Ensure you have write access to `BAGEL_DATA_DIR`
- On shared systems, check directory permissions

### Model not found
- Update model paths in `.env`
- Download model weights from HuggingFace if needed
