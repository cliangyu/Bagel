# BAGEL Documentation

Comprehensive documentation for the BAGEL multimodal foundation model.

## Quick Start

- [Main README](../README.md) - Project overview and quick start guide

## Documentation Structure

### 🏋️ Training
- [TRAIN.md](training/TRAIN.md) - Training guide and configuration
- [CHECKPOINT_CONVERSION.md](training/CHECKPOINT_CONVERSION.md) - Converting between checkpoint formats
- [CHECKPOINT_CONVERSION_TRAIN.md](training/CHECKPOINT_CONVERSION_TRAIN.md) - Training-specific checkpoint details
- [DIFFUSION_FORCING_GUIDE.md](training/DIFFUSION_FORCING_GUIDE.md) - Diffusion forcing techniques

### 📊 Evaluation
- [EVAL.md](evaluation/EVAL.md) - Evaluation benchmarks and metrics

### 🔮 Inference
- [INFERENCE_GUIDE.md](inference/INFERENCE_GUIDE.md) - Running inference with BAGEL
- **[AUTO_SEQUENCE_GENERATION_GUIDE.md](inference/AUTO_SEQUENCE_GENERATION_GUIDE.md)** - **How to use automatic sequence generation** ⭐ NEW
- **[IMPLEMENTATION_SUMMARY.md](inference/IMPLEMENTATION_SUMMARY.md)** - **Summary of what was implemented** 📋 NEW
- [SPLIT_START_END_EXPLAINED.md](inference/SPLIT_START_END_EXPLAINED.md) - Understanding split_start/split_end parameters
- **[INFERENCE_DEEP_DIVE_SUMMARY.md](inference/INFERENCE_DEEP_DIVE_SUMMARY.md)** - **Summary of deep dive investigation** ⭐
- [SPECIAL_TOKENS_AND_EXECUTION_PATHS.md](inference/SPECIAL_TOKENS_AND_EXECUTION_PATHS.md) - Deep dive into special tokens and detailed execution traces

### 🏗️ Architecture
- [IMPLEMENTATION_DETAILS.md](architecture/IMPLEMENTATION_DETAILS.md) - Core implementation details
- [flow_matching_and_vae_analysis.md](architecture/flow_matching_and_vae_analysis.md) - Flow matching and VAE analysis

### 📁 Data
- [DATASET_TYPES_REFERENCE.md](data/DATASET_TYPES_REFERENCE.md) - Dataset types and formats

## Contributing

When adding new documentation:
1. Place it in the appropriate category directory
2. Update this README with a link
3. Use clear, descriptive filenames
4. Follow the existing markdown style
