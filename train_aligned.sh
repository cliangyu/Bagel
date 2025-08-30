#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Aligned with official script but customized for:
# 1. 4 GPUs (1,2,3,4)
# 2. W&B offline mode
# 3. Save every 2 steps
# 4. Train on both gen and und tasks

# Environment setup
export PATH=/home/leonlc/.conda/envs/grpo/bin:$PATH
export PYTHONPATH=/data/users/leonlc/fsdp_bagel/Bagel:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3,4,5

# Variables (following official script structure)
num_nodes=1
node_rank=0
nproc_per_node=4  # Using 4 GPUs instead of 8
master_addr=localhost
master_port=29500

# Model component paths (for pretraining style)
llm_path="hf/Qwen2.5-0.5B-Instruct"
vae_path="flux/vae/ae.safetensors"
vit_path="hf/siglip-so400m-14-980-flash-attn2-navit"

# For finetuning from existing model
model_path="/data/users/leonlc/BAGEL-7B-MoT"
# Resume from specific checkpoint
# resume_from="/data/users/leonlc/bagel_output/aligned_20250829_103553/checkpoints/0000006"
resume_from="/data/users/leonlc/bagel_output/aligned_20250830_083509/hf_0000006_final/"
# resume_from={model_path}

# Output paths
output_path="/data/users/leonlc/bagel_output/aligned_$(date +%Y%m%d_%H%M%S)"
ckpt_path="${output_path}/checkpoints"

# Create directories
mkdir -p ${output_path}
mkdir -p ${ckpt_path}

echo "Starting BAGEL training (aligned with official script)..."
echo "Using ${nproc_per_node} GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Output: ${output_path}"

# Run training (aligned with official script structure)
# --finetune_from_hf True always True
torchrun \
  --nnodes=${num_nodes} \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --master_addr=${master_addr} \
  --master_port=${master_port} \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path ${model_path} \
  --llm_path ${llm_path} \
  --vae_path ${vae_path} \
  --vit_path ${vit_path} \
  --use_flex True \
  --resume_from ${resume_from} \
  --finetune_from_hf True \
  --finetune_from_ema False \
  --resume_model_only True \
  --auto_resume True \
  --results_dir ${output_path} \
  --checkpoint_dir ${ckpt_path} \
  --max_latent_size 64 \
  --num_workers 1 \
  --num_shard 4 \
  --visual_gen True \
  --visual_und True \
  --save_every 2 \
  --log_every 1 \
  --wandb_offline True \
  --wandb_project bagel_aligned \
  --wandb_name aligned_run_$(date +%Y%m%d_%H%M%S) \
  --lr 2e-5 \
  --total_steps 100 \
  --warmup_steps 10 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240

echo "Training completed!"
echo "Results saved to: ${output_path}"
