#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Convert sharded FSDP checkpoint to proper HuggingFace format that matches
the original model structure exactly.

This script ensures the converted model has the same tensor structure as
the original HuggingFace model, including:
- Correct vision model layer count
- Q/K norm layers if present in original
- Proper module naming scheme
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from safetensors.torch import save_file

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from train.fsdp_utils import fsdp_wrapper, FSDPConfig
from train.fsdp_checkpoint_manager import ShardedCheckpointManager


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
    else:
        # Single process mode
        rank = 0
        world_size = 1
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    return rank, world_size


def convert_checkpoint(args):
    """Convert checkpoint to HF format matching original model structure."""
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"Converting checkpoint from {args.checkpoint_dir} to {args.output_dir}")
        print(f"Using base model path: {args.base_model_path}")
        print("Matching original tensor structure: Yes")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files from base_model_path except model weights
        files_to_copy = [
            "config.json", "generation_config.json", "llm_config.json", "vit_config.json",
            "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", 
            "model.safetensors.index.json", "README.md"
        ]
        
        for file_name in files_to_copy:
            src = os.path.join(args.base_model_path, file_name)
            if os.path.exists(src):
                dst = output_dir / file_name
                shutil.copy2(src, dst)
                print(f"Copied {file_name}")
        
        # Copy ae.safetensors from base_model_path
        ae_src = os.path.join(args.base_model_path, "ae.safetensors")
        if os.path.exists(ae_src):
            ae_dst = output_dir / "ae.safetensors"
            shutil.copy2(ae_src, ae_dst)
            print("Copied ae.safetensors")
    
    # Load configs exactly as they exist in the original model
    llm_config = Qwen2Config.from_json_file(
        os.path.join(args.base_model_path, "llm_config.json")
    )
    
    # Set the layer_module to match original (usually Qwen2MoTDecoderLayer)
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # Always match original model structure
    # Check if original model has qk_norm by looking at the original EMA model
    from safetensors import safe_open
    ema_path = os.path.join(args.base_model_path, "ema.safetensors")
    if os.path.exists(ema_path):
        with safe_open(ema_path, framework='pt', device='cpu') as f:
            original_keys = f.keys()
            # Check if q_norm exists in original
            has_qk_norm = any('q_norm' in key for key in original_keys)
            llm_config.qk_norm = has_qk_norm
            if rank == 0:
                print(f"Detected qk_norm in original model: {has_qk_norm}")
    else:
        llm_config.qk_norm = False
    
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    
    # Create language model
    language_model = Qwen2ForCausalLM(llm_config)
    
    # Create vision model with original layer count
    vit_model = None
    vit_config_path = os.path.join(args.base_model_path, "vit_config.json")
    if os.path.exists(vit_config_path):
        vit_config = SiglipVisionConfig.from_json_file(vit_config_path)
        
        # Always keep original layer count (don't subtract 9)
        # The original config has the correct number of layers
        original_layers = vit_config.num_hidden_layers
        if rank == 0:
            print(f"Vision model layers: {original_layers}")
        
        vit_config.rope = False
        vit_model = SiglipVisionModel(vit_config)
    
    # Load VAE config if exists
    vae_config = None
    ae_path = os.path.join(args.base_model_path, "ae.safetensors")
    if os.path.exists(ae_path):
        _, vae_config = load_ae(local_path=ae_path)
    
    # Create BagelConfig
    config = BagelConfig(
        visual_gen=True if vae_config else False,
        visual_und=True if vit_model else False,
        llm_config=llm_config,
        vit_config=vit_config if vit_model else None,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
    )
    
    # Create the Bagel model
    model = Bagel(language_model, vit_model, config)
    model = model.to(torch.cuda.current_device())
    
    # Apply NavIT conversion exactly like training script
    if vit_model is not None:
        if rank == 0:
            print("Applying NavIT conversion (Conv2d -> Linear) to match training architecture")
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    
    # Wrap with FSDP
    fsdp_config = FSDPConfig(
        sharding_strategy='FULL_SHARD',
        backward_prefetch='BACKWARD_PRE',
        cpu_offload=False,
        num_replicate=1,
        num_shard=world_size,
    )
    model = fsdp_wrapper(model, fsdp_config)
    
    # Create logger wrapper
    class DummyLogger:
        def info(self, msg):
            if rank == 0:
                print(msg)
        def warning(self, msg):
            if rank == 0:
                print(f"WARNING: {msg}")
        def error(self, msg):
            if rank == 0:
                print(f"ERROR: {msg}")
    
    logger = DummyLogger()
    
    def load_and_save_model(load_main=True, load_ema=False, model_type="Main"):
        """Helper function to load and save a specific model type"""
        # Load the sharded checkpoint
        loaded_model, _, _, _, _, _ = ShardedCheckpointManager.load_sharded_checkpoint(
            resume_from=args.checkpoint_dir,
            model=model,
            ema_model=None,
            optimizer=None,
            scheduler=None,
            logger=logger,
            load_model=load_main,
            load_ema=load_ema,
        )
        
        # NavIT conversion already applied during model creation
        
        # Convert to full state dict and save on rank 0
        if rank == 0:
            print(f"Gathering {model_type} model state dict...")
        
        with FSDP.state_dict_type(
            loaded_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            state_dict = loaded_model.state_dict()
            
            if rank == 0:
                # Verify NavIT format (should already be converted)
                patch_embed_key = 'vit_model.vision_model.embeddings.patch_embedding.weight'
                if patch_embed_key in state_dict:
                    shape = state_dict[patch_embed_key].shape
                    expected_navit = (1152, 588)
                    if shape == expected_navit:
                        print(f"✅ Verified NavIT format: {shape}")
                    else:
                        print(f"⚠️  Unexpected shape: {shape}, expected: {expected_navit}")
                
                output_dir = Path(args.output_dir)
                
                # Determine filename based on model type
                filename = "ema.safetensors" if load_ema else "model.safetensors"
                output_path = output_dir / filename
                
                print(f"Saving {model_type} model to {output_path}")
                print(f"Total tensors: {len(state_dict)}")
                
                # Show structure comparison with original
                orig_model_path = os.path.join(args.base_model_path, filename)
                if os.path.exists(orig_model_path):
                    from safetensors import safe_open
                    with safe_open(orig_model_path, framework='pt', device='cpu') as f:
                        orig_keys = set(f.keys())
                        conv_keys = set(state_dict.keys())
                        print(f"Original model tensors: {len(orig_keys)}")
                        print(f"Converted model tensors: {len(conv_keys)}")
                        print(f"Missing tensors: {len(orig_keys - conv_keys)}")
                        print(f"Extra tensors: {len(conv_keys - orig_keys)}")
                
                save_file(state_dict, str(output_path))
                print(f"{model_type} model conversion complete!")
        
        # Clear memory
        del state_dict
        torch.cuda.empty_cache()
        dist.barrier()
    
    # Handle different conversion modes
    if rank == 0:
        print(f"Converting: {args.models}")
    
    if args.models in ["main", "both"]:
        load_and_save_model(load_main=True, load_ema=False, model_type="Main")
    
    if args.models in ["ema", "both"]:
        load_and_save_model(load_main=False, load_ema=True, model_type="EMA")
    
    # Clean up
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Convert sharded FSDP checkpoint to HF format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to sharded checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the converted HF model (will contain ema.safetensors, ae.safetensors, configs, etc.)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to base model directory with configs (e.g., /data/users/leonlc/BAGEL-7B-MoT)",
    )
    parser.add_argument(
        "--models",
        choices=["main", "ema", "both"],
        default="both",
        help="Which models to convert (default: both)",
    )
    
    args = parser.parse_args()
    convert_checkpoint(args)


if __name__ == "__main__":
    main()