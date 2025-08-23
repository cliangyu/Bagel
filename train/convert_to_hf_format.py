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
        print(f"Converting checkpoint from {args.checkpoint_dir} to {args.output_path}")
        print(f"Using model path: {args.model_path}")
        print(f"Matching original tensor structure: {'Yes' if args.match_original else 'No'}")
    
    # Load configs exactly as they exist in the original model
    llm_config = Qwen2Config.from_json_file(
        os.path.join(args.model_path, "llm_config.json")
    )
    
    # Set the layer_module to match original (usually Qwen2MoTDecoderLayer)
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    if args.match_original:
        # Check if original model has qk_norm by looking at the original EMA model
        from safetensors import safe_open
        ema_path = os.path.join(args.model_path, "ema.safetensors")
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
    else:
        llm_config.qk_norm = False
    
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    
    # Create language model
    language_model = Qwen2ForCausalLM(llm_config)
    
    # Create vision model with original layer count
    vit_model = None
    vit_config_path = os.path.join(args.model_path, "vit_config.json")
    if os.path.exists(vit_config_path):
        vit_config = SiglipVisionConfig.from_json_file(vit_config_path)
        
        if args.match_original:
            # Keep original layer count (don't subtract 9)
            # The original config has the correct number of layers
            original_layers = vit_config.num_hidden_layers
            if rank == 0:
                print(f"Vision model layers: {original_layers}")
        else:
            # Apply the training script's adjustment
            vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 - 9
        
        vit_config.rope = False
        vit_model = SiglipVisionModel(vit_config)
    
    # Load VAE config if exists
    vae_config = None
    ae_path = os.path.join(args.model_path, "ae.safetensors")
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
    
    # Load the sharded checkpoint
    model, _, _, _, _, _ = ShardedCheckpointManager.load_sharded_checkpoint(
        resume_from=args.checkpoint_dir,
        model=model,
        ema_model=None,
        optimizer=None,
        scheduler=None,
        logger=logger,
        load_model=not args.save_ema,
        load_ema=args.save_ema,
    )
    
    # Convert to full state dict and save on rank 0
    if rank == 0:
        print("Gathering full model state dict...")
    
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state_dict = model.state_dict()
        
        if rank == 0:
            # Save the full model
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving to {output_path}")
            print(f"Total tensors: {len(state_dict)}")
            
            # Show structure comparison if matching original
            if args.match_original:
                ema_path = os.path.join(args.model_path, "ema.safetensors")
                if os.path.exists(ema_path):
                    from safetensors import safe_open
                    with safe_open(ema_path, framework='pt', device='cpu') as f:
                        orig_keys = set(f.keys())
                        conv_keys = set(state_dict.keys())
                        print(f"Original model tensors: {len(orig_keys)}")
                        print(f"Converted model tensors: {len(conv_keys)}")
                        print(f"Missing tensors: {len(orig_keys - conv_keys)}")
                        print(f"Extra tensors: {len(conv_keys - orig_keys)}")
            
            if str(output_path).endswith('.safetensors'):
                save_file(state_dict, str(output_path))
            else:
                torch.save(state_dict, str(output_path))
            
            print("Conversion complete!")
    
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
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted model (e.g., model.safetensors)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory with config files (e.g., /data/users/leonlc/BAGEL-7B-MoT)",
    )
    parser.add_argument(
        "--save_ema",
        action="store_true",
        help="Save EMA model instead of main model",
    )
    parser.add_argument(
        "--match_original",
        action="store_true",
        help="Match original model structure exactly (include all layers, q/k norms)",
    )
    
    args = parser.parse_args()
    convert_checkpoint(args)


if __name__ == "__main__":
    main()