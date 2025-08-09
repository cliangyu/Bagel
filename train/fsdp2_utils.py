# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
FSDP v2 utilities for BAGEL training.
Implementation follows: https://gist.github.com/jd-nuva/e211717a95fdcc2e025e45be8d170324
"""

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import (
    fully_shard,
    register_fsdp_forward_method,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
)
from torch.distributed.tensor import DeviceMesh, distribute_tensor


def get_fsdp_v2_policy(fsdp_config, cpu_offload=False):
    """Get FSDP v2 policy for sharding, mixed precision, and CPU offload."""
    world_size = dist.get_world_size()
    device_mesh = DeviceMesh("cuda", torch.arange(world_size))

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    offload_policy = (
        CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy()
    )

    return device_mesh, mp_policy, offload_policy


def shard_model_fsdp_v2(model_cls, bagel_init_params, fsdp_config, ignored_modules=[], cpu_offload=False):
    device_mesh, mp_policy, offload_policy = get_fsdp_v2_policy(
        fsdp_config, cpu_offload
    )

    with torch.device("meta"):
        model = model_cls(**bagel_init_params)

    # Shard individual layers - following gist pattern exactly
    for layer in model.language_model.model.layers:
        fully_shard(
            layer,
            mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
        register_fsdp_forward_method(layer, "forward_train")
        register_fsdp_forward_method(layer, "forward_inference")

    # Shard the top-level model
    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    # Register additional model-specific forward methods (from gist)
    register_fsdp_forward_method(model, "forward_cache_update_text")
    register_fsdp_forward_method(model, "generate_image")

    return model


def load_checkpoint_fsdp_v2(model, checkpoint_path, logger):
    """
    Load checkpoint into FSDP v2 model using proper DTensor distribution.
    Following the gist pattern exactly.
    """
    import os
    from safetensors.torch import load_file
    import torch.nn as nn
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.info("Training from scratch with FSDP v2")
        return model
    
    try:
        logger.info(f"Loading FSDP v2 checkpoint from {checkpoint_path}")
        
        # Determine checkpoint file path - prefer ema.safetensors as in gist
        if os.path.isdir(checkpoint_path):
            state_dict_path = os.path.join(checkpoint_path, "ema.safetensors")
            if not os.path.exists(state_dict_path):
                state_dict_path = os.path.join(checkpoint_path, "model.safetensors")
        else:
            state_dict_path = checkpoint_path
            
        if not os.path.exists(state_dict_path):
            logger.warning(f"Checkpoint file not found: {state_dict_path}")
            return model
        
        # Load full state dict to CPU (following gist pattern)
        full_sd = load_file(state_dict_path, device="cpu")
        
        # Get model's sharded state dict structure
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        
        for param_name, full_tensor in full_sd.items():
            # Skip parameters not in current model
            if param_name not in meta_sharded_sd:
                continue
                
            # Distribute tensor across device mesh
            sharded_meta_param = meta_sharded_sd[param_name]
            if hasattr(sharded_meta_param, "device_mesh"):
                sharded_tensor = distribute_tensor(
                    full_tensor.to(sharded_meta_param.dtype),
                    device_mesh=sharded_meta_param.device_mesh,
                    placements=sharded_meta_param.placements,
                )
                sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        
        # Load distributed state dict
        incompatible_keys = model.load_state_dict(
            sharded_sd, assign=True, strict=False
        )
        
    except Exception as e:
        logger.error(f"Failed to load FSDP v2 checkpoint: {e}")
        logger.info("Continuing with randomly initialized model")
        
    return model