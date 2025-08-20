# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
FSDP v2 utilities for BAGEL training.
Implementation follows: https://gist.github.com/jd-nuva/e211717a95fdcc2e025e45be8d170324
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from copy import deepcopy
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import (
    fully_shard,
    register_fsdp_forward_method,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
)
from torch.distributed.tensor import DeviceMesh, distribute_tensor

# DCP API imports for enhanced checkpointing (available in PyTorch 2.8+)
try:
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict, 
        StateDictOptions,
    )
    DCP_AVAILABLE = True
except ImportError:
    # Fallback for older PyTorch versions
    DCP_AVAILABLE = False


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
    register_fsdp_forward_method(model, "forward_cache_update_vit")
    register_fsdp_forward_method(model, "forward_cache_update_vae")
    register_fsdp_forward_method(model, "generate_image")
    register_fsdp_forward_method(model, "_forward_flow")

    # Materialize parameters from meta device AFTER sharding (following official pattern)
    model.to_empty(device="cuda")

    return model


def load_checkpoint_fsdp_v2(model, checkpoint_path, logger, model_type="model", use_dcp_api=True):
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
        
        # Determine checkpoint file path based on model type
        if os.path.isdir(checkpoint_path):
            if model_type == "ema":
                # Try EMA checkpoint first, then fall back to main model
                state_dict_path = os.path.join(checkpoint_path, "ema.safetensors")
                if not os.path.exists(state_dict_path):
                    logger.warning(f"EMA checkpoint not found, using main model")
                    state_dict_path = os.path.join(checkpoint_path, "model.safetensors")
            else:
                # Try FSDP checkpoint format first, then HuggingFace format
                state_dict_path = os.path.join(checkpoint_path, "model.safetensors")
                if not os.path.exists(state_dict_path):
                    # Check for HuggingFace format (ema.safetensors as main model)
                    hf_path = os.path.join(checkpoint_path, "ema.safetensors")
                    if os.path.exists(hf_path):
                        state_dict_path = hf_path
                        logger.info(f"Using HuggingFace model format: {hf_path}")
        else:
            state_dict_path = checkpoint_path
            
        if not os.path.exists(state_dict_path):
            logger.warning(f"Checkpoint file not found: {state_dict_path}")
            return model
        
        # Load full state dict to CPU
        full_sd = load_file(state_dict_path, device="cpu")
        
        # NOTE: position embeds are fixed sinusoidal embeddings, so we can just pop it off,
        # which makes it easier to adapt to different resolutions (same as FSDP1)
        full_sd.pop('latent_pos_embed.pos_embed', None)
        full_sd.pop('vit_pos_embed.pos_embed', None)
        
        if use_dcp_api and DCP_AVAILABLE:
            # Enhanced DCP API approach (PyTorch official method)
            set_model_state_dict(
                model=model,
                model_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            logger.info(f"Loaded {model_type} model using DCP API")
        else:
            # Manual DTensor approach (following gist pattern)
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
                else:
                    # Handle non-DTensor parameters (buffers, etc.)
                    sharded_sd[param_name] = full_tensor
            
            # Load distributed state dict
            incompatible_keys = model.load_state_dict(
                sharded_sd, assign=True, strict=False
            )
            logger.info(f"Loaded {model_type} model using manual DTensor approach with incompatible keys: {incompatible_keys}")
        
    except Exception as e:
        logger.error(f"Failed to load FSDP v2 checkpoint: {e}")
        logger.info("Continuing with randomly initialized model")
        
    return model


def save_checkpoint_fsdp_v2(model, ema_model, optimizer, scheduler, checkpoint_dir, train_steps, logger, data_status=None, use_dcp_api=True):
    """
    Save FSDP v2 checkpoint using DTensor full tensor gathering for model weights 
    and traditional approach for optimizer states.
    
    Args:
        use_dcp_api: If True, use PyTorch's DCP API for model checkpointing (requires DCP_AVAILABLE).
                    If False, use manual DTensor approach (follows gist pattern).
    """
    save_path = os.path.join(checkpoint_dir, f"{train_steps:07d}")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving FSDP v2 checkpoint to {save_path} (DCP API: {use_dcp_api and DCP_AVAILABLE})")

    # Save main model - choose method based on use_dcp_api flag
    if use_dcp_api and DCP_AVAILABLE:
        # Enhanced DCP API approach (PyTorch official method)
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        if dist.get_rank() == 0:
            save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))
            logger.info("Saved main model state dict using DCP API")
    else:
        # Manual DTensor approach (following gist pattern)
        model_state_dict = {}
        for param_name, param in model.named_parameters():
            if hasattr(param, "full_tensor"):
                # DTensor - gather to full tensor on CPU
                full_tensor = param.full_tensor().to("cpu")
                model_state_dict[param_name] = full_tensor
            else:
                # Regular tensor - move to CPU
                model_state_dict[param_name] = param.data.to("cpu")
        
        # Include non-parameter state (buffers, etc.)
        for buffer_name, buffer in model.named_buffers():
            if hasattr(buffer, "full_tensor"):
                full_buffer = buffer.full_tensor().to("cpu")
                model_state_dict[buffer_name] = full_buffer
            else:
                model_state_dict[buffer_name] = buffer.data.to("cpu")

        if dist.get_rank() == 0:
            save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))
            logger.info("Saved main model state dict using manual DTensor approach")

    # Save parameter-level EMA values if they exist
    ema_state_dict = {}
    for param_name, param in model.named_parameters():
        if hasattr(param, '_ema_avg'):
            if hasattr(param._ema_avg, "full_tensor"):
                # DTensor - gather to full tensor on CPU
                ema_state_dict[param_name] = param._ema_avg.full_tensor().to("cpu")
            else:
                # Regular tensor - move to CPU
                ema_state_dict[param_name] = param._ema_avg.to("cpu")
    
    if len(ema_state_dict) > 0:
        if dist.get_rank() == 0:
            save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))
            logger.info("Saved parameter-level EMA state dict")

    # Save optimizer state - use rank-based naming for FSDP2 
    # (different from FSDP1 shard-based naming to distinguish implementations)
    if optimizer is not None:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        optimizer_state_path = os.path.join(save_path, f"optimizer_rank{rank:05d}_of{world_size:05d}.pt")
        torch.save(optimizer.state_dict(), optimizer_state_path)
        logger.info(f"Saved FSDP v2 optimizer state for rank {dist.get_rank()}")

    # Save scheduler state (rank 0 only)
    if scheduler is not None and dist.get_rank() == 0:
        torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        logger.info("Saved scheduler state")

    # Save data status (rank 0 only)
    if data_status is not None and dist.get_rank() == 0:
        torch.save(data_status, os.path.join(save_path, "data_status.pt"))
        logger.info("Saved data status")

    dist.barrier()
    logger.info(f"Completed FSDP v2 checkpoint save at step {train_steps}")


def load_model_and_optimizer_fsdp_v2(model, ema_model, optimizer, scheduler, checkpoint_path, logger):
    """
    Load complete FSDP v2 checkpoint including optimizer and scheduler states.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.info("Training from scratch with FSDP v2")
        return model, ema_model, optimizer, scheduler, 0, None

    try:
        logger.info(f"Loading complete FSDP v2 checkpoint from {checkpoint_path}")
        
        # Load main model
        model = load_checkpoint_fsdp_v2(model, checkpoint_path, logger, model_type="model")
        
        # Load parameter-level EMA values if they exist
        ema_checkpoint_path = os.path.join(checkpoint_path, "ema.safetensors")
        if os.path.exists(ema_checkpoint_path):
            try:
                ema_state_dict = load_file(ema_checkpoint_path, device="cpu")
                # Load EMA values back into parameter attributes
                for param_name, param in model.named_parameters():
                    if param_name in ema_state_dict:
                        # Move to same device as parameter and store as attribute
                        param._ema_avg = ema_state_dict[param_name].to(param.device)
                logger.info("Loaded parameter-level EMA values")
            except Exception as e:
                logger.warning(f"Failed to load EMA values: {e}")
        else:
            logger.info("No EMA checkpoint found, EMA values initialized from current parameters")

        # Load optimizer state (FSDP2 rank-based naming)
        if optimizer is not None:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            optimizer_path = os.path.join(checkpoint_path, f"optimizer_rank{rank:05d}_of{world_size:05d}.pt")
            if os.path.exists(optimizer_path):
                optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(optimizer_state)
                logger.info(f"Loaded FSDP v2 optimizer state for rank {dist.get_rank()}")
            else:
                logger.warning(f"FSDP v2 optimizer checkpoint not found for rank {dist.get_rank()}")

        # Load scheduler state (rank 0 only)
        if scheduler is not None:
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.exists(scheduler_path):
                scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
                scheduler.load_state_dict(scheduler_state)
                logger.info("Loaded scheduler state")
            else:
                logger.warning("Scheduler checkpoint not found")

        # Load data status
        data_status = None
        data_status_path = os.path.join(checkpoint_path, "data_status.pt")
        if os.path.exists(data_status_path):
            data_status = torch.load(data_status_path, map_location="cpu", weights_only=True)
            local_rank = dist.get_rank()
            if local_rank < len(data_status):
                data_status = data_status[local_rank]
            else:
                data_status = None
            logger.info("Loaded data status")

        # Extract training step from checkpoint path
        train_steps = int(os.path.basename(os.path.normpath(checkpoint_path))) + 1
        
        return model, ema_model, optimizer, scheduler, train_steps, data_status

    except Exception as e:
        logger.error(f"Failed to load complete FSDP v2 checkpoint: {e}")
        logger.info("Continuing with current model state")
        return model, ema_model, optimizer, scheduler, 0, None


def fsdp2_ema_setup(fsdp_model):
    """
    Setup parameter-level EMA for FSDP v2.
    
    Instead of creating a separate EMA model (which doesn't work with deepcopy in FSDP2),
    we store EMA values as attributes on each parameter of the main FSDP2 model.
    
    This approach is compatible with FSDP2 DTensor and avoids deepcopy issues.
    """
    if dist.get_rank() == 0:
        print("Setting up FSDP v2 parameter-level EMA...")
    
    # Initialize EMA values for each parameter that requires gradients
    for name, param in fsdp_model.named_parameters():
        if param.requires_grad:
            # Store EMA value as a parameter attribute
            # Use clone().detach() to create a copy that doesn't require gradients
            param._ema_avg = param.data.clone().detach()
            # Explicitly ensure EMA values never require gradients
            param._ema_avg.requires_grad_(False)
    
    if dist.get_rank() == 0:
        print("✅ FSDP v2 parameter-level EMA setup completed")
    
    return fsdp_model


@torch.no_grad()
def fsdp2_ema_update(ema_model, model, decay=0.9999):
    """
    Update parameter-level EMA for FSDP v2.
    
    Since we don't have a separate EMA model in FSDP2, we update the EMA values
    stored as parameter attributes on the main model.
    
    Args:
        ema_model: Ignored for FSDP2 (should be None)
        model: The main FSDP2 model with parameter-level EMA
        decay: EMA decay rate
    """
    # Parameter-level EMA update for FSDP2
    for param in model.parameters():
        if param.requires_grad and hasattr(param, '_ema_avg'):
            # Update EMA: ema = decay * ema + (1 - decay) * current
            param._ema_avg.mul_(decay).add_(param.data, alpha=1 - decay)