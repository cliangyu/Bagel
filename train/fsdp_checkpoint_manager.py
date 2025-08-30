# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
try:
    # PyTorch 2.0+ distributed checkpoint API
    from torch.distributed.checkpoint import (
        save as dcp_save,
        load as dcp_load,
        FileSystemReader,
        FileSystemWriter,
    )
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
        StateDictOptions,
    )
    HAS_DCP = True
except ImportError:
    # Fallback for older PyTorch versions
    HAS_DCP = False
from safetensors.torch import load_file, save_file


class ShardedCheckpointManager:
    """
    Manages sharded checkpoint saving/loading for FSDP models.
    Supports:
    - Memory-efficient sharded saving (prevents OOM)
    - Loading from both HF format and sharded checkpoints
    - Conversion between formats
    """
    
    @staticmethod
    def detect_checkpoint_format(checkpoint_path: str) -> str:
        """
        Detect checkpoint format: 'sharded', 'huggingface', or 'unknown'
        """
        if not os.path.exists(checkpoint_path):
            return "unknown"
        
        # Check for sharded checkpoint metadata
        if os.path.exists(os.path.join(checkpoint_path, "metadata.json")):
            try:
                with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                    if metadata.get("checkpoint_type") == "sharded":
                        return "sharded"
            except:
                pass
        
        # Check for HuggingFace format files
        hf_files = ["model.safetensors", "pytorch_model.bin", "model.pt", "config.json"]
        for fname in hf_files:
            if os.path.exists(os.path.join(checkpoint_path, fname)):
                return "huggingface"
        
        # Check if it's a direct safetensors file
        if checkpoint_path.endswith(".safetensors") and os.path.isfile(checkpoint_path):
            return "huggingface"
        
        return "unknown"
    
    @staticmethod
    def save_sharded_checkpoint(
        ckpt_dir: str,
        train_steps: int,
        model: FSDP,
        ema_model: Optional[FSDP],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        data_status: Optional[Dict],
        logger: Any,
        save_full_model: bool = False,
    ):
        """
        Save checkpoint in sharded format to prevent OOM.
        Each rank saves its own shard.
        """
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        logger.info(f"Rank {rank}: Saving sharded checkpoint to {save_path}")
        
        # Save metadata on rank 0
        if rank == 0:
            metadata = {
                "train_steps": train_steps,
                "world_size": world_size,
                "checkpoint_type": "sharded",
                "has_ema": ema_model is not None,
                "has_scheduler": scheduler is not None,
                "has_data_status": data_status is not None,
            }
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        
        if HAS_DCP:
            # Use new PyTorch distributed checkpoint API
            model_state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(
                    full_state_dict=False,  # Keep sharded
                    cpu_offload=True,  # Offload to CPU to save memory
                )
            )
            
            dcp_save(
                state_dict=model_state_dict,
                storage_writer=FileSystemWriter(os.path.join(save_path, "model_sharded")),
            )
            
            # Save EMA model shards if present
            if ema_model is not None:
                logger.info(f"Rank {rank}: Saving EMA shards")
                ema_state_dict = get_model_state_dict(
                    ema_model,
                    options=StateDictOptions(
                        full_state_dict=False,
                        cpu_offload=True,
                    )
                )
                logger.info(f"Rank {rank}: EMA state dict keys: {len(ema_state_dict)}")
                dcp_save(
                    state_dict=ema_state_dict,
                    storage_writer=FileSystemWriter(os.path.join(save_path, "ema_sharded")),
                )
                logger.info(f"Rank {rank}: EMA shards saved successfully")
            else:
                logger.info(f"Rank {rank}: EMA model is None, skipping EMA save")
            
            # Save optimizer shards
            optimizer_state_dict = get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                )
            )
        else:
            # Fallback: Use FSDP's native sharded state dict
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                model_state_dict = model.state_dict()
                torch.save(
                    model_state_dict,
                    os.path.join(save_path, f"model_rank{rank:05d}.pt")
                )
            
            if ema_model is not None:
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=True),
                ):
                    ema_state_dict = ema_model.state_dict()
                    torch.save(
                        ema_state_dict,
                        os.path.join(save_path, f"ema_rank{rank:05d}.pt")
                    )
            
            # Save optimizer state per rank
            optimizer_state_dict = optimizer.state_dict()
        
        torch.save(
            optimizer_state_dict,
            os.path.join(save_path, f"optimizer_rank{rank:05d}.pt")
        )
        
        # Optionally save full model on rank 0 (for inference)
        if save_full_model and rank == 0:
            logger.info("Rank 0: Additionally saving full model for inference")
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                full_state_dict = model.state_dict()
                save_file(full_state_dict, os.path.join(save_path, "model_full.safetensors"))
        
        # Save scheduler and data status on rank 0
        if rank == 0:
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            
            if data_status is not None:
                torch.save(data_status, os.path.join(save_path, "data_status.pt"))
        
        dist.barrier()
        logger.info(f"Rank {rank}: Checkpoint saved successfully")
    
    @staticmethod
    def load_sharded_checkpoint(
        resume_from: str,
        model: FSDP,
        ema_model: Optional[FSDP],
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        logger: Any,
        load_model: bool = True,  # Add flag to control model loading
        load_ema: bool = True,    # Add flag to control ema loading
    ) -> tuple:
        """
        Load sharded checkpoint.
        Returns: (model, ema_model, optimizer, scheduler, train_steps, data_status)
        """
        if not os.path.exists(resume_from):
            logger.warning(f"Checkpoint path {resume_from} does not exist")
            return model, ema_model, optimizer, scheduler, 0, None
        
        logger.info(f"Loading sharded checkpoint from {resume_from}")
        
        # Load metadata
        metadata_path = os.path.join(resume_from, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            # Fallback for old checkpoints
            metadata = {"checkpoint_type": "legacy"}
        
        if metadata.get("checkpoint_type") == "sharded":
            # Load sharded checkpoint
            if load_model:
                logger.info("Loading model weights from sharded checkpoint")
                if HAS_DCP and os.path.exists(os.path.join(resume_from, "model_sharded")):
                    # Use new DCP API
                    model_state_dict = {}
                    dcp_load(
                        state_dict=model_state_dict,
                        storage_reader=FileSystemReader(os.path.join(resume_from, "model_sharded")),
                    )
                    set_model_state_dict(
                        model,
                        model_state_dict,
                        options=StateDictOptions(strict=False),
                    )
                else:
                    # Fallback: Load from rank-specific files
                    rank = dist.get_rank()
                    model_path = os.path.join(resume_from, f"model_rank{rank:05d}.pt")
                    if os.path.exists(model_path):
                        with FSDP.state_dict_type(
                            model,
                            StateDictType.SHARDED_STATE_DICT,
                            ShardedStateDictConfig(offload_to_cpu=True),
                        ):
                            model_state_dict = torch.load(model_path, map_location="cpu")
                            model.load_state_dict(model_state_dict, strict=False)
            else:
                logger.info("Skipping model weight loading (load_model=False)")
                
            # Load EMA if present
            if load_ema and ema_model is not None and metadata.get("has_ema", False):
                logger.info("Loading EMA weights from sharded checkpoint")
                if HAS_DCP and os.path.exists(os.path.join(resume_from, "ema_sharded")):
                    ema_state_dict = {}
                    dcp_load(
                        state_dict=ema_state_dict,
                        storage_reader=FileSystemReader(os.path.join(resume_from, "ema_sharded")),
                    )
                    set_model_state_dict(
                        ema_model,
                        ema_state_dict,
                        options=StateDictOptions(strict=False),
                    )
                else:
                    # Fallback: Load from rank-specific files
                    rank = dist.get_rank()
                    ema_path = os.path.join(resume_from, f"ema_rank{rank:05d}.pt")
                    if os.path.exists(ema_path):
                        with FSDP.state_dict_type(
                            ema_model,
                            StateDictType.SHARDED_STATE_DICT,
                            ShardedStateDictConfig(offload_to_cpu=True),
                        ):
                            ema_state_dict = torch.load(ema_path, map_location="cpu")
                            ema_model.load_state_dict(ema_state_dict, strict=False)
            
            # Load optimizer
            if optimizer is not None:
                rank = dist.get_rank()
                optimizer_path = os.path.join(resume_from, f"optimizer_rank{rank:05d}.pt")
                if os.path.exists(optimizer_path):
                    optimizer_state_dict = torch.load(optimizer_path, map_location="cpu")
                    if HAS_DCP:
                        set_optimizer_state_dict(
                            model,
                            optimizer,
                            optimizer_state_dict,
                            options=StateDictOptions(strict=False),
                        )
                    else:
                        optimizer.load_state_dict(optimizer_state_dict)
        else:
            # Load legacy full checkpoint (backward compatibility)
            return ShardedCheckpointManager._load_legacy_checkpoint(
                resume_from, model, ema_model, optimizer, scheduler, logger
            )
        
        # Load scheduler and data status
        train_steps = 0
        data_status = None
        
        if scheduler is not None:
            scheduler_path = os.path.join(resume_from, "scheduler.pt")
            if os.path.exists(scheduler_path):
                scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        
        data_status_path = os.path.join(resume_from, "data_status.pt")
        if os.path.exists(data_status_path):
            all_data_status = torch.load(data_status_path, map_location="cpu")
            rank = dist.get_rank()
            if rank < len(all_data_status):
                data_status = all_data_status[rank]
        
        # Extract train steps from directory name
        try:
            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
        except:
            train_steps = metadata.get("train_steps", 0) + 1
        
        return model, ema_model, optimizer, scheduler, train_steps, data_status
    
    @staticmethod
    def _load_legacy_checkpoint(
        resume_from: str,
        model: FSDP,
        ema_model: Optional[FSDP],
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        logger: Any,
    ) -> tuple:
        """
        Load legacy full checkpoint format (backward compatibility).
        """
        logger.info(f"Loading legacy checkpoint from {resume_from}")
        
        # Load model
        model_path = os.path.join(resume_from, "model.safetensors")
        if os.path.exists(model_path):
            model_state_dict = load_file(model_path, device="cpu")
            # Remove position embeddings for flexibility
            model_state_dict.pop('latent_pos_embed.pos_embed', None)
            model_state_dict.pop('vit_pos_embed.pos_embed', None)
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"Model loading: {msg}")
        
        # Load EMA
        if ema_model is not None:
            ema_path = os.path.join(resume_from, "ema.safetensors")
            if os.path.exists(ema_path):
                ema_state_dict = load_file(ema_path, device="cpu")
                ema_state_dict.pop('latent_pos_embed.pos_embed', None)
                ema_state_dict.pop('vit_pos_embed.pos_embed', None)
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(f"EMA loading: {msg}")
        
        # Load optimizer and scheduler using original logic
        train_steps = 0
        data_status = None
        
        if optimizer is not None:
            # This would need the fsdp_config to determine shard indices
            # For now, we'll skip optimizer loading for legacy format
            logger.warning("Optimizer loading not supported for legacy checkpoints in new manager")
        
        if scheduler is not None:
            scheduler_path = os.path.join(resume_from, "scheduler.pt")
            if os.path.exists(scheduler_path):
                scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        
        return model, ema_model, optimizer, scheduler, train_steps, data_status
    
    @staticmethod
    def load_hf_checkpoint(
        checkpoint_path: str,
        model: FSDP,
        ema_model: Optional[FSDP],
        logger: Any,
        load_ema_from_main: bool = False,
    ) -> tuple:
        """
        Load HuggingFace format checkpoint (for initial training from Bagel checkpoint).
        """
        if not os.path.exists(checkpoint_path):
            logger.info(f"HF checkpoint {checkpoint_path} not found, training from scratch")
            return model, ema_model
        
        logger.info(f"Loading HF checkpoint from {checkpoint_path}")
        
        # Check if it's a file or directory
        if os.path.isfile(checkpoint_path):
            model_state_dict = load_file(checkpoint_path, device="cpu")
        else:
            # Try to find model file in directory
            possible_files = ["model.safetensors", "pytorch_model.bin", "model.pt"]
            model_file = None
            for fname in possible_files:
                fpath = os.path.join(checkpoint_path, fname)
                if os.path.exists(fpath):
                    model_file = fpath
                    break
            
            if model_file is None:
                logger.error(f"No model file found in {checkpoint_path}")
                return model, ema_model
            
            if model_file.endswith(".safetensors"):
                model_state_dict = load_file(model_file, device="cpu")
            else:
                model_state_dict = torch.load(model_file, map_location="cpu")
        
        # Remove position embeddings for flexibility
        model_state_dict.pop('latent_pos_embed.pos_embed', None)
        model_state_dict.pop('vit_pos_embed.pos_embed', None)
        
        # Load into model
        msg = model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Model loading result: {msg}")
        
        # Initialize EMA from model if requested
        if ema_model is not None and load_ema_from_main:
            logger.info("Initializing EMA from main model")
            msg = ema_model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"EMA loading result: {msg}")
        
        return model, ema_model