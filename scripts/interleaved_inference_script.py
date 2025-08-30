#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Interleaved Inference Script for BAGEL-7B-MoT
Converted from inference.ipynb notebook

Usage:
    python interleaved_inference_script.py --mode generation --prompt "your prompt here"
    python interleaved_inference_script.py --mode editing --image_path "path/to/image.jpg" --prompt "edit description"
    python interleaved_inference_script.py --mode understanding --image_path "path/to/image.jpg" --prompt "explain this image"
"""

import os
import sys
import argparse
import random
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from inferencer import InterleaveInferencer


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_gpu_devices(gpu_ids="2,3,4,5"):
    """Setup GPU device visibility"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"Using GPUs: {gpu_ids}")
    return torch.cuda.device_count()


def load_model(model_path, max_mem_per_gpu="40GiB"):
    """Load and initialize the BAGEL model with multi-GPU support"""
    
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Device mapping for multi-GPU
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print("Device mapping:", device_map)

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Load model
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )

    model = model.eval()
    print('Model loaded successfully')

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def get_inference_hyperparams(mode):
    """Get inference hyperparameters for different modes"""
    
    if mode == "generation":
        return {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
            "enable_taylorseer": False,
        }
    elif mode == "generation_with_think":
        return {
            "max_think_token_n": 1000,
            "do_sample": False,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
            "enable_taylorseer": False,
        }
    elif mode == "editing":
        return {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 2.0,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "text_channel",
            "enable_taylorseer": False,
        }
    elif mode == "editing_with_think":
        return {
            "max_think_token_n": 1000,
            "do_sample": False,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 2.0,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "text_channel",
            "enable_taylorseer": False,
        }
    elif mode == "understanding":
        return {
            "max_think_token_n": 1000,
            "do_sample": False,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="BAGEL Interleaved Inference")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to BAGEL-7B-MoT weights directory")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generation", "generation_with_think", "editing", "editing_with_think", "understanding"],
                        help="Inference mode")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image_path", type=str, default=None, 
                        help="Path to input image (required for editing and understanding modes)")
    parser.add_argument("--output_path", type=str, default="output.png",
                        help="Output image path (for generation and editing modes)")
    parser.add_argument("--gpu_ids", type=str, default="2,3,4,5",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB",
                        help="Maximum memory per GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ["editing", "editing_with_think", "understanding"] and args.image_path is None:
        raise ValueError(f"Mode {args.mode} requires --image_path")

    # Setup
    set_seed(args.seed)
    num_gpus = setup_gpu_devices(args.gpu_ids)
    print(f"Available GPUs: {num_gpus}")

    # Load model
    print("Loading model...")
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path, args.max_mem_per_gpu
    )

    # Initialize inferencer
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )

    # Get inference hyperparameters
    inference_hyper = get_inference_hyperparams(args.mode)

    # Load image if required
    image = None
    if args.image_path:
        image = Image.open(args.image_path)
        print(f"Loaded image: {args.image_path}")

    # Run inference
    print(f"\nRunning {args.mode} with prompt: {args.prompt}")
    print("-" * 50)

    if args.mode == "understanding":
        output_dict = inferencer(image=image, text=args.prompt, understanding_output=True, **inference_hyper)
        print("Model response:")
        print(output_dict['text'])
        
    elif args.mode in ["generation_with_think", "editing_with_think"]:
        output_dict = inferencer(image=image, text=args.prompt, think=True, **inference_hyper)
        print("Model thinking:")
        print(output_dict['text'])
        print("\nGenerated image saved to:", args.output_path)
        output_dict['image'].save(args.output_path)
        
    else:  # generation, editing
        output_dict = inferencer(image=image, text=args.prompt, **inference_hyper)
        print("Generated image saved to:", args.output_path)
        output_dict['image'].save(args.output_path)

    print("Inference completed!")


if __name__ == "__main__":
    main()