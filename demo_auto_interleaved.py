#!/usr/bin/env python3
"""Clean auto-interleaved generation demo."""

import os
import argparse
import torch
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

# Model imports
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, SiglipVisionConfig
from modeling.bagel.qwen2_navit import Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.siglip import SiglipVisionModel
from modeling.autoencoder import load_ae

# Data and inference imports
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from auto_interleaved_inference import AutoInterleavedInferencer


def load_model(model_path):
    """Load and setup the Bagel model components."""
    print("Loading model configurations...")
    
    # Load configurations
    llm_config = Qwen2Config.from_json_file(f"{model_path}/llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    vit_config = SiglipVisionConfig.from_json_file(f"{model_path}/vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # Load VAE
    vae_model, vae_config = load_ae(f"{model_path}/ae.safetensors")
    
    # Create Bagel configuration
    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config,
        vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, use_moe=True
    )
    
    # Initialize model with empty weights
    print("Initializing model...")
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        
        # Skip optional method if not available
        try:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        except AttributeError:
            pass
    
    # Load checkpoint with error handling - use single device to avoid device mismatch
    print("Loading checkpoint...")
    try:
        model = load_checkpoint_and_dispatch(
            model, checkpoint=f"{model_path}/ema.safetensors",
            device_map={"": "cuda:0"}, dtype=torch.float16, no_split_module_classes=["Qwen2MoTDecoderLayer"]
        )
    except ValueError as e:
        if "pos_embed" in str(e):
            print("⚠️  Skipping pos_embed tensor shape mismatch")
        else:
            raise e
    
    model.eval()
    
    # Fix rotary embedding device mismatch (single device loading)
    for name, module in model.named_modules():
        if 'rotary_emb' in name and hasattr(module, 'inv_freq'):
            if module.inv_freq.device.type == 'cpu':
                module.inv_freq = module.inv_freq.to('cuda:0')
            break
    
    return model, vae_model


def setup_tokenizer_and_transforms(model_path):
    """Setup tokenizer and image transforms."""
    print("Setting up tokenizer...")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    print("Setting up transforms...")
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    
    return tokenizer, new_token_ids, vae_transform, vit_transform


def run_generation(inferencer, prompt, output_dir):
    """Run auto-interleaved generation and process outputs."""
    print(f"Generating for: '{prompt}'")
    print("Monitoring for <|vision_start|> tokens...")
    
    outputs = inferencer.auto_interleaved_generation(
        prompt, max_interleaved_blocks=3, max_text_length=100,
        do_sample=True, text_temperature=0.8
    )
    
    # Process outputs
    os.makedirs(output_dir, exist_ok=True)
    
    text_count = image_count = 0
    for output in outputs:
        if isinstance(output, str):
            text_count += 1
            print(f"📝 Text {text_count}: {output[:100]}...")
        else:
            image_count += 1
            filename = f"{output_dir}/image_{image_count}.png"
            output.save(filename)
            print(f"🖼️  Image {image_count}: {filename}")
    
    print(f"\nResult: {text_count} text blocks, {image_count} images")
    if image_count == 0:
        print("💡 No images - model needs training on vision tokens")


def main():
    parser = argparse.ArgumentParser(description="Auto-interleaved generation demo")
    parser.add_argument("--model-path", default="/tmp/bagel_models/BAGEL-7B-MoT", 
                        help="Path to model directory")
    parser.add_argument("--prompt", default="Tell a story about a robot",
                        help="Input prompt for generation") 
    parser.add_argument("--output-dir", default="./outputs",
                        help="Directory to save generated images")
    args = parser.parse_args()
    
    print("🤖 Auto-Interleaved Generation Demo")
    print("=" * 40)
    
    # Load model components
    model, vae_model = load_model(args.model_path)
    
    # Setup tokenizer and transforms
    tokenizer, new_token_ids, vae_transform, vit_transform = setup_tokenizer_and_transforms(args.model_path)
    
    # Create inferencer
    print("Creating inferencer...")
    inferencer = AutoInterleavedInferencer(
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
    )
    
    # Run generation
    run_generation(inferencer, args.prompt, args.output_dir)

if __name__ == "__main__":
    main()