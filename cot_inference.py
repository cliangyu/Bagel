#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Chain of Thought Inference

This script uses the BAGEL model for chain-of-thought image generation with
iterative refinement capabilities.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch

from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from PIL import Image


def log_with_timestamp(message, log_file=None):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")


def setup_bagel_model():
    """Setup BAGEL model exactly like the working notebook"""
    print("🚀 Setting up BAGEL model...")

    # Model path - using the local downloaded model
    model_path = "./BAGEL-7B-MoT"

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None

    print(f"✅ Model path exists: {model_path}")

    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(model_path, "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(
        local_path=os.path.join(model_path, "ae.safetensors")
    )
    # Fix: Convert VAE to bfloat16 to match main model
    vae_model = vae_model.to(torch.bfloat16)
    print("✅ VAE loaded and converted to bfloat16")

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    print("✅ Model structure initialized")

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    print("✅ Tokenizer and transforms ready")

    # Model loading - exactly like the working notebook
    max_mem_per_gpu = "80GiB"

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
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

    print(f"✅ Device map: {device_map}")

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload",
    )

    model = model.eval()
    print("✅ Model loaded and set to eval mode")

    # Move VAE to same device as main model
    device = next(model.parameters()).device
    vae_model = vae_model.to(device)
    print(f"✅ VAE moved to {device}")

    # Initialize inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("✅ Inferencer created and seeds set")
    return inferencer


def real_chain_of_thought_generation(inferencer, prompt, output_dir, max_iterations=3, accumulative=False):
    """Run real chain-of-thought generation using exact working parameters
    
    Args:
        inferencer: The BAGEL model inferencer
        prompt: Initial text prompt
        output_dir: Directory to save outputs
        max_iterations: Maximum refinement iterations
        accumulative: If True, accumulate context across iterations
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "real_cot_log.txt")

    # Save original prompt for markdown visualization
    with open(os.path.join(output_dir, "original_prompt.txt"), "w") as f:
        f.write(prompt)

    # Initialize log
    with open(log_file, "w") as f:
        f.write(f"Real BAGEL Chain of Thought Generation\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write("=" * 80 + "\n\n")

    log_with_timestamp("🚀 Starting REAL Chain of Thought Generation", log_file)
    log_with_timestamp(f"📝 Prompt: {prompt}", log_file)
    log_with_timestamp(f"🔄 Max iterations: {max_iterations}", log_file)
    log_with_timestamp(f"📚 Accumulative mode: {accumulative}", log_file)

    results = []
    
    # Initialize context accumulation
    reflection_history = []
    edit_history = []

    # PHASE 1: Initial Generation (using exact working parameters)
    log_with_timestamp("\n" + "=" * 60, log_file)
    log_with_timestamp("PHASE 1: INITIAL GENERATION", log_file)
    log_with_timestamp("=" * 60, log_file)

    # Use exact parameters from working examples
    initial_params = {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
    }

    log_with_timestamp("🎨 Generating initial image...", log_file)

    try:
        start_time = time.time()
        initial_result = inferencer(text=prompt, **initial_params)
        gen_time = time.time() - start_time

        current_image = initial_result["image"]

        # Save initial result
        current_image.save(os.path.join(output_dir, "step_0_initial.png"))

        log_with_timestamp(
            f"✅ Initial generation complete ({gen_time:.1f}s)", log_file
        )

        results.append(
            {
                "step": 0,
                "type": "initial_generation",
                "image": current_image,
                "time": gen_time,
                "prompt": prompt,
            }
        )

    except Exception as e:
        log_with_timestamp(f"❌ Error in initial generation: {e}", log_file)
        return None

    # PHASE 2: Reflection and Refinement Loop
    best_image = current_image.copy()

    for iteration in range(1, max_iterations + 1):
        log_with_timestamp(f"\n" + "*" * 60, log_file)
        log_with_timestamp(
            f"ITERATION {iteration}: REFLECTION AND REFINEMENT", log_file
        )
        log_with_timestamp("*" * 60, log_file)

        # Reflection Phase (using understanding parameters)
        log_with_timestamp("🔍 REFLECTION PHASE", log_file)

        reflection_params = {
            "understanding_output": True,  # Key parameter for reflection
            "think": True,                 # Enable thinking mode for better analysis
            "max_think_token_n": 2000,     # Allow longer reasoning
            "do_sample": False,            # Deterministic output for consistent reflection
            "text_temperature": 0.3,       # Lower temperature for more focused analysis
        }

        # Build reflection prompt with optional history
        if accumulative and reflection_history:
            history_text = "\n\nPrevious reflection history:\n"
            for i, (prev_reflection, prev_edit) in enumerate(zip(reflection_history, edit_history)):
                history_text += f"\n--- Iteration {i+1} ---\n"
                history_text += f"Previous assessment:\n{prev_reflection}\n"
                history_text += f"Edit applied: {prev_edit}\n"
            
            reflection_prompt = f"""Look at this image and the original prompt: "{prompt}"
{history_text}

Based on the history above, examine the current image carefully and assess the progress made.

Reflect on whether the image NOW fully satisfies the prompt requirements:
1. Does the image contain ALL the elements mentioned in the prompt?
2. Are the details accurate and properly executed?
3. What improvements from previous iterations were successful?
4. What issues remain unresolved?

Provide your assessment in this format:
OBSERVATION: [What I see in the current image]
REQUIREMENTS_MET: [Yes/No - whether all prompt requirements are satisfied]
NEEDED_IMPROVEMENTS: [Specific list of what still needs to be changed or added]"""
        else:
            reflection_prompt = f"""Look at this image and the original prompt: "{prompt}"

CRITICAL ANALYSIS INSTRUCTIONS:
- Look VERY carefully at the actual image content
- Do NOT assume elements are present just because they were requested
- Count objects, colors, arrangements precisely  
- Identify specific missing or incorrect details

First, examine the image carefully and verbalize EXACTLY what you see:

Then, for EACH element mentioned in the prompt, check if it's actually present and correct:
1. Does the image contain ALL the elements mentioned in the prompt?
2. Are the details accurate and properly executed?
3. Are quantities, colors, positions, and arrangements exactly as specified?

BE SPECIFIC about what is missing, incorrect, or needs improvement. Do not claim something is present if you cannot clearly see it.

Provide your assessment in this format:
OBSERVATION: [Detailed description of what I actually see in the image]
REQUIREMENTS_MET: [Yes/No - whether all prompt requirements are satisfied]
NEEDED_IMPROVEMENTS: [Specific list of what needs to be changed or added]"""

        try:
            start_time = time.time()
            reflection_result = inferencer(
                image=current_image, text=reflection_prompt, **reflection_params
            )
            reflect_time = time.time() - start_time

            reflection_text = reflection_result.get("text", "")

            # Save reflection
            with open(
                os.path.join(output_dir, f"step_{iteration}_reflection.txt"), "w"
            ) as f:
                f.write(reflection_text)

            log_with_timestamp(
                f"✅ Reflection complete ({reflect_time:.1f}s)", log_file
            )
            log_with_timestamp(f"🤔 Analysis: {reflection_text}", log_file)

            # Parse structured reflection response
            reflection_lower = reflection_text.lower()
            
            # Check if requirements are met
            requirements_met = False
            if "requirements_met:" in reflection_lower:
                req_line = [line for line in reflection_text.split('\n') if 'requirements_met:' in line.lower()]
                if req_line:
                    req_text = req_line[0].split(':')[1].strip().lower()
                    requirements_met = req_text.startswith('yes')
            
            # Extract needed improvements for edit instruction
            needed_improvements = ""
            if "needed_improvements:" in reflection_lower:
                imp_lines = []
                found_section = False
                for line in reflection_text.split('\n'):
                    if 'needed_improvements:' in line.lower():
                        found_section = True
                        imp_lines.append(line.split(':', 1)[1].strip())
                    elif found_section and line.strip():
                        imp_lines.append(line.strip())
                    elif found_section and not line.strip():
                        break
                needed_improvements = ' '.join(imp_lines).strip()
            
            satisfactory = requirements_met

            log_with_timestamp(f"✅ Requirements met: {'Yes' if requirements_met else 'No'}", log_file)
            log_with_timestamp(f"✅ Satisfactory: {'Yes' if satisfactory else 'No'}", log_file)
            if needed_improvements:
                log_with_timestamp(f"🔧 Needed improvements: {needed_improvements}", log_file)

            # Update best image
            if satisfactory:
                best_image = current_image.copy()
                log_with_timestamp(f"🏆 Satisfactory image saved as best", log_file)

            # Check if satisfactory
            if satisfactory:
                log_with_timestamp(f"\n{'!'*60}", log_file)
                log_with_timestamp("🎉 SATISFACTORY RESULT ACHIEVED!", log_file)
                log_with_timestamp(f"All requirements met!", log_file)
                log_with_timestamp("!" * 60, log_file)
                results.append(
                    {
                        "step": iteration,
                        "type": "reflection",
                        "requirements_met": requirements_met,
                        "satisfactory": True,
                        "text": reflection_text,
                        "needed_improvements": needed_improvements,
                        "time": reflect_time,
                    }
                )
                break

            # Store reflection in history for accumulative mode
            if accumulative:
                reflection_history.append(reflection_text)
            
            # Edit Phase (using exact editing parameters)
            log_with_timestamp("\n✏️ EDIT PHASE", log_file)

            edit_params = {
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 2.0,
                "cfg_interval": [0.0, 1.0],
                "timestep_shift": 3.0,
                "num_timesteps": 50,
                "cfg_renorm_min": 0.0,
                "cfg_renorm_type": "text_channel",
            }

            # Generate edit instruction based on model's own reflection
            if accumulative and edit_history:
                history_context = f"\n\nPrevious edits attempted: {'; '.join(edit_history[-3:])}"  # Last 3 edits
                if needed_improvements:
                    edit_instruction = f"Based on your analysis and previous attempts{history_context}, edit this image to address these specific issues: {needed_improvements}. Ensure the final result perfectly matches the original prompt: '{prompt}'"
                else:
                    edit_instruction = f"Improve this image to better match: {prompt}. Make it more accurate and higher quality.{history_context}"
            else:
                if needed_improvements:
                    edit_instruction = f"Based on your analysis, edit this image to address these specific issues: {needed_improvements}. Ensure the final result perfectly matches the original prompt: '{prompt}'"
                else:
                    edit_instruction = f"Improve this image to better match: {prompt}. Make it more accurate and higher quality."

            log_with_timestamp(f"📝 Edit instruction: {edit_instruction}", log_file)

            try:
                start_time = time.time()
                edit_result = inferencer(
                    image=current_image, text=edit_instruction, **edit_params
                )
                edit_time = time.time() - start_time

                edited_image = edit_result["image"]

                # Save edit result
                edited_image.save(
                    os.path.join(output_dir, f"step_{iteration}_edited.png")
                )

                current_image = edited_image

                log_with_timestamp(f"✅ Edit complete ({edit_time:.1f}s)", log_file)
                
                # Store edit instruction in history for accumulative mode
                if accumulative:
                    edit_history.append(edit_instruction)

                results.append(
                    {
                        "step": iteration,
                        "type": "reflection",
                        "requirements_met": requirements_met,
                        "satisfactory": satisfactory,
                        "text": reflection_text,
                        "needed_improvements": needed_improvements,
                        "time": reflect_time,
                    }
                )
                
                results.append(
                    {
                        "step": iteration,
                        "type": "edit",
                        "image": edited_image,
                        "time": edit_time,
                        "edit_instruction": edit_instruction,
                        "based_on_reflection": True,
                        "satisfactory": False,
                    }
                )

            except Exception as e:
                log_with_timestamp(f"❌ Error in edit: {e}", log_file)
                break

        except Exception as e:
            log_with_timestamp(f"❌ Error in iteration {iteration}: {e}", log_file)
            break

    # Final Summary
    final_iteration = len([r for r in results if r["type"] in ["edit", "reflection"]])

    log_with_timestamp(f"\n" + "#" * 80, log_file)
    log_with_timestamp("🎯 REAL CHAIN OF THOUGHT GENERATION COMPLETE", log_file)
    log_with_timestamp(f"Total iterations: {final_iteration}", log_file)
    log_with_timestamp("#" * 80, log_file)

    # Save final results
    best_image.save(os.path.join(output_dir, "final_best.png"))
    current_image.save(os.path.join(output_dir, "final_current.png"))

    summary = {
        "prompt": prompt,
        "model": "BAGEL-7B-MoT",
        "generation_type": "real_chain_of_thought",
        "total_iterations": final_iteration,
        "generation_timestamp": datetime.now().isoformat(),
        "results": [
            {
                k: v for k, v in r.items() if k != "image"
            }  # Don't save image objects in JSON
            for r in results
        ],
    }

    with open(os.path.join(output_dir, "cot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log_with_timestamp(f"\n📁 All outputs saved to: {output_dir}", log_file)
    log_with_timestamp(f"🖼️ Best image: final_best.png", log_file)
    log_with_timestamp(f"📄 Summary: cot_summary.json", log_file)
    
    # Generate markdown visualization
    log_with_timestamp("📝 Generating markdown visualization...", log_file)
    compile_markdown_visualization(output_dir)
    log_with_timestamp(f"📄 Markdown visualization: visualization.md", log_file)

    return summary


def compile_markdown_visualization(output_dir):
    """
    Compile a markdown file with images and prompts from chain-of-thought experiments.
    
    Args:
        output_dir (str): Directory containing the experiment outputs
    """
    import glob
    from pathlib import Path
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory {output_dir} does not exist")
        return
    
    md_content = []
    md_content.append(f"# Chain of Thought Visualization")
    md_content.append(f"**Experiment Directory:** `{output_dir}`")
    md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("")
    
    # Read original prompt if available
    prompt_file = os.path.join(output_dir, "original_prompt.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            original_prompt = f.read().strip()
        md_content.append(f"## Original Prompt")
        md_content.append(f"```")
        md_content.append(original_prompt)
        md_content.append(f"```")
        md_content.append("")
    
    # Find all step files and organize them
    step_files = {}
    for file_path in glob.glob(os.path.join(output_dir, "step_*")):
        filename = os.path.basename(file_path)
        
        # Extract step number
        step_match = re.match(r'step_(\d+)', filename)
        if step_match:
            step_num = int(step_match.group(1))
            if step_num not in step_files:
                step_files[step_num] = {}
            
            if filename.endswith('.png'):
                step_files[step_num]['image'] = filename
            elif filename.endswith('_reflection.txt'):
                step_files[step_num]['reflection'] = filename
            elif filename.endswith('_edit.txt'):
                step_files[step_num]['edit'] = filename
            elif filename.endswith('_think.txt'):
                step_files[step_num]['think'] = filename
    
    # Generate markdown for each step
    for step_num in sorted(step_files.keys()):
        files = step_files[step_num]
        md_content.append(f"## Step {step_num}")
        
        # Add image if available
        if 'image' in files:
            image_path = files['image']
            md_content.append(f"### Generated Image")
            md_content.append(f"![Step {step_num} Image](./{image_path})")
            md_content.append("")
        
        # Add thinking process if available
        if 'think' in files:
            think_path = os.path.join(output_dir, files['think'])
            if os.path.exists(think_path):
                with open(think_path, 'r') as f:
                    think_content = f.read().strip()
                md_content.append(f"### Thinking Process")
                md_content.append("```")
                md_content.append(think_content)
                md_content.append("```")
                md_content.append("")
        
        # Add reflection if available
        if 'reflection' in files:
            reflection_path = os.path.join(output_dir, files['reflection'])
            if os.path.exists(reflection_path):
                with open(reflection_path, 'r') as f:
                    reflection_content = f.read().strip()
                md_content.append(f"### Reflection")
                md_content.append("```")
                md_content.append(reflection_content)
                md_content.append("```")
                md_content.append("")
        
        # Add edit prompt if available
        if 'edit' in files:
            edit_path = os.path.join(output_dir, files['edit'])
            if os.path.exists(edit_path):
                with open(edit_path, 'r') as f:
                    edit_content = f.read().strip()
                md_content.append(f"### Edit Prompt")
                md_content.append("```")
                md_content.append(edit_content)
                md_content.append("```")
                md_content.append("")
        
        md_content.append("---")
        md_content.append("")
    
    # Add summary section
    md_content.append(f"## Summary")
    md_content.append(f"- **Total Steps:** {len(step_files)}")
    md_content.append(f"- **Images Generated:** {len([f for f in step_files.values() if 'image' in f])}")
    md_content.append(f"- **Reflections:** {len([f for f in step_files.values() if 'reflection' in f])}")
    md_content.append(f"- **Edit Prompts:** {len([f for f in step_files.values() if 'edit' in f])}")
    md_content.append("")
    
    # Add file listing
    all_files = sorted([f for f in os.listdir(output_dir) if f.endswith(('.png', '.txt', '.json'))])
    if all_files:
        md_content.append(f"## All Files in Directory")
        for filename in all_files:
            md_content.append(f"- `{filename}`")
        md_content.append("")
    
    # Write markdown file
    md_filename = os.path.join(output_dir, "visualization.md")
    with open(md_filename, 'w') as f:
        f.write('\n'.join(md_content))
    
    return md_filename


def main():
    """Main function to run chain-of-thought inference"""
    
    parser = argparse.ArgumentParser(description="BAGEL Chain of Thought Inference")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for image generation")
    parser.add_argument("--accumulative", action="store_true",
                        help="Enable accumulative context across iterations")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum number of refinement iterations (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated with timestamp)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()

    print("🚀 BAGEL Chain of Thought Inference")
    print("=" * 80)
    print(f"Mode: {'Accumulative' if args.accumulative else 'Non-accumulative'} context")
    print()

    # Setup the model
    inferencer = setup_bagel_model()
    if inferencer is None:
        print("❌ Failed to setup BAGEL model. Exiting.")
        return

    print("\n✅ BAGEL model setup complete!")
    print("🎯 Ready for chain-of-thought generation")

    # Set random seed
    if args.seed != 42:  # Only reset if different from default
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "accumulative" if args.accumulative else "no_accumulative"
        output_dir = f"./results/cot_outputs/cot_output_{mode_suffix}_{timestamp}"
    else:
        output_dir = args.output_dir

    print(f"\n📝 Prompt: {args.prompt}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔄 Max iterations: {args.max_iterations}")
    print(f"📚 Accumulative: {args.accumulative}")
    print("-" * 80)

    # Run chain-of-thought generation
    try:
        result = real_chain_of_thought_generation(
            inferencer=inferencer,
            prompt=args.prompt,
            output_dir=output_dir,
            max_iterations=args.max_iterations,
            accumulative=args.accumulative,
        )

        if result:
            print(f"\n🎉 Chain of Thought generation completed!")
            print(f"📊 Total iterations: {result['total_iterations']}")
            print(f"📁 All images and logs saved to: {output_dir}")

            # List generated files
            files = [
                f
                for f in os.listdir(output_dir)
                if f.endswith((".png", ".txt", ".json", ".md"))
            ]
            print(f"\n🖼️ Generated files:")
            for filename in sorted(files):
                print(f"   • {filename}")
        else:
            print("❌ Chain of thought generation failed")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
