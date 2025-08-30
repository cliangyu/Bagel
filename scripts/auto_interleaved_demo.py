#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Auto Interleaved Text-Image Generation Demo
Demonstrates automatic interleaved generation by appending results to context
"""

import os
import sys
import argparse
from typing import List, Union
from PIL import Image

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferencer import InterleaveInferencer
from scripts.interleaved_inference_script import load_model, setup_gpu_devices, set_seed


class AutoInterleaveInferencer:
    """Wrapper for automatic interleaved text-image generation"""
    
    def __init__(self, inferencer: InterleaveInferencer):
        self.inferencer = inferencer
        self.context_list = []  # Maintains the full conversation context
        
    def add_to_context(self, item: Union[str, Image.Image]):
        """Add text or image to the ongoing context"""
        self.context_list.append(item)
        
    def generate_next(self, **kwargs) -> Union[str, Image.Image]:
        """Generate next item (text or image) based on current context"""
        if not self.context_list:
            raise ValueError("Context is empty. Add initial input first.")
            
        # Use the accumulated context for generation
        output_list = self.inferencer.interleave_inference(self.context_list, **kwargs)
        
        # Get the last generated item
        if output_list:
            generated_item = output_list[-1]
            # Automatically append to context for next generation
            self.context_list.append(generated_item)
            return generated_item
        else:
            return None
            
    def generate_text(self, **kwargs) -> str:
        """Generate text continuation"""
        kwargs['understanding_output'] = True
        return self.generate_next(**kwargs)
        
    def generate_image(self, **kwargs) -> Image.Image:
        """Generate image based on current context"""
        kwargs['understanding_output'] = False
        return self.generate_next(**kwargs)
        
    def clear_context(self):
        """Clear the conversation context"""
        self.context_list = []
        
    def get_context(self) -> List[Union[str, Image.Image]]:
        """Get current context"""
        return self.context_list.copy()


def demo_conversation(auto_inferencer: AutoInterleaveInferencer):
    """Demonstrate an auto-interleaved conversation"""
    
    print("=== Auto Interleaved Generation Demo ===\n")
    
    # Step 1: Start with initial text
    print("1. Starting conversation with initial prompt...")
    initial_prompt = "Let's create a story about a magical forest. First, describe the setting."
    auto_inferencer.add_to_context(initial_prompt)
    print(f"User: {initial_prompt}")
    
    # Generate text response
    text_response = auto_inferencer.generate_text(
        max_think_token_n=500,
        do_sample=True,
        text_temperature=0.7
    )
    print(f"AI: {text_response[:200]}..." if len(text_response) > 200 else f"AI: {text_response}")
    print()
    
    # Step 2: Generate an image based on the text description
    print("2. Generating image based on the description...")
    image_prompt = "Now create a visual representation of this magical forest scene"
    auto_inferencer.add_to_context(image_prompt)
    print(f"User: {image_prompt}")
    
    generated_image = auto_inferencer.generate_image(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        num_timesteps=30,  # Reduced for demo
        enable_taylorseer=False
    )
    generated_image.save("demo_forest.png")
    print("AI: [Generated image saved as demo_forest.png]")
    print()
    
    # Step 3: Continue the story based on the image
    print("3. Continuing story based on the generated image...")
    continue_prompt = "Based on this image, what happens next in our story?"
    auto_inferencer.add_to_context(continue_prompt)
    print(f"User: {continue_prompt}")
    
    continuation = auto_inferencer.generate_text(
        max_think_token_n=500,
        do_sample=True,
        text_temperature=0.7
    )
    print(f"AI: {continuation[:200]}..." if len(continuation) > 200 else f"AI: {continuation}")
    print()
    
    # Step 4: Generate another image for the next scene
    print("4. Generating image for the next scene...")
    next_scene_prompt = "Create an image showing this new development in the story"
    auto_inferencer.add_to_context(next_scene_prompt)
    print(f"User: {next_scene_prompt}")
    
    next_image = auto_inferencer.generate_image(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        num_timesteps=30,
        enable_taylorseer=False
    )
    next_image.save("demo_next_scene.png")
    print("AI: [Generated image saved as demo_next_scene.png]")
    print()
    
    print("=== Context Summary ===")
    context = auto_inferencer.get_context()
    for i, item in enumerate(context):
        if isinstance(item, str):
            print(f"{i+1}. Text: {item[:100]}{'...' if len(item) > 100 else ''}")
        else:
            print(f"{i+1}. Image: [PIL Image object]")
    
    print(f"\nTotal context items: {len(context)}")
    print("Context is automatically maintained for coherent generation!")
    


def main():
    parser = argparse.ArgumentParser(description="Auto Interleaved Generation Demo")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to BAGEL-7B-MoT weights directory")
    parser.add_argument("--gpu_ids", type=str, default="2,3,4,5",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_gpu_devices(args.gpu_ids)
    
    # Load model
    print("Loading BAGEL model...")
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(args.model_path)
    
    # Initialize inferencer
    base_inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )
    
    # Create auto inferencer
    auto_inferencer = AutoInterleaveInferencer(base_inferencer)
    
    # Run demo
    demo_conversation(auto_inferencer)


if __name__ == "__main__":
    main()