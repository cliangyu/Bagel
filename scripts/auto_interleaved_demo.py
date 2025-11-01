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

    def generate_auto_sequence(
        self,
        num_images: int = 1,
        think: bool = False,
        max_think_token_n: int = 500,
        do_sample: bool = True,
        text_temperature: float = 0.7,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        enable_taylorseer: bool = False,
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        Automatically generate interleaved text-image sequence.

        Pattern: text → image → text → image → ...
        - After text input: optionally generate thinking, then generate image
        - After image: generate refinement text for next iteration

        Args:
            num_images: Number of images to generate (controls sequence length)
            think: Whether to use chain-of-thought before each image
            max_think_token_n: Max tokens for thinking/refinement text
            do_sample: Whether to sample during text generation
            text_temperature: Temperature for text generation
            cfg_text_scale: Classifier-free guidance scale for text conditioning
            cfg_img_scale: Classifier-free guidance scale for image conditioning
            num_timesteps: Number of denoising steps for image generation
            enable_taylorseer: Whether to use TaylorSeer acceleration
            **kwargs: Additional parameters passed to generation methods

        Returns:
            List of generated outputs [text?, image, text?, image, ...]

        Use Cases:
            1. Pure generation: Start with text prompt
               >>> auto_inferencer.add_to_context("A magical forest")
               >>> outputs = auto_inferencer.generate_auto_sequence(num_images=3, think=True)
               Returns: [thinking_1, image_1, refinement_1, thinking_2, image_2, refinement_2, thinking_3, image_3]

            2. Image editing: Start with image + instruction
               >>> auto_inferencer.add_to_context(original_image)
               >>> auto_inferencer.add_to_context("Make it warmer")
               >>> outputs = auto_inferencer.generate_auto_sequence(num_images=2, think=True)
               Returns: [thinking_1, edited_1, refinement_1, thinking_2, edited_2]
        """
        if not self.context_list:
            raise ValueError("Context is empty. Add initial input first.")

        outputs = []
        images_generated = 0

        print(f"[Auto Sequence] Starting generation of {num_images} image(s)...")
        print(f"[Auto Sequence] Initial context length: {len(self.context_list)}")

        while images_generated < num_images:
            # Determine next action based on last item in context
            last_item = self.context_list[-1]

            if isinstance(last_item, Image.Image):
                # After image → generate refinement text
                should_generate_thinking = True
                should_generate_image = False
                action = "refinement text (after image)"
            elif isinstance(last_item, str):
                # After text → optionally think, then generate image
                should_generate_thinking = think
                should_generate_image = True
                action = "thinking + image" if think else "image"
            else:
                raise ValueError(f"Unexpected type in context: {type(last_item)}")

            print(f"\n[Auto Sequence] Step {len(outputs) + 1}: Generating {action}")

            # Generate thinking/refinement text if needed
            if should_generate_thinking:
                print(f"[Auto Sequence]   → Generating text...")

                thinking_text = self.generate_text(
                    max_think_token_n=max_think_token_n,
                    do_sample=do_sample,
                    text_temperature=text_temperature,
                    **kwargs
                )

                outputs.append(thinking_text)
                preview = thinking_text[:100] + "..." if len(thinking_text) > 100 else thinking_text
                print(f"[Auto Sequence]   ✓ Generated: {preview}")

            # Generate image if this iteration should produce one
            if should_generate_image:
                print(f"[Auto Sequence]   → Generating image {images_generated + 1}/{num_images}...")

                # Detect if we have an input image (editing mode)
                has_input_image = any(
                    isinstance(item, Image.Image) and item not in outputs
                    for item in self.context_list
                )

                # Adjust CFG scale based on mode
                current_cfg_img_scale = cfg_img_scale if has_input_image else 1.0
                mode = "editing" if has_input_image else "generation"

                print(f"[Auto Sequence]   Mode: {mode} (cfg_img_scale={current_cfg_img_scale})")

                image = self.generate_image(
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=current_cfg_img_scale,
                    num_timesteps=num_timesteps,
                    enable_taylorseer=enable_taylorseer,
                    **kwargs
                )

                outputs.append(image)
                images_generated += 1
                print(f"[Auto Sequence]   ✓ Generated image {images_generated}/{num_images}")

            # Check if we should continue
            if images_generated < num_images:
                # After generating an image, we need refinement text for next iteration
                # This text will prompt the next image generation
                if isinstance(self.context_list[-1], Image.Image):
                    print(f"[Auto Sequence]   → Generating refinement text for next iteration...")

                    refinement_text = self.generate_text(
                        max_think_token_n=max_think_token_n,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        **kwargs
                    )

                    outputs.append(refinement_text)
                    preview = refinement_text[:100] + "..." if len(refinement_text) > 100 else refinement_text
                    print(f"[Auto Sequence]   ✓ Refinement: {preview}")

        print(f"\n[Auto Sequence] ✓ Complete! Generated {images_generated} images and {len(outputs) - images_generated} text outputs.")
        print(f"[Auto Sequence] Final context length: {len(self.context_list)}")
        return outputs


def demo_auto_sequence_generation(auto_inferencer: AutoInterleaveInferencer):
    """Demonstrate automatic sequence generation (Use Case 1: Pure Generation)"""

    print("\n" + "="*70)
    print("=== Demo 1: Auto Sequence Generation (Pure Text-to-Image) ===")
    print("="*70 + "\n")

    # Clear any previous context
    auto_inferencer.clear_context()

    # Use Case 1: Start with text prompt, auto-generate sequence
    initial_prompt = "A magical forest with glowing mushrooms and ethereal light"
    auto_inferencer.add_to_context(initial_prompt)
    print(f"User Initial Prompt: {initial_prompt}\n")

    # Generate auto sequence with 3 images
    print("Generating auto sequence: text → image → text → image → text → image\n")
    outputs = auto_inferencer.generate_auto_sequence(
        num_images=3,
        think=True,
        max_think_token_n=300,
        do_sample=True,
        text_temperature=0.7,
        cfg_text_scale=4.0,
        num_timesteps=30,  # Reduced for demo
        enable_taylorseer=False
    )

    # Display outputs
    print("\n" + "-"*70)
    print("Generated Outputs:")
    print("-"*70)
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            preview = output[:150] + "..." if len(output) > 150 else output
            print(f"{i+1}. [TEXT] {preview}")
        elif isinstance(output, Image.Image):
            filename = f"demo_auto_gen_{i+1}.png"
            output.save(filename)
            print(f"{i+1}. [IMAGE] Saved as {filename}")

    print(f"\nTotal outputs: {len(outputs)}")
    print(f"Images generated: {sum(1 for x in outputs if isinstance(x, Image.Image))}")
    print(f"Text generated: {sum(1 for x in outputs if isinstance(x, str))}")


def demo_auto_sequence_editing(auto_inferencer: AutoInterleaveInferencer):
    """Demonstrate automatic sequence editing (Use Case 2: Image Editing)"""

    print("\n" + "="*70)
    print("=== Demo 2: Auto Sequence Editing (Image Editing) ===")
    print("="*70 + "\n")

    # Clear context
    auto_inferencer.clear_context()

    # Load an input image (create a simple one for demo)
    # In real usage, user would provide: original_image = Image.open("photo.jpg")
    print("Note: This demo requires an input image.")
    print("Example usage:")
    print("  auto_inferencer.add_to_context(Image.open('photo.jpg'))")
    print("  auto_inferencer.add_to_context('Make the lighting warmer and add sunset colors')")
    print("  outputs = auto_inferencer.generate_auto_sequence(num_images=2, think=True)")
    print("\nThis would generate:")
    print("  [thinking_1, edited_image_1, refinement_1, thinking_2, edited_image_2]")
    print("\nSkipping actual execution in this demo (no input image provided).")


def demo_conversation(auto_inferencer: AutoInterleaveInferencer):
    """Demonstrate an auto-interleaved conversation (Original Manual Demo)"""

    print("\n" + "="*70)
    print("=== Demo 3: Manual Interleaved Conversation (Original) ===")
    print("="*70 + "\n")
    
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
    parser = argparse.ArgumentParser(
        description="Auto Interleaved Generation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run auto sequence generation demo
  python auto_interleaved_demo.py --model_path /path/to/model --demo auto_generation

  # Run manual conversation demo (original)
  python auto_interleaved_demo.py --model_path /path/to/model --demo manual

  # Run all demos
  python auto_interleaved_demo.py --model_path /path/to/model --demo all
        """
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to BAGEL-7B-MoT weights directory")
    parser.add_argument("--demo", type=str, default="auto_generation",
                        choices=["auto_generation", "auto_editing", "manual", "all"],
                        help="Which demo to run (default: auto_generation)")
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

    # Run selected demo(s)
    if args.demo == "auto_generation" or args.demo == "all":
        demo_auto_sequence_generation(auto_inferencer)

    if args.demo == "auto_editing" or args.demo == "all":
        demo_auto_sequence_editing(auto_inferencer)

    if args.demo == "manual" or args.demo == "all":
        demo_conversation(auto_inferencer)


if __name__ == "__main__":
    main()