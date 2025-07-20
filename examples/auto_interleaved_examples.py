#!/usr/bin/env python3
"""
Examples demonstrating various use cases for auto-interleaved generation.
"""

import sys
sys.path.append("..")

from auto_interleaved_inference import AutoInterleavedInferencer


def recipe_example(inferencer, gen_params):
    """Generate a recipe with interleaved instructions and images."""
    prompt = """Create a recipe for chocolate chip cookies with images showing key steps. 
    Start with the ingredients list, then show each major step with an accompanying image."""
    
    print("Recipe Generation Example")
    print("=" * 80)
    outputs = inferencer.auto_interleaved_generation(prompt, **gen_params)
    return outputs


def tutorial_example(inferencer, gen_params):
    """Generate a tutorial with step-by-step images."""
    prompt = """Create a tutorial on how to tie a tie. 
    For each step, provide a clear text description followed by an image showing that step."""
    
    print("Tutorial Generation Example")
    print("=" * 80)
    outputs = inferencer.auto_interleaved_generation(prompt, **gen_params)
    return outputs


def story_example(inferencer, gen_params):
    """Generate an illustrated story."""
    prompt = """Write a short children's story about a brave little robot exploring space. 
    Include illustrations for key moments in the story."""
    
    print("Illustrated Story Example")
    print("=" * 80)
    outputs = inferencer.auto_interleaved_generation(prompt, **gen_params)
    return outputs


def educational_example(inferencer, gen_params):
    """Generate educational content with diagrams."""
    prompt = """Explain the water cycle with diagrams. 
    Start with an overview, then explain each stage (evaporation, condensation, precipitation) with accompanying diagrams."""
    
    print("Educational Content Example")
    print("=" * 80)
    outputs = inferencer.auto_interleaved_generation(prompt, **gen_params)
    return outputs


def product_description_example(inferencer, gen_params):
    """Generate a product description with multiple views."""
    prompt = """Create a product description for a futuristic smartwatch. 
    Describe its features and show images of the watch from different angles and in different use cases."""
    
    print("Product Description Example")
    print("=" * 80)
    outputs = inferencer.auto_interleaved_generation(prompt, **gen_params)
    return outputs


def travel_guide_example(inferencer, gen_params):
    """Generate a travel guide with location images."""
    prompt = """Create a brief travel guide for visiting Tokyo. 
    Include recommendations for places to visit with images of each location."""
    
    print("Travel Guide Example")
    print("=" * 80)
    outputs = inferencer.auto_interleaved_generation(prompt, **gen_params)
    return outputs


# Advanced example with explicit vision tokens
def explicit_vision_tokens_example(inferencer, gen_params):
    """Example using explicit vision tokens in the prompt."""
    prompt = """I'll describe three different scenes. 
    
    First, a peaceful mountain lake at sunset <|vision_start|>
    
    Second, a bustling city street at night with neon lights <|vision_start|>
    
    Third, a cozy library with warm lighting and old books <|vision_start|>"""
    
    print("Explicit Vision Tokens Example")
    print("=" * 80)
    outputs = inferencer.auto_interleaved_generation(prompt, **gen_params)
    return outputs


if __name__ == "__main__":
    # This file is meant to be imported and used with a properly initialized inferencer
    print("This file contains example functions for auto-interleaved generation.")
    print("Import it and use with a properly initialized AutoInterleavedInferencer.")