#!/bin/bash
# Example usage script for interleaved_inference_script.py
# Make sure you activate the grpo conda environment first: conda activate grpo

MODEL_PATH="/data/users/leonlc/BAGEL-7B-MoT"  # Update this path
GPU_IDS="2,3,4,5"

echo "BAGEL Inference Examples"
echo "======================="
echo "Make sure to:"
echo "1. Update MODEL_PATH in this script"
echo "2. Activate conda environment: conda activate grpo"
echo "3. Have the model weights downloaded"
echo ""

# Image Generation Example
echo "1. Image Generation Example:"
python interleaved_inference_script.py \
    --model_path "$MODEL_PATH" \
    --mode generation \
    --prompt "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver" \
    --output_path output_generation.png \
    --gpu_ids "$GPU_IDS"

echo ""

# Image Generation with Think Example
echo "2. Image Generation with Think Example:"
python interleaved_inference_script.py \
    --model_path "$MODEL_PATH" \
    --mode generation_with_think \
    --prompt "a car made of small cars" \
    --output_path output_generation_think.png \
    --gpu_ids "$GPU_IDS"

echo ""

# Image Editing Example (requires test image)
echo "3. Image Editing Example:"
if [ -f "../test_images/women.jpg" ]; then
    python interleaved_inference_script.py \
        --model_path "$MODEL_PATH" \
        --mode editing \
        --image_path "../test_images/women.jpg" \
        --prompt "She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes" \
        --output_path output_editing.png \
        --gpu_ids "$GPU_IDS"
else
    echo "Skipping editing example - test_images/women.jpg not found"
fi

echo ""

# Understanding Example
echo "4. Understanding Example:"
if [ -f "../test_images/meme.jpg" ]; then
    python interleaved_inference_script.py \
        --model_path "$MODEL_PATH" \
        --mode understanding \
        --image_path "../test_images/meme.jpg" \
        --prompt "Can someone explain what's funny about this meme??" \
        --gpu_ids "$GPU_IDS"
else
    echo "Skipping understanding example - test_images/meme.jpg not found"
fi

echo ""


echo ""
echo "All examples completed!"
