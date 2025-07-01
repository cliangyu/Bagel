#!/bin/bash

# Compare accumulative vs non-accumulative context using command line flags

echo "🔬 Comparing Chain of Thought with and without accumulative context"
echo "=================================================================="

# Test prompt - using the challenging prompt from the original code
PROMPT="A perfectly symmetrical dinner table viewed from above, with exactly 8 identical wine glasses arranged in a precise octagon, each glass casting a different colored shadow (red, blue, green, yellow, purple, orange, pink, white), with 3 lit candles of varying heights in the center forming a triangle, 2 identical forks placed at 45-degree angles flanking each candle, and 12 rose petals scattered in a specific spiral pattern between the glasses."

# Common parameters
MAX_ITERATIONS=10
SEED=42

echo -e "\n📝 Test prompt: $PROMPT"
echo "🔄 Max iterations: $MAX_ITERATIONS"
echo "🌱 Random seed: $SEED"
echo

# Run 1: WITHOUT accumulative context
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 RUN 1: WITHOUT Accumulative Context"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

START_TIME=$(date +%s)
conda run -n bagel python3 cot_inference.py \
    --prompt "$PROMPT" \
    --max-iterations $MAX_ITERATIONS \
    --seed $SEED
END_TIME=$(date +%s)
RUN1_TIME=$((END_TIME - START_TIME))

echo -e "\n⏱️  Run 1 completed in ${RUN1_TIME}s"

# Pause between runs
echo -e "\n⏸️  Pausing for 5 seconds before second run..."
sleep 5

# Run 2: WITH accumulative context
echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 RUN 2: WITH Accumulative Context"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

START_TIME=$(date +%s)
conda run -n bagel python3 cot_inference.py \
    --prompt "$PROMPT" \
    --max-iterations $MAX_ITERATIONS \
    --seed $SEED \
    --accumulative
END_TIME=$(date +%s)
RUN2_TIME=$((END_TIME - START_TIME))

echo -e "\n⏱️  Run 2 completed in ${RUN2_TIME}s"

# Summary
echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 COMPARISON SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Execution times:"
echo "  - Without accumulative: ${RUN1_TIME}s"
echo "  - With accumulative: ${RUN2_TIME}s"
echo
echo "📁 Output directories created:"
echo "  - ./results/cot_outputs/cot_output_no_accumulative_*"
echo "  - ./results/cot_outputs/cot_output_accumulative_*"
echo
echo "🔍 To analyze the differences:"
echo "  1. Compare reflection files to see if accumulative mode contains history"
echo "  2. Check if accumulative mode converges faster or produces better results"
echo "  3. Look for any regression issues in non-accumulative mode"

# Optional: Quick check for latest directories
echo -e "\n📂 Latest output directories:"
ls -dt ./results/cot_outputs/cot_output_*/ 2>/dev/null | head -2 || echo "No output directories found yet."