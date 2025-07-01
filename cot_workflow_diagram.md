# BAGEL Chain of Thought Iterative Refinement Workflow

## Overview
The `cot_inference.py` implements an iterative refinement workflow for image generation using the BAGEL-7B-MoT model. The process consists of two main phases:

1. **Initial Generation Phase**: Generate an image from a text prompt
2. **Iterative Refinement Loop**: Reflect on the generated image and refine it until satisfactory

## Key Features (cot_inference.py)
- **Chain of Thought Generation** - Iterative refinement with reflection
- **Accumulative Mode** - Optional context building across iterations
- **Configurable max_iterations** - Default 10, user-configurable
- **Structured Reflection** - REQUIREMENTS_MET determines completion
- **CLI Interface** - Command-line arguments for all parameters
- **Centralized Output** - Results saved to `./results/cot_outputs/`

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CHAIN OF THOUGHT WORKFLOW                        │
│                  real_chain_of_thought_generation()                  │
├─────────────────────────────────────────────────────────────────────┤
│ Parameters:                                                          │
│   • prompt: str                                                      │
│   • output_dir: str  (auto-generated in results/cot_outputs/)       │
│   • max_iterations: int = 10  (configurable via --max-iterations)   │
│   • accumulative: bool = False  (enabled via --accumulative flag)   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────────┐
                      │ accumulative == True?       │
                      └────────────┬────────────────┘
                                   │
                    YES ◄──────────┴──────────► NO
                     │                           │
                     ▼                           ▼
         ┌───────────────────────┐   ┌───────────────────────┐
         │ Initialize:           │   │ Initialize:           │
         │ • reflection_history  │   │ • No history tracking │
         │ • edit_history        │   │                       │
         └───────────────────────┘   └───────────────────────┘
                     │                           │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: INITIAL GENERATION                       │
├─────────────────────────────────────────────────────────────────────┤
│ Input: Text Prompt                                                   │
│ Parameters:                                                          │
│   • cfg_text_scale: 4.0                                             │
│   • cfg_img_scale: 1.0                                              │
│   • cfg_interval: [0.4, 1.0]                                        │
│   • timestep_shift: 3.0                                             │
│   • num_timesteps: 50                                               │
│   • cfg_renorm_min: 0.0                                             │
│   • cfg_renorm_type: "global"                                       │
│                                                                      │
│ Output: Initial Generated Image                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 2: ITERATIVE REFINEMENT LOOP                     │
│                    (max_iterations = 10)                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │     START ITERATION       │◄─────────┐
                    └───────────────────────────┘          │
                                    │                      │
                                    ▼                      │
┌─────────────────────────────────────────────────────────────────────┐
│                        REFLECTION PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│ Parameters:                                                          │
│   • understanding_output: True                                       │
│   • think: True                                                      │
│   • max_think_token_n: 2000                                         │
│   • do_sample: False                                                │
│   • text_temperature: 0.3                                           │
│                                                                      │
│ Reflection Prompt Structure (Non-Accumulative):                     │
│   1. "Look at this image and the original prompt: '{prompt}'"      │
│   2. "Examine the image carefully and verbalize what you see"       │
│   3. "Reflect on whether the image fully satisfies requirements"    │
│   4. "Identify what is missing, incorrect, or needs improvement"    │
│                                                                      │
│ Reflection Prompt Structure (Accumulative):                          │
│   1. Same as above PLUS:                                            │
│   2. "Previous reflection history: [shows past attempts]"           │
│   3. "Based on the history, examine the current image carefully"    │
│   4. "What improvements were successful?"                          │
│   5. "What issues remain unresolved?"                              │
│                                                                      │
│ Expected Output Format (SIMPLIFIED):                                 │
│   • OBSERVATION: [What I see in the image]                         │
│   • REQUIREMENTS_MET: [Yes/No]                                      │
│   • NEEDED_IMPROVEMENTS: [Specific list of changes]                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │    PARSE REFLECTION       │
                    │  Extract score & status   │
                    └───────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────┐
                         │   DECISION      │
                         │                 │
                         │ REQUIREMENTS_MET│
                         │    == "Yes"?    │
                         │                 │
                         │ (Score removed) │
                         └────────┬────────┘
                                  │
                    YES ◄─────────┴─────────► NO
                     │                         │
                     ▼                         ▼
        ┌──────────────────────┐  ┌─────────────────────────────────┐
        │   SATISFACTORY!      │  │          EDIT PHASE             │
        │                      │  ├─────────────────────────────────┤
        │  • Save best image   │  │ Parameters:                     │
        │  • Exit loop         │  │   • cfg_text_scale: 4.0        │
        │  • Return results    │  │   • cfg_img_scale: 2.0 ◄─(!)   │
        └──────────────────────┘  │   • cfg_interval: [0.0, 1.0]◄─(!)│
                                  │   • timestep_shift: 3.0        │
                                  │   • num_timesteps: 50          │
                                  │   • cfg_renorm_min: 0.0        │
                                  │   • cfg_renorm_type:           │
                                  │     "text_channel" ◄─(!)       │
                                  │                                 │
                                  │ Edit Instruction Generation:    │
                                  │ • Non-Accumulative:             │
                                  │   Based on NEEDED_IMPROVEMENTS │
                                  │ • Accumulative:                 │
                                  │   Includes context from last 3 │
                                  │   edit attempts in history      │
                                  └─────────────────────────────────┘
                                                │
                                                ▼
                                  ┌─────────────────────────────────┐
                                  │      EXECUTE IMAGE EDIT         │
                                  │                                 │
                                  │  Apply edit instruction to      │
                                  │  current image                  │
                                  └─────────────────────────────────┘
                                                │
                                                ▼
                                  ┌─────────────────────────────────┐
                                  │    UPDATE STATE & CONTINUE      │
                                  │                                 │
                                  │  • current_image = edited_image │
                                  │  • Update best_image            │
                                  │  • If accumulative:             │
                                  │    - Append to reflection_history│
                                  │    - Append to edit_history     │
                                  │  • Increment iteration          │
                                  └─────────────────────────────────┘
                                                │
                                                └─────────────────┘
```

## Key Components

### 1. Hyperparameters
The workflow uses different parameter sets for different phases:

#### Initial Generation Parameters
- **cfg_text_scale**: 4.0 - Controls text guidance strength
- **cfg_img_scale**: 1.0 - Controls image guidance (low for initial generation)
- **cfg_interval**: [0.4, 1.0] - Classifier-free guidance application range
- **cfg_renorm_type**: "global" - Normalization method

#### Edit Phase Parameters (Different!)
- **cfg_text_scale**: 4.0 - Same text guidance
- **cfg_img_scale**: 2.0 - Higher image guidance for edits
- **cfg_interval**: [0.0, 1.0] - Full range for editing
- **cfg_renorm_type**: "text_channel" - Different normalization for edits

### 2. Control Flow Conditions
- **max_iterations**: 10 - Maximum refinement loops (increased from 3)
- **requirements_met**: Boolean check from reflection
- **Termination**: When requirements_met == "Yes" (satisfaction score removed)
- **accumulative**: Boolean flag controlling context accumulation

### 3. Prompts Structure

#### Original Prompt
User-provided description of desired image

#### Accumulative Mode Flag
- **True**: Maintains history across iterations
- **False**: Each iteration is independent

#### Reflection Prompt (Structured)
```
Look at this image and the original prompt: "{prompt}"

First, examine the image carefully and verbalize what you see in detail.

Then, reflect on whether the image fully satisfies the prompt requirements:
1. Does the image contain ALL the elements mentioned in the prompt?
2. Are the details accurate and properly executed?

Specifically identify what is missing, incorrect, or needs improvement. Be very specific about what changes are needed.

Provide your assessment in this format:
OBSERVATION: [What I see in the image]
REQUIREMENTS_MET: [Yes/No - whether all prompt requirements are satisfied]
NEEDED_IMPROVEMENTS: [Specific list of what needs to be changed or added]
```

#### Edit Instruction (Dynamic)
Generated based on reflection output:
- If NEEDED_IMPROVEMENTS exists: "Based on your analysis, edit this image to address these specific issues: {needed_improvements}. Ensure the final result perfectly matches the original prompt: '{prompt}'"
- Otherwise: "Improve this image to better match: {prompt}. Make it more accurate and higher quality."

### 4. State Management
- **current_image**: Updated after each edit
- **best_image**: Updated when requirements are met
- **results**: Array tracking all steps, reflections, and edits
- **reflection_history**: (Accumulative mode only) List of all reflection texts
- **edit_history**: (Accumulative mode only) List of all edit instructions

### 5. Output Files
For each run, the system generates:
- `step_0_initial.png` - Initial generation
- `step_N_reflection.txt` - Reflection analysis for iteration N
- `step_N_edited.png` - Edited image for iteration N
- `final_best.png` - Best scoring image
- `final_current.png` - Last generated image
- `cot_summary.json` - Complete process summary
- `real_cot_log.txt` - Detailed execution log
- `original_prompt.txt` - Saved original prompt
- `visualization.md` - Markdown visualization of the process

## Example Flow Comparison

### Non-Accumulative Mode (accumulative=False)
```
1. User provides prompt: "a red sports car..."
2. Initial generation → step_0_initial.png
3. Iteration 1:
   - Reflect: "Missing person in blue jacket"
   - Edit: "Add person in blue jacket"
   - Result: step_1_edited.png
4. Iteration 2:
   - Reflect: "Car not red enough" (no awareness of previous fix)
   - Edit: "Make car more red"
   - Result: step_2_edited.png (might lose the person)
5. Iteration 3:
   - Reflect: "Missing person again"
   - Edit: "Add person" (repeating work)
   - Continue until REQUIREMENTS_MET or max iterations
```

### Accumulative Mode (accumulative=True)
```
1. User provides prompt: "a red sports car..."
2. Initial generation → step_0_initial.png
3. Iteration 1:
   - Reflect: "Missing person in blue jacket"
   - Edit: "Add person in blue jacket"
   - History saved: ["Added person"]
4. Iteration 2:
   - Reflect: "Previous: Added person ✓. Still need: Car more red"
   - Edit: "Make car more red while keeping the person"
   - History saved: ["Added person", "Made car red"]
5. Iteration 3:
   - Reflect: "All elements present. REQUIREMENTS_MET: Yes"
   - SATISFACTORY! Exit loop
6. Save final_best.png with all improvements preserved
```

## Key Differences Summary

| Feature | Non-Accumulative | Accumulative |
|---------|------------------|--------------|
| Context | Each iteration independent | History builds across iterations |
| Reflection Prompt | Standard format only | Includes previous attempts |
| Edit Instructions | Based on current state only | References last 3 edit attempts |
| Risk of Regression | High - may undo fixes | Low - aware of what worked |
| Convergence Speed | Slower, may cycle | Faster, learns from attempts |
| Memory Usage | Constant | Grows with iterations |

## Command Line Usage

```bash
# Basic usage
python cot_inference.py --prompt "your prompt here"

# With accumulative mode
python cot_inference.py --prompt "your prompt" --accumulative

# Custom iterations and output
python cot_inference.py --prompt "your prompt" --max-iterations 15 --output-dir ./my_output

# With custom seed
python cot_inference.py --prompt "your prompt" --seed 123
```