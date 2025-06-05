import os
import sys
sys.path.append("/home/jiahuikchen/BAGEL")
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from inferencer import InterleaveInferencer
import random
import numpy as np


### MODEL LOADING
model_path = "/checkpoint/dream_v2/transfusion/BAGEL-7B-MoT"

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

gpu_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"] 
# Initialize an empty list to store the loaded models
models = []
# Loop through each model path and GPU device
for gpu_device_i in range(len(gpu_devices)):
    gpu_device = gpu_devices[gpu_device_i]

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Create a new Bagel configuration
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
    # Initialize the model with empty weights
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    # Load the model weights
    device_map = infer_auto_device_map(
        model,
        max_memory={gpu_device_i: "80GiB"},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    # Move the model to the specified GPU device
    model.to(gpu_device)
    # Append the loaded model to the list
    models.append(model.eval())

print(f"{len(gpu_devices)} models loaded")


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

inferencers = []
for model in models:
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )
    inferencers.append(inferencer)

# Image gen with think params 
t2i_think_params =dict(
    think=True,
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
)
# # Edit with think params
# edit_think_params = dict(
#     think=True,
#     max_think_token_n=1000,
#     do_sample=False,
#     # text_temperature=0.3,
#     cfg_text_scale=4.0,
#     cfg_img_scale=2.0,
#     cfg_interval=[0.0, 1.0],
#     timestep_shift=3.0,
#     num_timesteps=50,
#     cfg_renorm_min=0.0,
#     cfg_renorm_type="text_channel",
# )


### Multi-GPU inference (1 model per GPU)
# Load the JSON file containing the prompts
with open('/home/jiahuikchen/BAGEL/mmgw/mmgw.json', 'r') as f:
    data = json.load(f)
# category = "relative_positions"
category = "text"
prompts = data[category]

# Create an output directory for the generated images
output_dir = f"/home/jiahuikchen/BAGEL/mmgw/bagel/{category}"
os.makedirs(output_dir, exist_ok=True)

# Distribute prompts among the models
num_models = len(models)
prompts_per_model = len(prompts) // num_models

# Function to generate and save images
def generate_and_save_images(prompts, inferencer, device, output_dir, start_index):
    for i, prompt in enumerate(tqdm(prompts, desc=f"Processing on {device}", position=start_index)):
        print(prompt)
        output_dict = inferencer(text=prompt, **t2i_think_params)
        
        # Save the image
        image_path = os.path.join(output_dir, f"image_{start_index + i}.png")
        output_dict['image'].save(image_path)

# Use ThreadPoolExecutor for parallel execution
with ThreadPoolExecutor(max_workers=num_models) as executor:
    futures = []
    for i, inferencer in enumerate(inferencers):
        start_index = i * prompts_per_model
        end_index = start_index + prompts_per_model
        model_prompts = prompts[start_index:end_index]
        
        # Submit the task to the executor
        futures.append(executor.submit(generate_and_save_images, model_prompts, inferencer, gpu_devices[i], output_dir, start_index))
    # Handle any remaining prompts if the number of prompts is not perfectly divisible
    remaining_prompts = prompts[num_models * prompts_per_model:]
    if remaining_prompts:
        futures.append(executor.submit(generate_and_save_images, remaining_prompts, inferencers[-1], gpu_devices[-1], output_dir, num_models * prompts_per_model))
    # Wait for all futures to complete
    for future in as_completed(futures):
        future.result()
print("Image generation complete.")