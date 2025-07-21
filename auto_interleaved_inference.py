from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple
import torch
from PIL import Image
from tqdm import tqdm
import logging

from inferencer import InterleaveInferencer
from data.data_utils import pil_img2rgb

# Set up logging
logger = logging.getLogger(__name__)


class AutoInterleavedInferencer(InterleaveInferencer):
    """
    Extends InterleaveInferencer to support automatic switching between text and image generation
    based on special tokens (<|vision_start|> and <|vision_end|>).
    """
    
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        super().__init__(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
        
        # Validate required token IDs
        required_tokens = ['start_of_image', 'end_of_image', 'eos_token_id']
        for token_name in required_tokens:
            if token_name not in new_token_ids:
                raise ValueError(f"Required token '{token_name}' not found in new_token_ids")
        
        self.vision_start_token_id = new_token_ids['start_of_image']
        self.vision_end_token_id = new_token_ids['end_of_image']
        self.eos_token_id = new_token_ids['eos_token_id']
        
        logger.info(f"Initialized AutoInterleavedInferencer with vision tokens: "
                   f"start={self.vision_start_token_id}, end={self.vision_end_token_id}")
        
    @torch.no_grad()
    def auto_interleaved_generation(
        self,
        prompt: str,
        max_text_length: int = 500,
        max_interleaved_blocks: int = 10,
        think: bool = False,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: List[float] = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shape: Tuple[int, int] = (1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        """
        Automatically generates interleaved text and images based on special tokens in the output.
        
        Args:
            prompt: Input prompt text
            max_text_length: Maximum tokens per text generation block
            max_interleaved_blocks: Maximum number of text/image blocks to generate
            Other args follow the same pattern as the parent class methods
            
        Returns:
            List of alternating text strings and PIL Images
            
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If generation fails
        """
        # Validate inputs
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        if max_text_length <= 0:
            raise ValueError("max_text_length must be positive")
            
        if max_interleaved_blocks <= 0:
            raise ValueError("max_interleaved_blocks must be positive")
            
        if len(image_shape) != 2 or any(s <= 0 for s in image_shape):
            raise ValueError("image_shape must be a 2-tuple of positive integers")
        
        output_list = []
        
        logger.info(f"Starting auto-interleaved generation with prompt: {prompt[:100]}...")
        
        # Initialize contexts
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Add system prompt if thinking is enabled
            if think:
                system_prompt = '''You should first think about the planning process in the mind and then generate content.
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> content here'''
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
            
            # Update context with user prompt
            cfg_text_context = deepcopy(gen_context)
            gen_context = self.update_context_text(prompt, gen_context)
            cfg_img_context = self.update_context_text(prompt, cfg_img_context)
            
            # Main generation loop
            for block_idx in range(max_interleaved_blocks):
                try:
                    # Generate text tokens one by one until we hit a special token
                    text_tokens = []
                    current_gen_context = deepcopy(gen_context)
                    
                    for token_idx in range(max_text_length):
                        # Generate next token
                        try:
                            next_token, current_gen_context = self._generate_next_token(
                                current_gen_context,
                                do_sample=do_sample,
                                temperature=text_temperature
                            )
                        except Exception as e:
                            logger.error(f"Error generating token at position {token_idx}: {e}")
                            raise RuntimeError(f"Token generation failed: {e}") from e
                        
                        if next_token is None:
                            logger.warning("Received None token, ending generation")
                            break
                        
                        token_id = next_token.item()
                        
                        # Check for special tokens
                        if token_id == self.eos_token_id:
                            # End of sequence
                            if text_tokens:
                                try:
                                    decoded_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
                                    if decoded_text.strip():  # Only add non-empty text
                                        output_list.append(decoded_text.strip())
                                except Exception as e:
                                    logger.warning(f"Error decoding text tokens: {e}")
                            return output_list
                        
                        elif token_id == self.vision_start_token_id:
                            # Switch to image generation
                            # First, decode and save any accumulated text
                            if text_tokens:
                                try:
                                    decoded_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
                                    if decoded_text.strip():  # Only add non-empty text
                                        output_list.append(decoded_text.strip())
                                except Exception as e:
                                    logger.warning(f"Error decoding text tokens: {e}")
                                    continue
                                # Update main context with the generated text
                                if 'decoded_text' in locals() and decoded_text.strip():
                                    gen_context = self.update_context_text(decoded_text, gen_context)
                                    cfg_img_context = self.update_context_text(decoded_text, cfg_img_context)
                            
                            # Add vision start token to context
                            vision_start_text = self.tokenizer.decode([token_id])
                            gen_context = self.update_context_text(vision_start_text, gen_context)
                            cfg_img_context = self.update_context_text(vision_start_text, cfg_img_context)
                            cfg_text_context = deepcopy(gen_context)
                            
                            # Generate image
                            logger.info(f"Generating image for block {block_idx + 1}")
                            try:
                                image = self.gen_image(
                                    image_shape,
                                    gen_context,
                                    cfg_text_precontext=cfg_text_context,
                                    cfg_img_precontext=cfg_img_context,
                                    cfg_text_scale=cfg_text_scale,
                                    cfg_img_scale=cfg_img_scale,
                                    cfg_interval=cfg_interval,
                                    timestep_shift=timestep_shift,
                                    num_timesteps=num_timesteps,
                                    cfg_renorm_min=cfg_renorm_min,
                                    cfg_renorm_type=cfg_renorm_type,
                                )
                                output_list.append(image)
                                logger.info(f"Successfully generated image {len([o for o in output_list if isinstance(o, Image.Image)])}")
                            except Exception as e:
                                logger.error(f"Error generating image: {e}")
                                raise RuntimeError(f"Image generation failed: {e}") from e
                            
                            # Update context with generated image for future generation
                            gen_context = self.update_context_image(
                                image, 
                                gen_context, 
                                vae=True,
                                vit=False  # Only VAE for generation context
                            )
                            cfg_text_context = deepcopy(gen_context)
                            
                            # Add vision end token
                            vision_end_text = self.tokenizer.decode([self.vision_end_token_id])
                            gen_context = self.update_context_text(vision_end_text, gen_context)
                            cfg_img_context = self.update_context_text(vision_end_text, cfg_img_context)
                            
                            break  # Start new text generation block
                        
                        else:
                            # Regular text token
                            text_tokens.append(token_id)
                            # Note: current_gen_context is already updated by _generate_next_token
                    
                    # If we accumulated text tokens without hitting special tokens
                    if text_tokens:
                        try:
                            decoded_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
                            if decoded_text.strip():
                                # Update main context with accumulated tokens
                                gen_context = self.update_context_text(decoded_text, gen_context)
                                if block_idx == max_interleaved_blocks - 1:
                                    output_list.append(decoded_text.strip())
                                    break
                        except Exception as e:
                            logger.warning(f"Error decoding final text tokens: {e}")
                            break
                        
                except Exception as e:
                    logger.error(f"Error in generation block {block_idx}: {e}")
                    # Return what we have so far rather than losing everything
                    if output_list:
                        logger.warning(f"Returning {len(output_list)} outputs generated before error")
                    return output_list
                    
        logger.info(f"Generation complete. Generated {len(output_list)} outputs.")
        return output_list
    
    @torch.no_grad()
    def _generate_next_token(
        self, 
        gen_context: Dict,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> tuple[Optional[torch.Tensor], Dict]:
        """
        Generate a single next token given the current context.
        
        Returns:
            Tuple of (next_token, updated_gen_context) or (None, gen_context) if generation should stop
            
        Raises:
            RuntimeError: If generation fails
        """
        if not isinstance(gen_context, dict) or 'past_key_values' not in gen_context:
            raise ValueError("Invalid generation context")
            
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        logger.debug(f"_generate_next_token: kv_lens={kv_lens}, type={type(kv_lens)}")
        logger.debug(f"_generate_next_token: ropes={ropes}, type={type(ropes)}")
        
        # Prepare generation input
        try:
            generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        except Exception as e:
            logger.error(f"Error in prepare_start_tokens: {e}")
            logger.error(f"kv_lens: {kv_lens}, type: {type(kv_lens)}")
            logger.error(f"ropes: {ropes}, type: {type(ropes)}")
            logger.error(f"new_token_ids: {self.new_token_ids}")
            if isinstance(kv_lens, list) and len(kv_lens) > 0:
                logger.error(f"kv_lens[0]: {kv_lens[0]}, type: {type(kv_lens[0])}")
            if isinstance(ropes, list) and len(ropes) > 0:
                logger.error(f"ropes[0]: {ropes[0]}, type: {type(ropes[0])}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Move tensors to device - use inherited device property
        generation_input = {
            key: tensor.to(device=self.device, dtype=self.dtype if tensor.dtype.is_floating_point else tensor.dtype)
            for key, tensor in generation_input.items()
        }
        
        # Get token from current position
        packed_text_embedding = self.model.language_model.model.embed_tokens(generation_input['packed_start_tokens'])
        query_lens = torch.ones_like(generation_input['packed_start_tokens'])
        
        # Convert kv_lens to tensor if it's a list - use inherited device
        if isinstance(kv_lens, list):
            if len(kv_lens) == 1 and isinstance(kv_lens[0], int):
                # Single element list with integer
                kv_lens_tensor = torch.tensor(kv_lens, device=self.device, dtype=torch.int32)
            else:
                kv_lens_tensor = torch.tensor(kv_lens, device=self.device, dtype=torch.int32)
        else:
            kv_lens_tensor = kv_lens
        
        packed_query_indexes = torch.cumsum(kv_lens_tensor, dim=0) + torch.arange(
            0, len(kv_lens) if isinstance(kv_lens, list) else kv_lens.shape[0], 
            device=self.device,
            dtype=torch.int32
        )
        
        # Prepare key-value indexes
        try:
            if isinstance(kv_lens, list):
                kv_len_value = kv_lens[0] if isinstance(kv_lens[0], int) else kv_lens[0].item()
            else:
                kv_len_value = kv_lens[0].item()
            uppacked = list(generation_input['packed_key_value_indexes'].split(kv_len_value, dim=0))
        except Exception as e:
            logger.error(f"Error preparing key-value indexes: {e}")
            logger.error(f"kv_lens: {kv_lens}, type: {type(kv_lens)}")
            logger.error(f"generation_input keys: {generation_input.keys()}")
            raise
        for i in range(len(uppacked)):
            uppacked[i] = uppacked[i] + i
        packed_key_value_indexes = torch.cat(uppacked, dim=0)
        
        # Forward pass
        extra_inputs = {}
        if self.model.use_moe:
            extra_inputs = {"mode": "und"}
            
        output = self.model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=generation_input['packed_query_position_ids'],
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=kv_lens_tensor,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,  # Fix: Update KV cache after each token
            is_causal=True,
            **extra_inputs,
        )
        
        # Get logits and sample next token
        pred_logits = self.model.language_model.lm_head(output.packed_query_sequence)
        
        if do_sample:
            probs = torch.nn.functional.softmax(pred_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(pred_logits, dim=-1)
        
        # Update the generation context with new KV cache state
        # The forward_inference call with update_past_key_values=True already updated past_key_values
        # We need to update kv_lens and ropes to reflect the new token
        updated_gen_context = gen_context.copy()
        updated_gen_context['kv_lens'] = [kv_lens[0] + 1] if isinstance(kv_lens, list) else [kv_lens.item() + 1]
        updated_gen_context['ropes'] = [ropes[0] + 1] if isinstance(ropes, list) else [ropes.item() + 1]
        
        return next_token, updated_gen_context