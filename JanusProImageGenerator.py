import torch
import numpy as np
from PIL import Image
from .JanusProUtils import JanusProUtils
from typing import List

class JanusProImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JANUS_PRO_MODEL",),
                "processor": ("VLC_PROCESSOR",),
                "prompt": ("STRING", {"default": "A beautiful sunset over mountains", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "parallel_size": ("INT", {"default": 16, "min": 1, "max": 64}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0}),
                "image_size": ("STRING", {"default": "384"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "Janus-Pro/Generation"
    FUNCTION = "generate_image"

    def generate_image(self, model, processor, prompt, temperature, parallel_size, cfg_scale, image_size, seed):
        try:
            image_size = int(image_size)
            JanusProUtils.validate_image_size(image_size)
        except ValueError as e:
            raise ValueError(f"Invalid image size: {str(e)}")

        torch.manual_seed(seed)
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""}
        ]
        
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt=""
        )
        full_prompt = sft_format + processor.image_start_tag

        generated_tokens = self._generate_tokens(
            model=model,
            processor=processor,
            prompt=full_prompt,
            parallel_size=parallel_size,
            cfg_scale=cfg_scale,
            temperature=temperature,
            image_size=image_size
        )

        images = self._decode_tokens(
            model=model,
            tokens=generated_tokens,
            image_size=image_size,
            parallel_size=parallel_size
        )

        return (self._images_to_tensor(images),)

    def _generate_tokens(self, model, processor, prompt, parallel_size, cfg_scale, temperature, image_size):
        input_ids = processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(model.device)
        
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.long, device=model.device)
        tokens[:, :] = input_ids
        tokens[1::2, 1:-1] = processor.pad_id

        image_token_num = (image_size // 16)**2 * 3
        generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.long, device=model.device)
        
        with torch.no_grad():
            inputs_embeds = model.language_model.get_input_embeddings()(tokens)
            past_key_values = None
            
            for i in range(image_token_num):
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                
                logits = model.gen_head(outputs.last_hidden_state[:, -1, :])
                logit_cond = logits[0::2]
                logit_uncond = logits[1::2]
                
                logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)
                
                next_token = torch.multinomial(probs, 1)
                generated_tokens[:, i] = next_token.squeeze()
                
                img_embeds = model.prepare_gen_img_embeds(
                    torch.cat([next_token, next_token], dim=0)
                )
                inputs_embeds = img_embeds.unsqueeze(1)

        return generated_tokens

    def _decode_tokens(self, model, tokens, image_size, parallel_size):
        patch_size = 16
        shape = [parallel_size, 3, image_size//patch_size, image_size//patch_size]
        
        decoded = model.gen_vision_model.decode_code(
            tokens.to(torch.int),
            shape=shape
        )
        
        decoded = decoded.to(torch.float32).cpu().numpy()
        decoded = np.transpose(decoded, (0, 2, 3, 1))
        decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return [Image.fromarray(img) for img in decoded]

    def _images_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in images:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr)[None,])
        return torch.cat(tensors, dim=0)
  
