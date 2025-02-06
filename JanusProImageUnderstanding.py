import torch
from .JanusProUtils import JanusProUtils
from typing import List, Dict

class JanusProImageUnderstanding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JANUS_PRO_MODEL",),
                "processor": ("VLC_PROCESSOR",),
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "default": "Describe this image in detail for stable diffusion prompt",
                    "multiline": True
                }),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    CATEGORY = "Janus-Pro/Processing"
    FUNCTION = "analyze_image"

    def analyze_image(self, model, processor, image, question, max_tokens):
        pil_image = JanusProUtils.tensor_to_pil(image)
        conversation = [{
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [pil_image]
        }, {"role": "<|Assistant|>", "content": ""}]

        pil_images = JanusProUtils.load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(model.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        tokenizer = processor.tokenizer
        
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return (self._clean_output(answer),)

    def _clean_output(self, text: str) -> str:
        return text.strip().replace("<|im_end|>", "").replace("<|im_start|>", "")
      
