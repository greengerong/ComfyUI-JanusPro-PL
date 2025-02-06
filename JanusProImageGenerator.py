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
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0}),
                "image_size": ("STRING", {"default": "384"}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "Janus-Pro/Generation"
    FUNCTION = "generate_image"

    def generate_image(self, model, processor, prompt, system_prompt, temperature, num_images, cfg_scale, image_size, seed):
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
            system_prompt=system_prompt
        )
        full_prompt = sft_format + processor.image_start_tag

        generated_tokens = self._generate_tokens(
            model=model,
            processor=processor,
            prompt=full_prompt,
            num_images=num_images,
            image_size=image_size,
            max_tokens=576
        )

        images = self._decode_tokens(
            model=model,
            tokens=generated_tokens,
            image_size=image_size,
            parallel_size=num_images
        )

        return (self._images_to_tensor(images),)

    def _generate_tokens(self, model, processor, prompt, num_images, image_size, max_tokens):
        try:
            # 准备对话格式
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # 应用SFT模板
            sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt="",
            )
            formatted_prompt = sft_format + processor.image_start_tag

            # 编码输入
            input_ids = processor.tokenizer.encode(formatted_prompt)
            input_ids = torch.LongTensor(input_ids)

            # 准备并行生成的输入
            parallel_size = num_images
            tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
            for i in range(parallel_size*2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = processor.pad_id

            # 获取输入嵌入
            inputs_embeds = model.language_model.get_input_embeddings()(tokens)

            # 初始化生成的tokens
            image_token_num = 576  # 默认值
            generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()
            outputs = None

            # 逐token生成
            for i in range(image_token_num):
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds, 
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if i != 0 else None
                )
                hidden_states = outputs.last_hidden_state
                
                # 获取logits并应用CFG
                logits = model.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                cfg_weight = 5.0  # 默认CFG权重
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                
                # 采样下一个token
                temperature = 1.0
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                # 准备下一步的输入
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            return generated_tokens

        except Exception as e:
            print(f"生成过程中出错: {str(e)}")
            raise e

    def _decode_tokens(self, model, tokens, image_size, parallel_size):
        patch_size = 16
        shape = [parallel_size, 8, image_size//patch_size, image_size//patch_size]
        
        try:
            # 解码生成的tokens为图像
            decoded = model.gen_vision_model.decode_code(
                tokens.to(dtype=torch.int),
                shape=shape
            )
            
            # 转换为numpy数组并调整格式
            decoded = decoded.to(torch.float32).cpu().numpy()
            decoded = np.transpose(decoded, (0, 2, 3, 1))
            decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
            
            # 创建输出图像数组
            visual_img = np.zeros((parallel_size, image_size, image_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = decoded
            
            # 转换为PIL图像列表
            return [Image.fromarray(img) for img in visual_img]
            
        except Exception as e:
            print(f"解码错误: {str(e)}")
            print(f"tokens shape: {tokens.shape}")
            print(f"requested shape: {shape}")
            raise e

    def _images_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in images:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr)[None,])
        return torch.cat(tensors, dim=0)
  