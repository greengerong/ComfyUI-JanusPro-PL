import torch
import io
import base64
import numpy as np
from PIL import Image
from typing import List, Dict

class JanusProUtils:
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.clone().detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        image_np = tensor.numpy().transpose(1, 2, 0) * 255
        return Image.fromarray(image_np.astype(np.uint8))

    @staticmethod
    def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
        pil_images = []
        for message in conversations:
            if "images" not in message:
                continue
            for image_data in message["images"]:
                if isinstance(image_data, Image.Image):
                    pil_images.append(image_data.convert("RGB"))
                elif image_data.startswith("data:image"):
                    _, data = image_data.split(",", 1)
                    pil_images.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
                else:
                    pil_images.append(Image.open(image_data).convert("RGB"))
        return pil_images

    @staticmethod
    def validate_image_size(image_size: int):
        if image_size != 384:
            raise ValueError(f"Currently only support 384 resolution, got {image_size}")
          
