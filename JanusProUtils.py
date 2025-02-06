import torch
import io
import base64
import numpy as np
from PIL import Image
from typing import List, Dict

class JanusProUtils:
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convert a ComfyUI tensor to a PIL Image.
        Supports multiple input formats:
        - (B x H x W x C)
        - (H x W x C)
        - (C x H x W)
        - (B x C x H x W)
        """
        try:
            # Ensure tensor is on CPU and detached
            tensor = tensor.detach().cpu()

            # Handle batch dimension
            if tensor.dim() == 4:  # B x H x W x C or B x C x H x W
                tensor = tensor[0]  # Take the first image in the batch

            # Handle channel position
            if tensor.dim() == 3:
                if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # C x H x W
                    tensor = tensor.permute(1, 2, 0)  # Convert to H x W x C
                elif tensor.shape[2] == 3 or tensor.shape[2] == 1:  # H x W x C
                    pass  # Already in correct format
                else:
                    raise ValueError(f"Unsupported channel format: {tensor.shape}")

            # Handle single-channel (grayscale) images
            if tensor.shape[2] == 1:
                tensor = tensor.repeat(1, 1, 3)  # Convert to RGB by repeating the channel

            # Normalize pixel values
            if tensor.max() <= 1.0:  # Assume values are in [0, 1]
                tensor = tensor * 255

            # Convert to numpy and ensure uint8 type
            image_np = tensor.numpy().astype(np.uint8)

            # Validate final shape
            if image_np.shape[2] != 3:
                raise ValueError(f"Invalid channel dimension: {image_np.shape}. Expected 3 channels (RGB).")

            return Image.fromarray(image_np)

        except Exception as e:
            raise RuntimeError(
                f"Failed to convert tensor to PIL: {str(e)}\n"
                f"Input tensor shape: {tensor.shape}\n"
                f"Input tensor range: [{tensor.min()}, {tensor.max()}]"
            )
  

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
          
