import torch
import comfy.model_management
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from janus.models import VLChatProcessor

class JanusProModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"], {"default": "deepseek-ai/Janus-Pro-7B"}),
                "precision": (["fp32", "fp16", "bf16", "int8", "int4"], {"default": "bf16"}),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": "models/janus_pro"})
            }
        }

    RETURN_TYPES = ("JANUS_PRO_MODEL", "VLC_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    CATEGORY = "Janus-Pro/Loaders"
    FUNCTION = "load_model"

    def load_model(self, model_name, precision, cache_dir="models/janus_pro"):
        bnb_config = self._get_quant_config(precision)
        vl_chat_processor = VLChatProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }.get(precision, torch.bfloat16)

        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir
        ).eval()

        self._optimize_memory(vl_gpt, precision)
        return (vl_gpt, vl_chat_processor)

    def _get_quant_config(self, precision):
        if precision == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        if precision == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        return None

    def _optimize_memory(self, model, precision):
        device = comfy.model_management.get_torch_device()
        if precision not in ["int8", "int4"]:
            if comfy.model_management.should_use_fp16(device) and precision == "fp16":
                model = model.half()
            model.to(device)
          
