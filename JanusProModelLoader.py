import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from janus.models import VLChatProcessor
import comfy.model_management

class JanusProModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],
                    {"default": "deepseek-ai/Janus-Pro-7B"}
                ),
                "precision": (
                    ["fp32", "fp16", "bf16", "int8", "int4"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "local_dir": ("STRING", {"default": "models/Janus-Pro"})
            }
        }

    RETURN_TYPES = ("JANUS_PRO_MODEL", "VLC_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    CATEGORY = "Janus-Pro/Loaders"
    FUNCTION = "load_model"

    def load_model(self, model_name, precision, local_dir="models/Janus-Pro"):
        # 创建本地目录（如果不存在）
        os.makedirs(local_dir, exist_ok=True)
        
        # 提取模型短名称（如Janus-Pro-7B）
        model_shortname = model_name.split("/")[-1]
        model_local_path = os.path.join(local_dir, model_shortname)
        
        # 检查本地模型是否存在
        if not self._validate_local_model(model_local_path):
            print(f"Model not found locally, downloading {model_name}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_local_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
        return self._load_from_local(
            model_local_path=model_local_path,
            precision=precision
        )

    def _validate_local_model(self, model_path):
        """验证本地模型完整性"""
        required_files = [
            "config.json",
            # "pytorch_model.bin",
            # "vocab.json",
            # "special_tokens_map.json"
        ]
        
        return all(
            os.path.exists(os.path.join(model_path, f))
            for f in required_files
        )

    def _load_from_local(self, model_local_path, precision):
        """从本地目录加载模型"""
        # 配置量化参数
        bnb_config = self._get_quant_config(precision)
        
        # 加载处理器
        processor = VLChatProcessor.from_pretrained(model_local_path)
        
        # 配置精度
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }.get(precision, torch.bfloat16)

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_local_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto"
        ).eval()

        # 显存优化
        self._optimize_memory(model, precision)
        
        return (model, processor)

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