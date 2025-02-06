from .JanusProModelLoader import JanusProModelLoader
from .JanusProImageUnderstanding import JanusProImageUnderstanding
from .JanusProImageGenerator import JanusProImageGenerator

NODE_CLASS_MAPPINGS = {
    "JanusProModelLoader": JanusProModelLoader,
    "JanusProImageUnderstanding": JanusProImageUnderstanding,
    "JanusProImageGenerator": JanusProImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JanusProModelLoader": "🔮 Janus-Pro Model Loader",
    "JanusProImageUnderstanding": "🖼️ Janus Image Understanding",
    "JanusProImageGenerator": "🎨 Janus Image Generator"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
