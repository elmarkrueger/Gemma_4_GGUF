"""ComfyUI-Gemma4-GGUF — Multimodal analysis node using Gemma-4-E4B-it via llama.cpp."""

import os

import folder_paths  # type: ignore  — provided by ComfyUI runtime

from .nodes import GemmaGGUFAnalyzer

# ---------------------------------------------------------------------------
# Register the custom "LLM" model folder so ComfyUI scans models/LLM/
# ---------------------------------------------------------------------------
_llm_dir = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(_llm_dir, exist_ok=True)
folder_paths.add_model_folder_path("LLM", _llm_dir)

# ---------------------------------------------------------------------------
# Node registration (required by ComfyUI)
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "GemmaGGUFAnalyzer": GemmaGGUFAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmaGGUFAnalyzer": "Gemma-4 GGUF Multimodal Analyzer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
