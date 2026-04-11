"""ComfyUI-Gemma4-GGUF — Multimodal analysis & custom sampling nodes."""

import logging
import os

import folder_paths  # type: ignore  — provided by ComfyUI runtime
from comfy_api.latest import ComfyExtension, io

# ---------------------------------------------------------------------------
# Register the custom "LLM" model folder early so node schemas can enumerate it
# ---------------------------------------------------------------------------
_llm_dir = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(_llm_dir, exist_ok=True)
folder_paths.add_model_folder_path("LLM", _llm_dir)

WEB_DIRECTORY = "./web"


# ---------------------------------------------------------------------------
# V3 Extension (Nodes 2.0)
# ---------------------------------------------------------------------------

class DuffyNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .nodes import NODE_LIST
        return NODE_LIST


async def comfy_entrypoint():
    """ComfyUI V3 entry point."""
    logging.info("")
    logging.info("*** ComfyUI-Gemma4-GGUF extension detected...")
    logging.info("*** Initializing (Nodes 2.0 V3 Schema).")
    logging.info("")
    return DuffyNodesExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
