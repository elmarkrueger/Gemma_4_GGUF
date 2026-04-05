import logging
import os
import re
from typing import Any, Optional

import folder_paths  # type: ignore  — provided by ComfyUI runtime
import torch
from llama_cpp import Llama  # type: ignore

from .utils.media import (audio_to_data_uri, image_tensor_to_data_uri,
                          video_tensor_to_frame_list)
from .utils.memory import unload_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: filter GGUF file lists for the dropdown menus
# ---------------------------------------------------------------------------

def _get_gguf_models() -> list[str]:
    """Return .gguf filenames from the LLM folder, excluding mmproj files."""
    try:
        files = folder_paths.get_filename_list("LLM")
    except Exception:
        return []
    return sorted(
        f for f in files
        if f.lower().endswith(".gguf") and "mmproj" not in f.lower()
    )


def _get_mmproj_models() -> list[str]:
    """Return mmproj .gguf filenames from the LLM folder."""
    try:
        files = folder_paths.get_filename_list("LLM")
    except Exception:
        return []
    return sorted(
        f for f in files
        if f.lower().endswith(".gguf") and "mmproj" in f.lower()
    )


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class GemmaGGUFAnalyzer:
    """ComfyUI node for multimodal analysis using Gemma-4-E4B-it via llama.cpp.

    Supports text, image, video (≤30 s), and audio (≤60 s) inputs.
    """

    def __init__(self):
        self.model_instance: Optional[Llama] = None
        self.current_model_path: Optional[str] = None
        self.current_mmproj_path: Optional[str] = None
        self.current_n_gpu_layers: Optional[int] = None
        self.current_n_ctx: Optional[int] = None

    # ------------------------------------------------------------------ #
    # ComfyUI interface
    # ------------------------------------------------------------------ #

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "gguf_model": (
                    _get_gguf_models(),
                    {"tooltip": "GGUF model file from models/LLM (excludes mmproj files)"},
                ),
                "mmproj_model": (
                    _get_mmproj_models(),
                    {"tooltip": "Multimodal projector file (mmproj-*.gguf) — must be in the same models/LLM folder as the model"},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful multimodal analyzer.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Analyze the provided input.",
                    },
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable model reasoning / thinking mode"},
                ),
                "strip_thinking_tags": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Remove <think>…</think> blocks from output"},
                ),
                "unload_model": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Aggressively free VRAM/RAM after inference"},
                ),
                # --- Inference parameters ---
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 128000, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_k": (
                    "INT",
                    {"default": 40, "min": 0, "max": 500, "step": 1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "min_p": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1},
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1},
                ),
                "mirostat_mode": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2, "step": 1},
                ),
                "mirostat_tau": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "mirostat_eta": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "-1 = offload all layers to GPU",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 512,
                        "max": 131072,
                        "step": 512,
                        "tooltip": "Context window size (Gemma-4 supports up to 128 K)",
                    },
                ),
                "video_fps": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Temporal sampling rate for video input (frames per second)",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),   # batch of frames [F, H, W, 3]
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_text",)
    FUNCTION = "analyze"
    CATEGORY = "LLM"
    OUTPUT_NODE = False

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    def _needs_reload(
        self, model_path: str, mmproj_path: str, n_gpu_layers: int, n_ctx: int
    ) -> bool:
        """Check whether a new model load is required."""
        if self.model_instance is None:
            return True
        return (
            self.current_model_path != model_path
            or self.current_mmproj_path != mmproj_path
            or self.current_n_gpu_layers != n_gpu_layers
            or self.current_n_ctx != n_ctx
        )

    def _load_model(
        self, model_path: str, mmproj_path: str, n_gpu_layers: int, n_ctx: int
    ) -> None:
        """Load the GGUF model and multimodal projector into memory."""
        # Unload previous instance first
        if self.model_instance is not None:
            logger.info("Unloading previous model before reload")
            unload_llm(self.model_instance)
            self.model_instance = None

        logger.info("Loading model: %s", model_path)
        logger.info("Loading mmproj: %s", mmproj_path)

        self.model_instance = Llama(
            model_path=model_path,
            clip_model_path=mmproj_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

        self.current_model_path = model_path
        self.current_mmproj_path = mmproj_path
        self.current_n_gpu_layers = n_gpu_layers
        self.current_n_ctx = n_ctx
        logger.info("Model loaded successfully (n_gpu_layers=%d, n_ctx=%d)", n_gpu_layers, n_ctx)

    # ------------------------------------------------------------------ #
    # Main execution
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        gguf_model: str,
        mmproj_model: str,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: bool,
        strip_thinking_tags: bool,
        unload_model: bool,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repeat_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        seed: int,
        n_gpu_layers: int,
        n_ctx: int,
        video_fps: float,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[dict] = None,
    ) -> tuple[str]:
        try:
            return self._run_inference(
                gguf_model=gguf_model,
                mmproj_model=mmproj_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                enable_thinking=enable_thinking,
                strip_thinking_tags=strip_thinking_tags,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                seed=seed,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                video_fps=video_fps,
                image=image,
                video=video,
                audio=audio,
            )
        except FileNotFoundError as e:
            error_msg = f"[GemmaGGUFAnalyzer] File not found: {e}"
            logger.error(error_msg)
            return (error_msg,)
        except ValueError as e:
            error_msg = f"[GemmaGGUFAnalyzer] Validation error: {e}"
            logger.error(error_msg)
            return (error_msg,)
        except Exception as e:
            error_msg = f"[GemmaGGUFAnalyzer] Inference failed: {e}"
            logger.error(error_msg, exc_info=True)
            return (error_msg,)
        finally:
            if unload_model:
                logger.info("Unloading model (unload_model=True)")
                unload_llm(self.model_instance)
                self.model_instance = None
                self.current_model_path = None
                self.current_mmproj_path = None
                self.current_n_gpu_layers = None
                self.current_n_ctx = None

    def _run_inference(
        self,
        gguf_model: str,
        mmproj_model: str,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: bool,
        strip_thinking_tags: bool,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repeat_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        seed: int,
        n_gpu_layers: int,
        n_ctx: int,
        video_fps: float,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[dict] = None,
    ) -> tuple[str]:
        # ----- Resolve file paths ----- #
        model_path = folder_paths.get_full_path("LLM", gguf_model)
        mmproj_path = folder_paths.get_full_path("LLM", mmproj_model)

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"GGUF model not found: '{gguf_model}'. "
                "Ensure it is placed in ComfyUI/models/LLM/"
            )
        if not mmproj_path or not os.path.isfile(mmproj_path):
            raise FileNotFoundError(
                f"Multimodal projector not found: '{mmproj_model}'. "
                "Both the model .gguf and its mmproj-*.gguf must reside "
                "together in ComfyUI/models/LLM/"
            )

        # ----- Load / cache model ----- #
        if self._needs_reload(model_path, mmproj_path, n_gpu_layers, n_ctx):
            self._load_model(model_path, mmproj_path, n_gpu_layers, n_ctx)

        # ----- Build user content array ----- #
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
        ]

        if image is not None:
            user_content.append(image_tensor_to_data_uri(image))

        if video is not None:
            user_content.extend(
                video_tensor_to_frame_list(
                    video, target_fps=video_fps, n_ctx=n_ctx,
                )
            )

        if audio is not None:
            user_content.append(audio_to_data_uri(audio))

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # ----- Thinking mode ----- #
        chat_template_kwargs: dict[str, Any] = {}
        if not enable_thinking:
            chat_template_kwargs["enable_thinking"] = False
            # Adjust defaults when thinking is disabled (model developer recommendation)
            if temperature == 0.8:
                temperature = 0.7
                logger.info("Thinking disabled: temperature auto-adjusted to 0.7")
            if presence_penalty == 0.0:
                presence_penalty = 1.5
                logger.info("Thinking disabled: presence_penalty auto-adjusted to 1.5")

        # ----- Run inference ----- #
        completion = self.model_instance.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            seed=seed if seed >= 0 else None,
            chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
        )

        response_text: str = completion["choices"][0]["message"]["content"] or ""

        # ----- Strip thinking tags if requested ----- #
        if strip_thinking_tags:
            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            ).strip()

        return (response_text,)
