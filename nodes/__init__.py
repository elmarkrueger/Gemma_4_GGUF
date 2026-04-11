from .gemma_gguf_analyzer import DuffyGemmaGGUFAnalyzer
from .klein_skin_sampler import DuffyKleinSkinSampler

NODE_LIST: list[type] = [
    DuffyGemmaGGUFAnalyzer,
    DuffyKleinSkinSampler,
]
