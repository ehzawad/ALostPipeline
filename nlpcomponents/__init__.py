from typing import TYPE_CHECKING

__all__ = [
    "NLPPipelineConfig",
    "NLPPipeline",
]

if TYPE_CHECKING:
    from .config import NLPPipelineConfig
    from .core.pipeline import NLPPipeline

def __getattr__(name: str):
    if name == "NLPPipelineConfig":
        from .config import NLPPipelineConfig
        return NLPPipelineConfig
    if name == "NLPPipeline":
        from .core.pipeline import NLPPipeline
        return NLPPipeline
    raise AttributeError(f"module 'nlpcomponents' has no attribute {name!r}")
