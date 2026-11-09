"""
Speculative Decoding Engine: Low-Latency Inference with Rejection Sampling.
"""

from .sampler import modified_rejection_sampling, sample_from_logits
from .models import SimpleCausalLM
from .telemetry import GenerationTelemetry
from .engine import SpeculativeDecodingEngine

__all__ = [
    "modified_rejection_sampling",
    "sample_from_logits",
    "SimpleCausalLM",
    "GenerationTelemetry",
    "SpeculativeDecodingEngine",
]
