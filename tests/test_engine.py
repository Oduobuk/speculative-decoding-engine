import pytest
import torch
from speculative_decoding.models import SimpleCausalLM
from speculative_decoding.engine import SpeculativeDecodingEngine


def test_engine_generation_length():
    torch.manual_seed(42)
    vocab_size = 50
    target_model = SimpleCausalLM(vocab_size=vocab_size, hidden_dim=64, num_layers=2, num_heads=2)
    draft_model = SimpleCausalLM(vocab_size=vocab_size, hidden_dim=32, num_layers=1, num_heads=2)

    engine = SpeculativeDecodingEngine(target_model, draft_model)

    prompt = [1, 2, 3]
    max_new_tokens = 16

    gen_tokens, telemetry = engine.generate(prompt, max_new_tokens=max_new_tokens, gamma=3)

    assert len(gen_tokens) == max_new_tokens
    assert telemetry.total_tokens_generated == max_new_tokens
    assert telemetry.target_forward_passes > 0
    assert telemetry.target_forward_passes <= max_new_tokens
    assert 0.0 <= telemetry.acceptance_rate <= 1.0


def test_standard_vs_speculative_consistency():
    torch.manual_seed(42)
    vocab_size = 50
    target = SimpleCausalLM(vocab_size=vocab_size, hidden_dim=64, num_layers=2, num_heads=2)
    draft = SimpleCausalLM(vocab_size=vocab_size, hidden_dim=32, num_layers=1, num_heads=2)

    engine = SpeculativeDecodingEngine(target, draft)
    prompt = [10, 20]

    out_std, tel_std = engine.generate_standard(prompt, max_new_tokens=10)
    out_spec, tel_spec = engine.generate(prompt, max_new_tokens=10, gamma=3)

    assert len(out_std) == 10
    assert len(out_spec) == 10
    # Speculative decoding takes fewer target model forward passes!
    assert tel_spec.target_forward_passes < tel_std.target_forward_passes
