import pytest
import torch
from speculative_decoding.sampler import modified_rejection_sampling, sample_from_logits


def test_sample_from_logits_greedy():
    logits = torch.tensor([1.0, 5.0, 2.0])
    tok, probs = sample_from_logits(logits, temperature=0.0)
    assert tok == 1
    assert probs[1].item() == 1.0


def test_sample_from_logits_temperature():
    logits = torch.tensor([1.0, 2.0, 3.0])
    tok, probs = sample_from_logits(logits, temperature=1.0)
    assert 0 <= tok < 3
    assert abs(torch.sum(probs).item() - 1.0) < 1e-5


def test_rejection_sampling_identical_distributions():
    """When target and draft have identical distributions, acceptance rate is near 100%."""
    torch.manual_seed(42)
    vocab_size = 10
    K = 3

    # Identical probabilities
    p = torch.softmax(torch.randn(vocab_size), dim=-1)
    draft_probs = [p.clone() for _ in range(K)]
    target_probs = [p.clone() for _ in range(K + 1)]

    # Draw draft tokens directly from p
    draft_tokens = [int(torch.multinomial(p, num_samples=1).item()) for _ in range(K)]

    # Run many trials to verify high acceptance rate
    total_proposed = 0
    total_accepted = 0
    for _ in range(100):
        accepted, num_acc = modified_rejection_sampling(draft_tokens, draft_probs, target_probs)
        total_proposed += K
        total_accepted += num_acc

    acceptance_rate = total_accepted / total_proposed
    # Under identical distributions, min(1, p/q) = 1.0, so 100% of tokens are accepted!
    assert acceptance_rate == 1.0, f"Expected 1.0 acceptance, got {acceptance_rate}"


def test_rejection_sampling_disjoint_distributions():
    """When target and draft distributions have disjoint support, tokens are rejected."""
    vocab_size = 4
    K = 1

    # Draft only predicts token 0
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # Target only predicts token 1, 2, or 3
    p = torch.tensor([0.0, 0.5, 0.5, 0.0])

    draft_tokens = [0]
    draft_probs = [q]
    target_probs = [p, p]

    accepted, num_acc = modified_rejection_sampling(draft_tokens, draft_probs, target_probs)

    # Draft token 0 must be rejected
    assert num_acc == 0
    # Replacement token must come from target support (1 or 2)
    assert accepted[0] in [1, 2]
