import torch
from typing import List, Tuple


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> Tuple[int, torch.Tensor]:
    """
    Sample a single token from a 1D logit vector [vocab_size].
    Returns (token_id, probabilities).
    """
    if temperature <= 0.0 or temperature < 1e-5:
        # Greedy argmax
        token_id = int(torch.argmax(logits).item())
        probs = torch.zeros_like(logits)
        probs[token_id] = 1.0
        return token_id, probs

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits[indices_to_remove] = -float("Inf")
        probs = torch.softmax(scaled_logits, dim=-1)

    token_id = int(torch.multinomial(probs, num_samples=1).item())
    return token_id, probs


def modified_rejection_sampling(
    draft_tokens: List[int],
    draft_probs_seq: List[torch.Tensor],
    target_probs_seq: List[torch.Tensor]
) -> Tuple[List[int], int]:
    """
    Leviathan et al. Speculative Rejection Sampling.

    Mathematically guarantees that the accepted tokens follow the EXACT distribution
    of the target model p(x), introducing zero distribution shift.

    Args:
        draft_tokens: List of K speculative tokens proposed by the draft model.
        draft_probs_seq: List of K probability vectors from the draft model q(x_i).
        target_probs_seq: List of K+1 probability vectors from the target model p(x_i).

    Returns:
        accepted_tokens: List of accepted tokens plus 1 replacement/bonus token.
        num_speculative_accepted: Number of speculative draft tokens accepted (0 to K).
    """
    K = len(draft_tokens)
    assert len(draft_probs_seq) == K
    assert len(target_probs_seq) == K + 1

    accepted_tokens: List[int] = []
    num_speculative_accepted = 0

    for i in range(K):
        tok = draft_tokens[i]
        q_val = draft_probs_seq[i][tok].item()
        p_val = target_probs_seq[i][tok].item()

        # Rejection threshold: r ~ U(0, 1)
        r = torch.rand(1).item()
        acceptance_prob = min(1.0, p_val / max(q_val, 1e-12))

        if r < acceptance_prob:
            # Accept draft token
            accepted_tokens.append(tok)
            num_speculative_accepted += 1
        else:
            # Reject draft token! Sample replacement token from the residual distribution:
            # p_res(x) = max(0, p(x) - q(x)) / sum(max(0, p(y) - q(y)))
            p = target_probs_seq[i]
            q = draft_probs_seq[i]
            residual = torch.clamp(p - q, min=0.0)
            residual_sum = torch.sum(residual).item()

            if residual_sum > 1e-12:
                res_probs = residual / residual_sum
                replacement_token = int(torch.multinomial(res_probs, num_samples=1).item())
            else:
                replacement_token = int(torch.multinomial(p, num_samples=1).item())

            accepted_tokens.append(replacement_token)
            # Once a token is rejected, we discard all subsequent speculative tokens
            return accepted_tokens, num_speculative_accepted

    # If all K speculative tokens were accepted, sample a bonus (K+1)-th token from target distribution
    bonus_probs = target_probs_seq[K]
    bonus_token = int(torch.multinomial(bonus_probs, num_samples=1).item())
    accepted_tokens.append(bonus_token)

    return accepted_tokens, num_speculative_accepted
