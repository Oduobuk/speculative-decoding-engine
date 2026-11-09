import time
import torch
import torch.nn as nn
from typing import List, Tuple
from .sampler import sample_from_logits, modified_rejection_sampling
from .telemetry import GenerationTelemetry


class SpeculativeDecodingEngine:
    """
    Engine executing Speculative Decoding with exact distribution preservation.
    
    Pairs a large target model with a small draft model:
    - Draft proposes K tokens autoregressively.
    - Target validates all K tokens in a single parallel forward pass.
    - Modified rejection sampling accepts or replaces tokens with zero quality loss.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        device: torch.device = torch.device("cpu")
    ):
        self.target_model = target_model.to(device).eval()
        self.draft_model = draft_model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        gamma: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> Tuple[List[int], GenerationTelemetry]:
        """
        Speculative decoding generation loop.
        """
        telemetry = GenerationTelemetry()
        t0 = time.perf_counter()

        current_tokens = list(prompt_tokens)
        num_generated = 0

        while num_generated < max_new_tokens:
            K = min(gamma, max_new_tokens - num_generated)
            if K <= 0:
                break

            # -------------------------------------------------------------
            # Step 1: Draft model generates K candidate tokens
            # -------------------------------------------------------------
            draft_tokens = []
            draft_probs = []
            draft_context = list(current_tokens)

            for _ in range(K):
                input_tensor = torch.tensor([draft_context], dtype=torch.long, device=self.device)
                logits = self.draft_model(input_tensor)[0, -1, :]
                telemetry.draft_forward_passes += 1

                tok, probs = sample_from_logits(logits, temperature=temperature, top_p=top_p)
                draft_tokens.append(tok)
                draft_probs.append(probs)
                draft_context.append(tok)

            telemetry.speculative_tokens_proposed += K

            # -------------------------------------------------------------
            # Step 2: Target model parallel verification forward pass
            # Evaluates all K draft positions simultaneously!
            # -------------------------------------------------------------
            eval_tokens = current_tokens + draft_tokens
            target_input = torch.tensor([eval_tokens], dtype=torch.long, device=self.device)
            target_logits_all = self.target_model(target_input)[0]  # [len(eval_tokens), vocab_size]
            telemetry.target_forward_passes += 1

            # Extract target probability distribution for positions:
            # from (len(current_tokens) - 1) to (len(current_tokens) + K - 1)
            # giving K + 1 distributions (K draft positions + 1 bonus token position)
            start_idx = len(current_tokens) - 1
            target_probs_seq = []
            for idx in range(start_idx, start_idx + K + 1):
                logits_pos = target_logits_all[idx]
                _, p = sample_from_logits(logits_pos, temperature=temperature, top_p=top_p)
                target_probs_seq.append(p)

            # -------------------------------------------------------------
            # Step 3: Rejection Sampling Verification
            # -------------------------------------------------------------
            accepted_tokens, num_accepted = modified_rejection_sampling(
                draft_tokens=draft_tokens,
                draft_probs_seq=draft_probs,
                target_probs_seq=target_probs_seq
            )

            telemetry.speculative_tokens_accepted += num_accepted

            # Trim to max_new_tokens if needed
            allowed = max_new_tokens - num_generated
            tokens_to_add = accepted_tokens[:allowed]

            current_tokens.extend(tokens_to_add)
            num_generated += len(tokens_to_add)

        telemetry.total_tokens_generated = num_generated
        telemetry.latency_seconds = time.perf_counter() - t0

        generated_only = current_tokens[len(prompt_tokens):]
        return generated_only, telemetry

    @torch.no_grad()
    def generate_standard(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> Tuple[List[int], GenerationTelemetry]:
        """
        Baseline standard autoregressive decoding using the Target Model alone.
        """
        telemetry = GenerationTelemetry()
        t0 = time.perf_counter()

        current_tokens = list(prompt_tokens)

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
            logits = self.target_model(input_tensor)[0, -1, :]
            telemetry.target_forward_passes += 1

            tok, _ = sample_from_logits(logits, temperature=temperature, top_p=top_p)
            current_tokens.append(tok)

        telemetry.total_tokens_generated = max_new_tokens
        telemetry.latency_seconds = time.perf_counter() - t0

        generated_only = current_tokens[len(prompt_tokens):]
        return generated_only, telemetry
