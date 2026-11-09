from dataclasses import dataclass


@dataclass
class GenerationTelemetry:
    """
    Telemetry and profiling metrics for Speculative Decoding.
    """
    total_tokens_generated: int = 0
    target_forward_passes: int = 0
    draft_forward_passes: int = 0
    speculative_tokens_proposed: int = 0
    speculative_tokens_accepted: int = 0
    latency_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed speculative tokens that passed rejection sampling (alpha)."""
        if self.speculative_tokens_proposed == 0:
            return 0.0
        return self.speculative_tokens_accepted / self.speculative_tokens_proposed

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput in tokens per second (TPS)."""
        if self.latency_seconds == 0:
            return 0.0
        return self.total_tokens_generated / self.latency_seconds

    @property
    def tokens_per_target_pass(self) -> float:
        """Average number of tokens emitted per costly target model forward pass."""
        if self.target_forward_passes == 0:
            return 0.0
        return self.total_tokens_generated / self.target_forward_passes

