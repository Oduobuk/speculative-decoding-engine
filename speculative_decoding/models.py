import torch
import torch.nn as nn
from typing import Optional


class SimpleCausalLM(nn.Module):
    """
    Decoder-only Causal Transformer Language Model.
    Used for benchmarking speculative decoding acceleration and rejection sampling.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs.
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)

        # Causal mask: upper triangular elements set to -inf
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)

        h = self.transformer(x, mask=causal_mask, is_causal=True)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
