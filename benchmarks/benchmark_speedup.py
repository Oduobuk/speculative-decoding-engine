import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=(), tablefmt="github"):
        headers = list(headers)
        rows = [[str(cell) for cell in row] for row in data]
        col_widths = [len(h) for h in headers] if headers else [0] * (len(rows[0]) if rows else 0)
        for row in rows:
            for i, cell in enumerate(row):
                if i >= len(col_widths):
                    col_widths.append(len(cell))
                else:
                    col_widths[i] = max(col_widths[i], len(cell))
        lines = []
        if headers:
            lines.append("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |")
            lines.append("|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|")
        for row in rows:
            lines.append("| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")
        return "\n".join(lines)
from speculative_decoding.models import SimpleCausalLM
from speculative_decoding.engine import SpeculativeDecodingEngine


def run_speculative_speedup_benchmark():
    print("=" * 75)
    print("      SPECULATIVE DECODING VS STANDARD AUTOREGRESSIVE GENERATION")
    print("=" * 75)

    torch.manual_seed(42)

    vocab_size = 2000
    max_new_tokens = 64
    prompt = [10, 45, 92, 103, 5]

    # Target Model: Larger capacity (simulates Llama-3-70B or 8B)
    target_model = SimpleCausalLM(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        max_seq_len=512
    )

    # Draft Model: 10x smaller footprint (simulates 1B draft or EAGLE head)
    draft_model = SimpleCausalLM(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=512
    )

    engine = SpeculativeDecodingEngine(target_model, draft_model)

    print(f"\nModel Configuration:")
    print(f"• Target Model Parameters: {target_model.count_parameters():,}")
    print(f"• Draft Model Parameters:  {draft_model.count_parameters():,} ({target_model.count_parameters() / draft_model.count_parameters():.1f}x smaller)")
    print(f"• Target Generation Tokens: {max_new_tokens}\n")

    # 1. Benchmark Standard Autoregressive Generation
    print("Running Baseline Standard Autoregressive Generation...")
    # Warmup
    _, _ = engine.generate_standard(prompt, max_new_tokens=8)
    _, tel_std = engine.generate_standard(prompt, max_new_tokens=max_new_tokens)

    # 2. Benchmark Speculative Decoding at Gamma = 3 and Gamma = 5
    results = [
        [
            "Standard Autoregressive (Baseline)",
            "N/A",
            f"{tel_std.latency_seconds:.3f} s",
            f"{tel_std.tokens_per_second:.1f} TPS",
            f"{tel_std.target_forward_passes}",
            "N/A",
            "1.00x (Baseline)"
        ]
    ]

    for gamma in [3, 5]:
        print(f"Running Speculative Decoding (Gamma={gamma})...")
        # Warmup
        _, _ = engine.generate(prompt, max_new_tokens=8, gamma=gamma)
        _, tel_spec = engine.generate(prompt, max_new_tokens=max_new_tokens, gamma=gamma)

        speedup = tel_std.latency_seconds / tel_spec.latency_seconds
        results.append([
            f"Speculative Decoding (K={gamma})",
            f"{gamma}",
            f"{tel_spec.latency_seconds:.3f} s",
            f"{tel_spec.tokens_per_second:.1f} TPS",
            f"{tel_spec.target_forward_passes} ({tel_std.target_forward_passes / tel_spec.target_forward_passes:.1f}x fewer)",
            f"{tel_spec.acceptance_rate * 100:.1f}%",
            f"{speedup:.2f}x Speedup"
        ])

    headers = [
        "Inference Strategy", "Speculative Window (K)", "Latency", 
        "Throughput (TPS)", "Target Passes", "Acceptance Rate (alpha)", "Wall-Clock Speedup"
    ]
    print("\n" + tabulate(results, headers=headers, tablefmt="github"))

    print("\n[KEY TAKEAWAY FOR INTERVIEWS & EVALUATION]:")
    print("Speculative decoding speeds up inference by reducing costly Target Model forward passes,")
    print("substituting them with cheap Draft passes and a single parallel verification step.")
    print("Through modified rejection sampling, the output distribution is mathematically identical")
    print("to sampling from the target model directly (zero quality penalty).")


if __name__ == "__main__":
    run_speculative_speedup_benchmark()

