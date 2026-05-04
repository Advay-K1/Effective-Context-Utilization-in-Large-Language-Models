"""
Input×Gradient Ablation — Gradient-Based Attribution
====================================================
Implements gradient-based attribution for measuring effective context utilization.
This method computes the gradient of the loss with respect to input embeddings,
then uses input × gradient to measure token-level importance.

Key advantages over physical deletion:
  1. Single forward/backward pass (O(1) vs O(m+1))
  2. No positional encoding shifts
  3. Differentiable importance scores
  4. Token-level granularity that can be aggregated to chunks

Key differences from attention masking:
  - Uses gradients rather than counterfactual generation
  - Much faster (1 pass vs m passes)
  - Provides smooth importance scores rather than discrete ablations
  - Based on actual gradient flow through the network

Usage:
    python inputxgrad_ablation.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --input_file sample_prompts.jsonl \
        --chunk_size 512 \
        --max_new_tokens 256 \
        --output_dir results_inputxgrad/

Input JSONL format (one object per line):
    {"prompt": "...", "reference": "optional gold answer", "task": "qa"}

Requirements:
    pip install torch transformers sentence-transformers accelerate
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkInfluence:
    chunk_index: int
    start_token: int
    end_token: int
    gradient_norm_importance: float = 0.0      # ||grad|| aggregated over chunk
    inputxgrad_importance: float = 0.0         # (input * grad).norm() aggregated
    gradient_sum_importance: float = 0.0       # gradient sum aggregated
    max_token_importance: float = 0.0          # max token importance in chunk


@dataclass
class ProfileStats:
    """Timing and memory stats for gradient computation."""
    wall_time_s: float = 0.0
    peak_vram_mb: float = 0.0
    forward_time_s: float = 0.0
    backward_time_s: float = 0.0


@dataclass
class ExampleResult:
    prompt_tokens: int = 0
    num_chunks: int = 0
    chunk_size: int = 0
    baseline_text: str = ""
    chunk_influences: list = field(default_factory=list)
    token_importances: list = field(default_factory=list)  # per-token scores
    attribution_method: str = "inputxgrad"  # which variant was used
    eucr: dict = field(default_factory=dict)       # threshold -> value
    pwup: dict = field(default_factory=dict)        # {"B": .., "M": .., "E": ..}
    gud: float = 0.0
    profile: dict = field(default_factory=dict)     # ProfileStats dict


# ---------------------------------------------------------------------------
# Input×Gradient Attribution Methods
# ---------------------------------------------------------------------------

def compute_inputxgrad_attribution(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    method: str = "inputxgrad",
) -> tuple[torch.Tensor, torch.Tensor, ProfileStats]:
    """
    Compute gradient-based attribution for input tokens.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        input_ids: (1, seq_len) tensor
        attention_mask: (1, seq_len) tensor
        max_new_tokens: Number of tokens to generate
        method: Attribution method ('inputxgrad', 'gradient_norm', 'gradient_sum')
    
    Returns:
        output_ids: (1, seq_len + gen_len) generated sequence
        importance_scores: (seq_len,) tensor of token-level importance
        profile_stats: Timing and memory statistics
    """
    device = next(model.parameters()).device
    seq_len = input_ids.shape[1]
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    
    profile = ProfileStats()
    t0 = time.perf_counter()
    
    # Forward pass: generate baseline text
    t_forward_start = time.perf_counter()
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    t_forward_end = time.perf_counter()
    profile.forward_time_s = t_forward_end - t_forward_start
    
    # Extract generated tokens
    generated_ids = baseline_output[0, seq_len:]
    
    # Create full sequence (prompt + generation) for gradient computation
    full_ids = baseline_output  # (1, seq_len + gen_len)
    full_len = full_ids.shape[1]
    
    # Get embeddings with gradients enabled
    embed_layer = model.get_input_embeddings()
    
    # Only compute gradients w.r.t. input prompt embeddings
    prompt_embeds = embed_layer(input_ids).requires_grad_(True)
    gen_embeds = embed_layer(generated_ids).detach()  # No gradients for generated tokens
    
    # Concatenate embeddings
    full_embeds = torch.cat([prompt_embeds, gen_embeds], dim=1)
    full_mask = torch.cat([
        attention_mask,
        torch.ones(1, generated_ids.shape[0], dtype=torch.long, device=device)
    ], dim=1)
    
    # Forward pass with embeddings
    t_backward_start = time.perf_counter()
    outputs = model(inputs_embeds=full_embeds, attention_mask=full_mask)
    logits = outputs.logits  # (1, full_len, vocab_size)
    
    # Compute loss on generated tokens (next-token prediction)
    # Shift logits and labels for causal LM loss
    shift_logits = logits[0, seq_len-1:-1, :]  # (gen_len, vocab_size)
    shift_labels = generated_ids  # (gen_len,)
    
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    
    # Backward pass
    loss.backward()
    t_backward_end = time.perf_counter()
    profile.backward_time_s = t_backward_end - t_backward_start
    
    # Compute importance scores based on method
    with torch.no_grad():
        prompt_grads = prompt_embeds.grad  # (1, seq_len, embed_dim)
        prompt_inputs = prompt_embeds      # (1, seq_len, embed_dim)
        
        if method == "inputxgrad":
            # Element-wise multiplication then norm
            importance = (prompt_grads[0] * prompt_inputs[0]).norm(dim=-1)
        elif method == "gradient_norm":
            # Just gradient norm
            importance = prompt_grads[0].norm(dim=-1)
        elif method == "gradient_sum":
            # Sum of gradients (signed)
            importance = prompt_grads[0].sum(dim=-1).abs()
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        importance = importance.cpu().float()  # (seq_len,)
    
    t1 = time.perf_counter()
    profile.wall_time_s = t1 - t0
    
    if device.type == "cuda":
        profile.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    return baseline_output, importance, profile


# ---------------------------------------------------------------------------
# Chunking and aggregation
# ---------------------------------------------------------------------------

def aggregate_importance_to_chunks(
    importance_scores: torch.Tensor,
    chunk_size: int,
    aggregation_methods: list[str] = ["mean", "max", "sum"]
) -> list[dict]:
    """
    Aggregate token-level importance scores to chunk-level scores.
    
    Args:
        importance_scores: (seq_len,) tensor of token-level scores
        chunk_size: Size of each chunk in tokens
        aggregation_methods: List of methods to use for aggregation
    
    Returns:
        List of chunk importance dictionaries
    """
    seq_len = len(importance_scores)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    chunk_influences = []
    for chunk_idx in range(num_chunks):
        start_token = chunk_idx * chunk_size
        end_token = min(start_token + chunk_size, seq_len)
        
        chunk_scores = importance_scores[start_token:end_token]
        
        chunk_influence = ChunkInfluence(
            chunk_index=chunk_idx,
            start_token=start_token,
            end_token=end_token,
        )
        
        if "mean" in aggregation_methods:
            chunk_influence.inputxgrad_importance = float(chunk_scores.mean())
        if "max" in aggregation_methods:
            chunk_influence.max_token_importance = float(chunk_scores.max())
        if "sum" in aggregation_methods:
            chunk_influence.gradient_sum_importance = float(chunk_scores.sum())
            
        chunk_influences.append(chunk_influence)
    
    return chunk_influences


# ---------------------------------------------------------------------------
# Summary metrics (same as other methods)
# ---------------------------------------------------------------------------

def compute_eucr(influence_scores: list[float], thresholds: list[float]) -> dict[float, float]:
    """EUCR[λ] = (1/m) * Σ 1[Δ_i > λ]"""
    m = len(influence_scores)
    return {
        lam: round(sum(1 for d in influence_scores if d > lam) / m, 4)
        for lam in thresholds
    }


def compute_pwup(influence_scores: list[float]) -> dict[str, float]:
    """
    Split chunks into beginning / middle / end thirds.
    PWUP = (U_B, U_M, U_E) where U_R = Σ_{i in R} Δ_i^N
    """
    m = len(influence_scores)
    total = sum(influence_scores)
    if total == 0:
        return {"B": 0.0, "M": 0.0, "E": 0.0}

    normalized = [d / total for d in influence_scores]
    third = m // 3
    remainder = m % 3

    b_end = third
    m_end = third + third + (1 if remainder >= 1 else 0)

    u_b = sum(normalized[:b_end])
    u_m = sum(normalized[b_end:m_end])
    u_e = sum(normalized[m_end:])

    return {
        "B": round(u_b, 4),
        "M": round(u_m, 4),
        "E": round(u_e, 4),
    }


def compute_gud(stage_influence_matrix: list[list[float]]) -> float:
    """
    GUD = (1/(S-1)) * Σ_{s=1}^{S-1} (1/2) * Σ_i |Δ_i^(s) - Δ_i^(s+1)|
    For gradient methods, we use a simplified version since we don't have stages.
    """
    # For gradient methods, GUD doesn't apply in the same way since we have
    # a single attribution pass. Return 0.0 as placeholder.
    return 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_inputxgrad_pipeline(
    model,
    tokenizer,
    prompt: str,
    chunk_size: int,
    max_new_tokens: int,
    eucr_thresholds: list[float],
    attribution_method: str = "inputxgrad",
) -> ExampleResult:
    device = next(model.parameters()).device
    result = ExampleResult(
        chunk_size=chunk_size,
        attribution_method=attribution_method
    )

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    result.prompt_tokens = input_ids.shape[1]

    print(f"  Prompt: {result.prompt_tokens} tokens")
    print(f"  Attribution method: {attribution_method}")

    # Compute attribution
    baseline_output, importance_scores, profile_stats = compute_inputxgrad_attribution(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        method=attribution_method,
    )

    # Decode baseline text
    baseline_text = tokenizer.decode(
        baseline_output[0, result.prompt_tokens:], skip_special_tokens=True
    )
    result.baseline_text = baseline_text

    # Aggregate to chunks
    chunk_influences = aggregate_importance_to_chunks(
        importance_scores, chunk_size, aggregation_methods=["mean", "max", "sum"]
    )
    result.num_chunks = len(chunk_influences)
    result.chunk_influences = [asdict(ci) for ci in chunk_influences]
    result.token_importances = importance_scores.tolist()

    print(f"  Generated {len(baseline_text.split())} words in {profile_stats.wall_time_s:.2f}s")
    print(f"  Forward: {profile_stats.forward_time_s:.2f}s, Backward: {profile_stats.backward_time_s:.2f}s")
    print(f"  Peak VRAM: {profile_stats.peak_vram_mb:.0f}MB")

    # Summary metrics using the primary importance scores
    primary_scores = [ci["inputxgrad_importance"] for ci in result.chunk_influences]
    result.eucr = compute_eucr(primary_scores, eucr_thresholds)
    result.pwup = compute_pwup(primary_scores)
    result.gud = compute_gud([])  # Not applicable for gradient methods

    result.profile = asdict(profile_stats)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Input×Gradient attribution ablation")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSONL file with prompts")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Chunk size in tokens (default: 512)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--output_dir", type=str, default="results_inputxgrad",
                        help="Output directory for results")
    parser.add_argument("--eucr_thresholds", type=float, nargs="+",
                        default=[0.01, 0.05, 0.10, 0.20],
                        help="EUCR thresholds to evaluate")
    parser.add_argument("--attribution_method", type=str,
                        choices=["inputxgrad", "gradient_norm", "gradient_sum"],
                        default="inputxgrad",
                        help="Attribution method to use")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu'")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (default: bfloat16)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    print(f"Loading model: {args.model} on {device} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=device if device == "auto" else {"": device},
        trust_remote_code=True,
    )
    model.eval()

    # Load prompts
    prompts = []
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    # Run attribution
    all_results = []
    for idx, entry in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Example {idx + 1}/{len(prompts)}")
        print(f"{'='*60}")

        result = run_inputxgrad_pipeline(
            model=model,
            tokenizer=tokenizer,
            prompt=entry["prompt"],
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            eucr_thresholds=args.eucr_thresholds,
            attribution_method=args.attribution_method,
        )

        # Print summary
        print(f"\n  --- Summary ---")
        print(f"  EUCR: {result.eucr}")
        print(f"  PWUP: {result.pwup}")
        print(f"  GUD:  {result.gud}")
        print(f"  Wall time: {result.profile['wall_time_s']:.2f}s")
        print(f"  Peak VRAM: {result.profile['peak_vram_mb']:.0f}MB")

        all_results.append(asdict(result))

    # Save results
    out_path = Path(args.output_dir) / "inputxgrad_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save profiling summary (compatible with benchmark scripts)
    profile_path = Path(args.output_dir) / "profiling_summary.json"
    profile_summary = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "attribution_method": args.attribution_method,
        "examples": [],
    }
    for idx, r in enumerate(all_results):
        profile_summary["examples"].append({
            "index": idx,
            "prompt_tokens": r["prompt_tokens"],
            "num_chunks": r["num_chunks"],
            "total_ablation_runs": 1,  # Only one forward+backward pass
            "total_wall_time_s": round(r["profile"]["wall_time_s"], 2),
            "mean_wall_time_s": round(r["profile"]["wall_time_s"], 4),
            "max_peak_vram_mb": round(r["profile"]["peak_vram_mb"], 2),
            "forward_time_s": round(r["profile"]["forward_time_s"], 4),
            "backward_time_s": round(r["profile"]["backward_time_s"], 4),
        })
    with open(profile_path, "w") as f:
        json.dump(profile_summary, f, indent=2)
    print(f"Profiling summary saved to {profile_path}")


if __name__ == "__main__":
    main()