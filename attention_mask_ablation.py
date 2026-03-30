"""
Attention Mask Ablation
=======================
Implements the mask-based ablation framework from the CS 498 proposal:
  1. Generate a baseline output from the full prompt.
  2. For each chunk C_i, zero out its positions in the attention mask
     and regenerate under the same decoding settings.
  3. Compute chunk-level influence scores and summary metrics
     (EUCR, PWUP, GUD) — identical definitions to the baseline.
  4. Profile wall-clock time and peak GPU memory per ablation.

Key difference from chunk_deletion_baseline.py:
  Tokens are never removed from the sequence. The full-length input_ids
  are reused for every ablation run; only the attention_mask changes.
  This preserves RoPE positional encodings for all unchanged tokens,
  eliminating the position-shift artifact that contaminates the baseline.
  It also means every forward pass (baseline and all ablations) operates
  on a sequence of identical length, making VRAM and timing numbers
  directly comparable across runs.

Usage:
    python attention_mask_ablation.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input_file sample_prompts.jsonl \\
        --chunk_size 512 \\
        --max_new_tokens 256 \\
        --output_dir results_mask/

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
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Data classes  (identical to baseline for output compatibility)
# ---------------------------------------------------------------------------

@dataclass
class ChunkInfluence:
    chunk_index: int
    start_token: int
    end_token: int
    semantic_similarity: float = 0.0
    influence_score: float = 0.0
    logprob_drop: float = 0.0


@dataclass
class ProfileStats:
    """Timing and memory stats for a single ablation run."""
    chunk_index: int  # -1 for baseline
    wall_time_s: float = 0.0
    peak_vram_mb: float = 0.0


@dataclass
class ExampleResult:
    prompt_tokens: int = 0
    num_chunks: int = 0
    chunk_size: int = 0
    baseline_text: str = ""
    ablated_texts: list = field(default_factory=list)
    chunk_influences: list = field(default_factory=list)
    eucr: dict = field(default_factory=dict)
    pwup: dict = field(default_factory=dict)
    gud: float = 0.0
    profile: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def chunk_token_ids(token_ids: list[int], chunk_size: int) -> list[list[int]]:
    """Split token_ids into contiguous chunks of chunk_size tokens.
    The last chunk may be shorter."""
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunks.append(token_ids[i : i + chunk_size])
    return chunks


def build_chunk_attention_mask(
    seq_len: int,
    chunk_start: int,
    chunk_end: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a (1, seq_len) long tensor of ones with zeros at [chunk_start, chunk_end).

    Setting these positions to 0 tells the model to treat those tokens as
    padding: no other token can attend to them as keys, so their content
    cannot directly influence the output. Unlike physical deletion, this
    leaves the token sequence — and therefore RoPE positional encodings —
    completely unchanged.
    """
    mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
    mask[0, chunk_start:chunk_end] = 0
    return mask


# ---------------------------------------------------------------------------
# Generation + profiling
# ---------------------------------------------------------------------------

def generate_with_profile(
    model,
    input_ids: torch.Tensor,
    generation_kwargs: dict,
    chunk_index: int = -1,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ProfileStats]:
    """Run generation and record wall-clock time and peak VRAM.

    attention_mask — if provided, passed to model.generate() so the model
    ignores the masked chunk positions. HuggingFace automatically extends
    the mask with 1s for each newly generated token.
    """
    device = next(model.parameters()).device

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    kwargs = dict(generation_kwargs)
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(input_ids, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    stats = ProfileStats(
        chunk_index=chunk_index,
        wall_time_s=round(t1 - t0, 4),
        peak_vram_mb=round(peak_vram_mb, 2),
    )
    return output_ids, stats


# ---------------------------------------------------------------------------
# Influence scoring
# ---------------------------------------------------------------------------

def compute_semantic_similarity(text_a: str, text_b: str, sim_model) -> float:
    """Cosine similarity between sentence embeddings."""
    embeddings = sim_model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def compute_logprob_drop(
    model,
    input_ids: torch.Tensor,
    baseline_output_ids: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
    device: torch.device,
) -> float:
    """Compute average log-probability drop of the baseline continuation
    when the target chunk is masked in the prompt.

    Both the baseline and ablated forward passes use sequences of the same
    length (prompt + output tokens). The ablated pass zeroes out the chunk
    positions in the prompt portion of the attention mask; the output
    positions remain unmasked so the model can attend to its own prior
    generated tokens.

    Returns drop = mean(log p_full - log p_masked), positive = chunk helps.
    """
    prompt_len = input_ids.shape[1]
    output_only = baseline_output_ids[0, prompt_len:]          # (output_len,)
    combined = torch.cat(
        [input_ids[0], output_only], dim=0
    ).unsqueeze(0).to(device)                                   # (1, prompt+output)
    combined_len = combined.shape[1]

    # ---- Ablated forward: mask the chunk in the prompt portion ----
    ablated_mask = torch.ones(1, combined_len, dtype=torch.long, device=device)
    ablated_mask[0, chunk_start:chunk_end] = 0
    with torch.no_grad():
        ablated_logits = model(combined, attention_mask=ablated_mask).logits

    # Logits at position t predict token t+1; extract output slice
    output_logits_ablated = ablated_logits[0, prompt_len - 1 : -1, :]
    log_probs_ablated = torch.log_softmax(output_logits_ablated, dim=-1)
    target_tokens = output_only.to(device)
    lp_ablated = log_probs_ablated.gather(
        1, target_tokens.unsqueeze(1)
    ).squeeze(1)

    # ---- Baseline forward: full attention (all-ones mask) ----
    with torch.no_grad():
        full_logits = model(combined).logits

    output_logits_full = full_logits[0, prompt_len - 1 : -1, :]
    log_probs_full = torch.log_softmax(output_logits_full, dim=-1)
    lp_full = log_probs_full.gather(
        1, target_tokens.unsqueeze(1)
    ).squeeze(1)

    drop = (lp_full - lp_ablated).mean().item()
    return round(drop, 6)


# ---------------------------------------------------------------------------
# Summary metrics  (identical to baseline)
# ---------------------------------------------------------------------------

def compute_eucr(influence_scores: list[float], thresholds: list[float]) -> dict[float, float]:
    """EUCR[λ] = (1/m) * Σ 1[Δ_i > λ]"""
    m = len(influence_scores)
    return {
        lam: round(sum(1 for d in influence_scores if d > lam) / m, 4)
        for lam in thresholds
    }


def compute_pwup(influence_scores: list[float]) -> dict[str, float]:
    """PWUP = (U_B, U_M, U_E) — normalised influence by position third."""
    m = len(influence_scores)
    total = sum(influence_scores)
    if total == 0:
        return {"B": 0.0, "M": 0.0, "E": 0.0}

    normalized = [d / total for d in influence_scores]
    third = m // 3
    remainder = m % 3

    b_end = third
    m_end = third + third + (1 if remainder >= 1 else 0)

    return {
        "B": round(sum(normalized[:b_end]), 4),
        "M": round(sum(normalized[b_end:m_end]), 4),
        "E": round(sum(normalized[m_end:]), 4),
    }


def compute_gud(stage_influence_matrix: list[list[float]]) -> float:
    """GUD = (1/(S-1)) * Σ_{s} (1/2) * Σ_i |Δ_i^(s) - Δ_i^(s+1)|"""
    S = len(stage_influence_matrix)
    if S <= 1:
        return 0.0

    total = 0.0
    for s in range(S - 1):
        diff = sum(
            abs(stage_influence_matrix[s][i] - stage_influence_matrix[s + 1][i])
            for i in range(len(stage_influence_matrix[s]))
        )
        total += 0.5 * diff

    return round(total / (S - 1), 6)


def compute_stage_influences(
    baseline_text: str,
    ablated_text: str,
    num_stages: int,
    sim_model,
) -> list[float]:
    """Per-stage influence (1 - cosine_sim) for one chunk, split into num_stages."""
    def split_text(text, n):
        words = text.split()
        if not words:
            return [""] * n
        k = max(1, len(words) // n)
        parts = [" ".join(words[i * k : (i + 1) * k]) for i in range(n - 1)]
        parts.append(" ".join(words[(n - 1) * k :]))
        return parts

    baseline_parts = split_text(baseline_text, num_stages)
    ablated_parts  = split_text(ablated_text,  num_stages)

    scores = []
    for bp, ap in zip(baseline_parts, ablated_parts):
        if not bp.strip() or not ap.strip():
            scores.append(0.0)
        else:
            sim = compute_semantic_similarity(bp, ap, sim_model)
            scores.append(max(0.0, 1.0 - sim))
    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ablation_pipeline(
    model,
    tokenizer,
    sim_model,
    prompt: str,
    chunk_size: int,
    max_new_tokens: int,
    eucr_thresholds: list[float],
    num_stages: int = 3,
    compute_logprobs: bool = False,
) -> ExampleResult:
    device = next(model.parameters()).device
    result = ExampleResult(chunk_size=chunk_size)

    # Tokenize once; reused for every ablation run
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    result.prompt_tokens = input_ids.shape[1]
    seq_len = result.prompt_tokens

    # Chunk boundaries (token indices, not a separate list of token ids)
    chunks = chunk_token_ids(input_ids[0].tolist(), chunk_size)
    result.num_chunks = len(chunks)
    print(
        f"  Prompt: {seq_len} tokens -> {result.num_chunks} chunks of {chunk_size} "
        f"(input length constant across all runs)"
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # ---- Baseline generation (full attention mask = all ones) ----
    print("  Generating baseline...")
    baseline_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
    baseline_output_ids, baseline_stats = generate_with_profile(
        model, input_ids, gen_kwargs,
        chunk_index=-1, attention_mask=baseline_mask,
    )
    baseline_text = tokenizer.decode(
        baseline_output_ids[0, seq_len:], skip_special_tokens=True
    )
    result.baseline_text = baseline_text
    result.profile.append(asdict(baseline_stats))

    # ---- Ablation runs ----
    all_influences = []
    stage_matrix = []

    for ci, chunk in enumerate(chunks):
        chunk_start = ci * chunk_size
        chunk_end   = chunk_start + len(chunk)   # handles shorter last chunk

        print(f"  Ablating chunk {ci + 1}/{result.num_chunks} "
              f"[{chunk_start}:{chunk_end}]...", end=" ")

        ablation_mask = build_chunk_attention_mask(
            seq_len, chunk_start, chunk_end, device
        )

        ablated_output_ids, ablation_stats = generate_with_profile(
            model, input_ids, gen_kwargs,
            chunk_index=ci, attention_mask=ablation_mask,
        )
        # Output slice starts at seq_len — identical for every run because
        # input_ids never changes length (key advantage over physical deletion)
        ablated_text = tokenizer.decode(
            ablated_output_ids[0, seq_len:], skip_special_tokens=True
        )
        result.ablated_texts.append(ablated_text)
        result.profile.append(asdict(ablation_stats))

        # Semantic influence
        sem_sim  = compute_semantic_similarity(baseline_text, ablated_text, sim_model)
        influence = max(0.0, 1.0 - sem_sim)

        # Optional log-prob drop (uses masked forward, same sequence length)
        lp_drop = 0.0
        if compute_logprobs:
            lp_drop = compute_logprob_drop(
                model, input_ids, baseline_output_ids,
                chunk_start, chunk_end, device,
            )

        all_influences.append(ChunkInfluence(
            chunk_index=ci,
            start_token=chunk_start,
            end_token=chunk_end,
            semantic_similarity=round(sem_sim, 6),
            influence_score=round(influence, 6),
            logprob_drop=lp_drop,
        ))

        stage_scores = compute_stage_influences(
            baseline_text, ablated_text, num_stages, sim_model
        )
        stage_matrix.append(stage_scores)

        print(
            f"influence={influence:.4f}  "
            f"time={ablation_stats.wall_time_s:.2f}s  "
            f"vram={ablation_stats.peak_vram_mb:.0f}MB"
        )

    result.chunk_influences = [asdict(ci) for ci in all_influences]

    # ---- Summary metrics ----
    scores = [ci.influence_score for ci in all_influences]
    result.eucr = compute_eucr(scores, eucr_thresholds)
    result.pwup  = compute_pwup(scores)

    if num_stages > 1 and stage_matrix:
        transposed = list(zip(*stage_matrix))   # S x m
        normalized = []
        for stage_row in transposed:
            total = sum(stage_row)
            if total > 0:
                normalized.append([v / total for v in stage_row])
            else:
                normalized.append(list(stage_row))
        result.gud = compute_gud(normalized)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Attention-mask chunk ablation (position-preserving)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--input_file", type=str, required=True,
                        help='JSONL file with prompts ({"prompt": ..., ...})')
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Chunk size in tokens (default: 512)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--output_dir", type=str, default="results_mask",
                        help="Output directory for results (default: results_mask)")
    parser.add_argument("--eucr_thresholds", type=float, nargs="+",
                        default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--compute_logprobs", action="store_true",
                        help="Also compute log-probability drop (slower)")
    parser.add_argument("--sim_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    print(f"Loading similarity model: {args.sim_model}")
    from sentence_transformers import SentenceTransformer
    sim_model = SentenceTransformer(args.sim_model, device=device)

    prompts = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    all_results = []
    for idx, entry in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Example {idx + 1}/{len(prompts)}")
        print(f"{'='*60}")

        result = run_ablation_pipeline(
            model=model,
            tokenizer=tokenizer,
            sim_model=sim_model,
            prompt=entry["prompt"],
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            eucr_thresholds=args.eucr_thresholds,
            num_stages=args.num_stages,
            compute_logprobs=args.compute_logprobs,
        )

        print(f"\n  --- Summary ---")
        print(f"  EUCR: {result.eucr}")
        print(f"  PWUP: {result.pwup}")
        print(f"  GUD:  {result.gud}")
        total_time = sum(p["wall_time_s"] for p in result.profile)
        print(f"  Total wall time: {total_time:.2f}s")
        print(f"  Avg time per ablation: {total_time / (result.num_chunks + 1):.2f}s")

        all_results.append(asdict(result))

    # Save per-example results
    out_path = Path(args.output_dir) / "mask_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save profiling summary — same schema as chunk_deletion_baseline.py so
    # run_benchmark.py can consume both without modification.
    # Extra field `input_tokens_per_run` documents that every run in this
    # method uses the full prompt length (unlike physical deletion where
    # ablated runs are shorter by chunk_size tokens).
    profile_path = Path(args.output_dir) / "profiling_summary.json"
    profile_summary = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "ablation_method": "attention_mask",
        "examples": [],
    }
    for idx, r in enumerate(all_results):
        times = [p["wall_time_s"] for p in r["profile"]]
        vrams = [p["peak_vram_mb"] for p in r["profile"]]
        profile_summary["examples"].append({
            "index": idx,
            "prompt_tokens": r["prompt_tokens"],
            "num_chunks": r["num_chunks"],
            "total_ablation_runs": r["num_chunks"] + 1,
            "input_tokens_per_run": r["prompt_tokens"],   # constant — no shifting
            "total_wall_time_s": round(sum(times), 2),
            "mean_wall_time_s": round(np.mean(times), 2),
            "max_peak_vram_mb": round(max(vrams), 2),
        })
    with open(profile_path, "w") as f:
        json.dump(profile_summary, f, indent=2)
    print(f"Profiling summary saved to {profile_path}")


if __name__ == "__main__":
    main()
