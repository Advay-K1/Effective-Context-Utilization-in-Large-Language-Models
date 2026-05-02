"""
Attention Mask Ablation — Batched
==================================
Implements the mask-based ablation framework from the CS 498 proposal.

Key engineering optimization over chunk_deletion_baseline.py
-------------------------------------------------------------
Physical deletion shifts RoPE positions for all subsequent tokens, requiring a
full long-context prefill from scratch for each ablation.  This script fixes
both problems:

  1. Position-preservation: input_ids never changes length.  Only the
     attention_mask is modified (zeros at the target chunk's positions), so RoPE
     positional encodings are identical across every run.

  2. Batched prefill: all B ablation masks are stacked into a single
     model.generate() call:
         batched_input_ids  — shape (B, seq_len)   input_ids.repeat(B, 1)
         batched_masks      — shape (B, seq_len)   one mask per chunk in batch

     The quadratic prefill cost is paid once per batch rather than once per
     chunk, reducing total forward passes from O(m+1) to O(ceil(m/B)+1).

     Wall time:  O(T_prefill + B*T_decode) per batch
     vs sequential: O(m*T_prefill + m*T_decode)

     VRAM scales with B — reduce --ablation_batch_size if you OOM.

Usage:
    python attention_mask_ablation.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input_file sample_prompts.jsonl \\
        --chunk_size 512 \\
        --max_new_tokens 256 \\
        --ablation_batch_size 0 \\
        --output_dir results_mask/

    --ablation_batch_size 0  → batch all chunks in one call (default, fastest)
    --ablation_batch_size 8  → process 8 chunks per call (lower VRAM)
    --ablation_batch_size 1  → fully sequential (same cost as baseline, for ablation)

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
# Data classes
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
class BatchProfileStats:
    """Timing and memory stats for one batched generation call.

    chunk_indices=[-1] for the baseline run.
    chunk_indices=[i, i+1, ...] for an ablation batch.
    """
    chunk_indices: list
    batch_size: int
    wall_time_s: float = 0.0
    peak_vram_mb: float = 0.0


@dataclass
class ExampleResult:
    prompt_tokens: int = 0
    num_chunks: int = 0
    chunk_size: int = 0
    ablation_batch_size: int = 0
    baseline_text: str = ""
    ablated_texts: list = field(default_factory=list)
    chunk_influences: list = field(default_factory=list)
    eucr: dict = field(default_factory=dict)
    pwup: dict = field(default_factory=dict)
    gud: float = 0.0
    profile: list = field(default_factory=list)   # list of BatchProfileStats dicts


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def chunk_token_ids(token_ids: list[int], chunk_size: int) -> list[list[int]]:
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunks.append(token_ids[i : i + chunk_size])
    return chunks


def build_ablation_masks(
    seq_len: int,
    chunk_boundaries: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """Return (m, seq_len) attention mask tensor.

    Row i has zeros at [chunk_boundaries[i][0], chunk_boundaries[i][1]) and
    ones everywhere else.  All rows are built in one vectorised operation.
    """
    m = len(chunk_boundaries)
    masks = torch.ones(m, seq_len, dtype=torch.long, device=device)
    for i, (start, end) in enumerate(chunk_boundaries):
        masks[i, start:end] = 0
    return masks


# ---------------------------------------------------------------------------
# Generation + profiling
# ---------------------------------------------------------------------------

def generate_with_profile(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_kwargs: dict,
    chunk_indices: list[int],
) -> tuple[torch.Tensor, BatchProfileStats]:
    """Run model.generate() on a (possibly batched) input and record stats.

    input_ids       — (B, seq_len)
    attention_mask  — (B, seq_len)
    chunk_indices   — which chunks are in this batch ([-1] for baseline)

    Returns output_ids of shape (B, seq_len + generated_len) and stats.
    HuggingFace automatically extends attention_mask with 1s for each new
    token, so the masked chunk positions remain ignored throughout decoding.
    """
    device = next(model.parameters()).device

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    kwargs = dict(generation_kwargs)
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

    stats = BatchProfileStats(
        chunk_indices=chunk_indices,
        batch_size=input_ids.shape[0],
        wall_time_s=round(t1 - t0, 4),
        peak_vram_mb=round(peak_vram_mb, 2),
    )
    return output_ids, stats


# ---------------------------------------------------------------------------
# Influence scoring
# ---------------------------------------------------------------------------

def compute_semantic_similarity(text_a: str, text_b: str, sim_model) -> float:
    embeddings = sim_model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def compute_logprob_drops_batched(
    model,
    input_ids: torch.Tensor,
    baseline_output_ids: torch.Tensor,
    chunk_boundaries: list[tuple[int, int]],
    ablation_batch_size: int,
    device: torch.device,
) -> list[float]:
    """Compute log-probability drops for all chunks using batched forward passes.

    One full forward pass computes baseline log-probs (shared across all chunks).
    The ablated forward passes are batched by ablation_batch_size, each using the
    same combined sequence (prompt + baseline output) with the chunk zeroed in the
    prompt portion of the attention mask.

    Returns a list of drop values, one per chunk, in the same order as
    chunk_boundaries.  Positive drop = chunk helped the model produce that output.
    """
    prompt_len = input_ids.shape[1]
    output_only = baseline_output_ids[0, prompt_len:]          # (output_len,)
    combined = torch.cat(
        [input_ids[0], output_only], dim=0
    ).unsqueeze(0).to(device)                                   # (1, combined_len)
    combined_len = combined.shape[1]
    target_tokens = output_only.to(device)

    # Baseline log-probs (one forward pass, reused for all chunks)
    with torch.no_grad():
        full_logits = model(combined).logits                    # (1, combined_len, vocab)
    output_logits_full = full_logits[0, prompt_len - 1 : -1, :]
    log_probs_full = torch.log_softmax(output_logits_full, dim=-1)
    lp_full = log_probs_full.gather(
        1, target_tokens.unsqueeze(1)
    ).squeeze(1)                                                # (output_len,)

    # Ablated forward passes — batched
    B = ablation_batch_size if ablation_batch_size > 0 else len(chunk_boundaries)
    drops: list[float] = []

    for b_start in range(0, len(chunk_boundaries), B):
        b_end = min(b_start + B, len(chunk_boundaries))
        batch_boundaries = chunk_boundaries[b_start:b_end]
        bsz = len(batch_boundaries)

        batched_combined = combined.repeat(bsz, 1)             # (bsz, combined_len)
        ablated_masks = torch.ones(
            bsz, combined_len, dtype=torch.long, device=device
        )
        for i, (start, end) in enumerate(batch_boundaries):
            ablated_masks[i, start:end] = 0                    # mask prompt chunk only

        with torch.no_grad():
            ablated_logits = model(
                batched_combined, attention_mask=ablated_masks
            ).logits                                            # (bsz, combined_len, vocab)

        for i in range(bsz):
            output_logits_abl = ablated_logits[i, prompt_len - 1 : -1, :]
            log_probs_abl = torch.log_softmax(output_logits_abl, dim=-1)
            lp_abl = log_probs_abl.gather(
                1, target_tokens.unsqueeze(1)
            ).squeeze(1)
            drop = (lp_full - lp_abl).mean().item()
            drops.append(round(drop, 6))

    return drops


# ---------------------------------------------------------------------------
# Summary metrics  (identical definitions to baseline)
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
    ablation_batch_size: int = 0,
    num_stages: int = 3,
    compute_logprobs: bool = False,
) -> ExampleResult:
    device = next(model.parameters()).device
    result = ExampleResult(chunk_size=chunk_size)

    # Tokenize once — reused for every ablation run unchanged
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    result.prompt_tokens = input_ids.shape[1]
    seq_len = result.prompt_tokens

    chunks = chunk_token_ids(input_ids[0].tolist(), chunk_size)
    result.num_chunks = len(chunks)

    # Effective batch size: 0 means "all chunks at once"
    B = ablation_batch_size if ablation_batch_size > 0 else result.num_chunks
    result.ablation_batch_size = B
    num_batches = (result.num_chunks + B - 1) // B

    print(
        f"  Prompt: {seq_len} tokens → {result.num_chunks} chunks × {chunk_size} "
        f"| batch_size={B} → {num_batches} ablation batch(es) "
        f"(was {result.num_chunks} sequential passes)"
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # ---- Baseline generation (all-ones mask, batch size 1) ----
    print("  Generating baseline...")
    baseline_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
    baseline_output_ids, baseline_stats = generate_with_profile(
        model, input_ids, baseline_mask, gen_kwargs, chunk_indices=[-1]
    )
    baseline_text = tokenizer.decode(
        baseline_output_ids[0, seq_len:], skip_special_tokens=True
    )
    result.baseline_text = baseline_text
    result.profile.append(asdict(baseline_stats))

    # ---- Pre-compute all chunk boundaries and ablation masks ----
    chunk_boundaries: list[tuple[int, int]] = []
    for ci, chunk in enumerate(chunks):
        start = ci * chunk_size
        end = start + len(chunk)
        chunk_boundaries.append((start, end))

    # Shape: (m, seq_len) — built once, sliced into batches below
    all_ablation_masks = build_ablation_masks(seq_len, chunk_boundaries, device)

    # ---- Batched ablation runs ----
    ablated_texts: list[str] = []

    for b_start in range(0, result.num_chunks, B):
        b_end = min(b_start + B, result.num_chunks)
        batch_indices = list(range(b_start, b_end))
        bsz = len(batch_indices)

        batch_masks = all_ablation_masks[b_start:b_end]        # (bsz, seq_len)
        batched_input_ids = input_ids.repeat(bsz, 1)           # (bsz, seq_len)

        print(
            f"  Ablating chunks {b_start + 1}–{b_end}/{result.num_chunks} "
            f"(batch {b_start // B + 1}/{num_batches}, size={bsz})...",
            end=" ",
        )

        batch_output_ids, batch_stats = generate_with_profile(
            model, batched_input_ids, batch_masks, gen_kwargs,
            chunk_indices=batch_indices,
        )
        result.profile.append(asdict(batch_stats))

        # Decode each sequence in the batch; output starts at seq_len for all
        # because input length never changes (key advantage over physical deletion)
        for local_i in range(bsz):
            text = tokenizer.decode(
                batch_output_ids[local_i, seq_len:], skip_special_tokens=True
            )
            ablated_texts.append(text)

        print(
            f"time={batch_stats.wall_time_s:.2f}s  "
            f"vram={batch_stats.peak_vram_mb:.0f}MB"
        )

    result.ablated_texts = ablated_texts

    # ---- Optional batched log-prob drops ----
    lp_drops: list[float] = [0.0] * result.num_chunks
    if compute_logprobs:
        print("  Computing log-prob drops (batched forward passes)...")
        lp_drops = compute_logprob_drops_batched(
            model, input_ids, baseline_output_ids,
            chunk_boundaries, ablation_batch_size, device,
        )

    # ---- Per-chunk influence scoring ----
    all_influences = []
    stage_matrix = []

    for ci in range(result.num_chunks):
        sem_sim  = compute_semantic_similarity(baseline_text, ablated_texts[ci], sim_model)
        influence = max(0.0, 1.0 - sem_sim)

        start, end = chunk_boundaries[ci]
        all_influences.append(ChunkInfluence(
            chunk_index=ci,
            start_token=start,
            end_token=end,
            semantic_similarity=round(sem_sim, 6),
            influence_score=round(influence, 6),
            logprob_drop=lp_drops[ci],
        ))
        print(
            f"  Chunk {ci + 1:>3}/{result.num_chunks}  "
            f"[{start}:{end}]  influence={influence:.4f}"
            + (f"  lp_drop={lp_drops[ci]:.4f}" if compute_logprobs else "")
        )

        stage_scores = compute_stage_influences(
            baseline_text, ablated_texts[ci], num_stages, sim_model
        )
        stage_matrix.append(stage_scores)

    result.chunk_influences = [asdict(ci) for ci in all_influences]

    # ---- Summary metrics ----
    scores = [ci.influence_score for ci in all_influences]
    result.eucr = compute_eucr(scores, eucr_thresholds)
    result.pwup  = compute_pwup(scores)

    if num_stages > 1 and stage_matrix:
        transposed = list(zip(*stage_matrix))   # S × m
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
        description="Attention-mask chunk ablation — batched, position-preserving"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--input_file", type=str, required=True,
                        help='JSONL file with prompts ({"prompt": ..., ...})')
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Chunk size in tokens (default: 512)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--ablation_batch_size", type=int, default=0,
                        help="Number of chunks per batched generate() call. "
                             "0 = all chunks at once (default, fastest). "
                             "Reduce if you OOM.")
    parser.add_argument("--output_dir", type=str, default="results_mask",
                        help="Output directory (default: results_mask)")
    parser.add_argument("--eucr_thresholds", type=float, nargs="+",
                        default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--compute_logprobs", action="store_true",
                        help="Also compute batched log-probability drops")
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
            ablation_batch_size=args.ablation_batch_size,
            num_stages=args.num_stages,
            compute_logprobs=args.compute_logprobs,
        )

        print(f"\n  --- Summary ---")
        print(f"  EUCR: {result.eucr}")
        print(f"  PWUP: {result.pwup}")
        print(f"  GUD:  {result.gud}")
        total_time = sum(p["wall_time_s"] for p in result.profile)
        num_batches = sum(1 for p in result.profile if p["chunk_indices"] != [-1])
        print(f"  Total wall time:    {total_time:.2f}s")
        print(f"  Ablation batches:   {num_batches}  (was {result.num_chunks} sequential)")
        print(f"  Effective batch size: {result.ablation_batch_size}")

        all_results.append(asdict(result))

    # Save per-example results
    out_path = Path(args.output_dir) / "mask_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save profiling summary — schema compatible with run_benchmark.py / parse_profiling().
    # mean_wall_time_s is total / (num_chunks + 1) to give an "effective per-run" time
    # comparable to the sequential baseline.  Extra fields document the batching.
    profile_path = Path(args.output_dir) / "profiling_summary.json"
    profile_summary = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "ablation_method": "attention_mask_batched",
        "examples": [],
    }
    for idx, r in enumerate(all_results):
        times = [p["wall_time_s"] for p in r["profile"]]
        vrams = [p["peak_vram_mb"] for p in r["profile"]]
        total_t = sum(times)
        n_runs = r["num_chunks"] + 1          # +1 for baseline, matches sequential count
        ablation_batches = sum(
            1 for p in r["profile"] if p["chunk_indices"] != [-1]
        )
        profile_summary["examples"].append({
            "index":                    idx,
            "prompt_tokens":            r["prompt_tokens"],
            "num_chunks":               r["num_chunks"],
            "total_ablation_runs":      n_runs,
            "input_tokens_per_run":     r["prompt_tokens"],   # constant — no shifting
            "ablation_batch_size":      r["ablation_batch_size"],
            "total_ablation_batches":   ablation_batches,
            "total_wall_time_s":        round(total_t, 2),
            "mean_wall_time_s":         round(total_t / n_runs, 4),
            "max_peak_vram_mb":         round(max(vrams), 2),
        })
    with open(profile_path, "w") as f:
        json.dump(profile_summary, f, indent=2)
    print(f"Profiling summary saved to {profile_path}")


if __name__ == "__main__":
    main()
