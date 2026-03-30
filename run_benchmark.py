#!/usr/bin/env python3
"""
run_benchmark.py
================
Sweeps chunk_deletion_baseline.py over multiple (prompt_length, chunk_size)
combinations and collects profiling data into a summary CSV.

The model is loaded once per chunk_size, so N chunk sizes = N model loads.
Each chunk_size run processes all target lengths in a single subprocess call.

Output CSV columns:
    prompt_length, chunk_size, num_ablations,
    total_wall_time_s, avg_time_per_ablation_s, peak_vram_mb

Usage:
    # Sweep default lengths and chunk sizes
    python run_benchmark.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output_csv benchmark_results.csv

    # Specify lengths and chunk sizes explicitly
    python run_benchmark.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --lengths 2k 4k 8k 16k \\
        --chunk_sizes 128 256 512 \\
        --num_examples 3 \\
        --output_csv benchmark_results.csv

    # Provide your own prompts (skips generation)
    python run_benchmark.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --input_file ruler_prompts.jsonl \\
        --chunk_sizes 256 512 \\
        --output_csv benchmark_results.csv
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Optional

# Import generate_ruler_prompts if available (same directory)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from generate_ruler_prompts import (
        generate_example,
        make_token_counter,
        parse_length,
    )
    _CAN_GENERATE = True
except ImportError:
    _CAN_GENERATE = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LENGTHS    = ["2k", "4k", "8k", "16k"]
DEFAULT_CHUNK_SIZES = [128, 256, 512]
DEFAULT_MAX_NEW_TOKENS = 64   # short outputs to keep ablation fast during benchmarking


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def generate_benchmark_prompts(
    target_lengths: list[int],
    num_examples: int,
    seed: int,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Generate `num_examples` prompts per target length.

    Returns:
        records   — list of dicts ready to write as JSONL
        manifest  — list of (record_index, target_length) for joining later
    """
    if not _CAN_GENERATE:
        raise RuntimeError(
            "generate_ruler_prompts.py not found in the same directory. "
            "Either add it or supply --input_file."
        )

    import random
    rng = random.Random(seed)
    count_tokens = make_token_counter(None)
    used_labels: set = set()

    records: list[dict] = []
    manifest: list[tuple[int, int]] = []

    for target_tokens in target_lengths:
        for _ in range(num_examples):
            ex = generate_example(
                target_tokens=target_tokens,
                position="middle",       # middle is the hardest; good default for benchmarking
                rng=rng,
                count_tokens=count_tokens,
                used_labels=used_labels,
            )
            idx = len(records)
            records.append({
                "prompt":    ex["prompt"],
                "reference": ex["reference"],
                "task":      ex["task"],
            })
            manifest.append((idx, target_tokens))

    return records, manifest


def load_prompts_from_file(
    path: str,
    target_lengths: Optional[list[int]],
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Load prompts from a JSONL file.

    If target_lengths is given and the records contain a 'metadata.target_tokens'
    field (written by generate_ruler_prompts.py --include_metadata), examples are
    grouped by target length.  Otherwise all records are loaded and target_length
    is approximated from actual prompt character count.
    """
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    manifest: list[tuple[int, int]] = []
    for idx, rec in enumerate(records):
        # Prefer explicit metadata
        meta_len = (rec.get("metadata") or {}).get("target_tokens")
        if meta_len:
            assigned = meta_len
        elif target_lengths:
            # Snap actual prompt length to the nearest requested target
            approx = len(rec["prompt"]) // 4
            assigned = min(target_lengths, key=lambda t: abs(t - approx))
        else:
            assigned = len(rec["prompt"]) // 4  # raw approximation
        manifest.append((idx, assigned))

    # Filter to requested lengths if specified
    if target_lengths:
        tset = set(target_lengths)
        pairs = [(r, m) for r, m in zip(records, manifest) if m[1] in tset]
        if not pairs:
            raise ValueError(
                f"No records matched target lengths {target_lengths}. "
                "Check --lengths or remove the flag to use all records."
            )
        records, manifest = zip(*pairs)  # type: ignore[assignment]
        records = list(records)
        manifest = list(manifest)

    return records, manifest


# ---------------------------------------------------------------------------
# Running the baseline subprocess
# ---------------------------------------------------------------------------

def run_baseline(
    model: str,
    input_jsonl: str,
    output_dir: str,
    chunk_size: int,
    max_new_tokens: int,
    dtype: str,
    extra_args: list[str],
) -> Path:
    """Invoke chunk_deletion_baseline.py and return the path to profiling_summary.json.

    Streams stdout/stderr live so the user sees progress, then raises on failure.
    """
    baseline_script = Path(__file__).parent / "chunk_deletion_baseline.py"
    cmd = [
        sys.executable, str(baseline_script),
        "--model",          model,
        "--input_file",     input_jsonl,
        "--output_dir",     output_dir,
        "--chunk_size",     str(chunk_size),
        "--max_new_tokens", str(max_new_tokens),
        "--dtype",          dtype,
        *extra_args,
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"chunk_size={chunk_size}  output_dir={output_dir}", flush=True)
    print(f"cmd: {' '.join(cmd)}", flush=True)
    print(f"{'='*70}", flush=True)

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"chunk_deletion_baseline.py exited with code {result.returncode} "
            f"(chunk_size={chunk_size})"
        )

    profile_path = Path(output_dir) / "profiling_summary.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Expected profiling output not found: {profile_path}"
        )
    return profile_path


# ---------------------------------------------------------------------------
# Parsing profiling output
# ---------------------------------------------------------------------------

def parse_profiling(
    profile_path: Path,
    manifest: list[tuple[int, int]],
) -> list[dict]:
    """Join profiling_summary.json with the manifest to produce flat rows.

    Returns a list of dicts, one per example, with target_length attached.
    """
    with profile_path.open() as f:
        summary = json.load(f)

    index_to_target = {idx: tlen for idx, tlen in manifest}

    rows = []
    for ex in summary["examples"]:
        idx = ex["index"]
        rows.append({
            "chunk_size":               summary["chunk_size"],
            "prompt_length":            index_to_target.get(idx, ex["prompt_tokens"]),
            "actual_prompt_tokens":     ex["prompt_tokens"],
            "num_ablations":            ex["total_ablation_runs"],
            "total_wall_time_s":        ex["total_wall_time_s"],
            "avg_time_per_ablation_s":  ex["mean_wall_time_s"],
            "peak_vram_mb":             ex["max_peak_vram_mb"],
        })
    return rows


# ---------------------------------------------------------------------------
# CSV aggregation
# ---------------------------------------------------------------------------

def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Average metrics across examples that share the same (prompt_length, chunk_size)."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["prompt_length"], row["chunk_size"])
        groups[key].append(row)

    agg = []
    for (prompt_length, chunk_size), group in sorted(groups.items()):
        agg.append({
            "prompt_length":            prompt_length,
            "chunk_size":               chunk_size,
            "num_examples":             len(group),
            "num_ablations":            round(mean(r["num_ablations"] for r in group), 1),
            "total_wall_time_s":        round(mean(r["total_wall_time_s"] for r in group), 3),
            "avg_time_per_ablation_s":  round(mean(r["avg_time_per_ablation_s"] for r in group), 4),
            "peak_vram_mb":             round(mean(r["peak_vram_mb"] for r in group), 1),
        })
    return agg


CSV_COLUMNS = [
    "prompt_length",
    "chunk_size",
    "num_examples",
    "num_ablations",
    "total_wall_time_s",
    "avg_time_per_ablation_s",
    "peak_vram_mb",
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark chunk_deletion_baseline.py across prompt lengths and chunk sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name or path (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )

    # Sweep dimensions
    parser.add_argument(
        "--lengths", nargs="+", default=DEFAULT_LENGTHS,
        metavar="L",
        help=f"Target prompt lengths to sweep (e.g. 2k 4k 8k 16k). "
             f"Default: {' '.join(DEFAULT_LENGTHS)}",
    )
    parser.add_argument(
        "--chunk_sizes", nargs="+", type=int, default=DEFAULT_CHUNK_SIZES,
        metavar="C",
        help=f"Chunk sizes to sweep. Default: {DEFAULT_CHUNK_SIZES}",
    )

    # Prompt source
    parser.add_argument(
        "--input_file", default=None,
        help="JSONL file of prompts to use. If omitted, prompts are generated "
             "automatically using generate_ruler_prompts.py.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=1,
        help="Examples per (length, chunk_size) combination when auto-generating "
             "prompts. Ignored if --input_file is given. Default: 1",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for prompt generation. Default: 42",
    )

    # Baseline pass-through
    parser.add_argument(
        "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max tokens to generate per ablation run. "
             f"Keep small to speed up benchmarking. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype. Default: bfloat16",
    )
    parser.add_argument(
        "--baseline_args", nargs=argparse.REMAINDER, default=[],
        help="Extra arguments forwarded verbatim to chunk_deletion_baseline.py "
             "(e.g. -- --compute_logprobs).",
    )

    # Output
    parser.add_argument(
        "--output_csv", default="benchmark_results.csv",
        help="Path for the summary CSV. Default: benchmark_results.csv",
    )
    parser.add_argument(
        "--work_dir", default=None,
        help="Directory for intermediate files (per-run JSONL and profiling JSON). "
             "Defaults to a temp directory that is kept on exit for inspection.",
    )

    args = parser.parse_args()

    # Parse lengths
    target_lengths = [parse_length(l) for l in args.lengths]

    # Strip leading '--' separator for baseline_args if present
    extra_baseline_args = [a for a in args.baseline_args if a != "--"]

    # Work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        keep_work_dir = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        keep_work_dir = True  # always keep for post-hoc inspection

    print(f"Work directory: {work_dir}", flush=True)

    # ------------------------------------------------------------------
    # Prepare prompts
    # ------------------------------------------------------------------
    if args.input_file:
        print(f"Loading prompts from {args.input_file}", flush=True)
        records, manifest = load_prompts_from_file(args.input_file, target_lengths)
    else:
        print(
            f"Generating {args.num_examples} example(s) × {len(target_lengths)} lengths "
            f"= {args.num_examples * len(target_lengths)} prompts ...",
            flush=True,
        )
        records, manifest = generate_benchmark_prompts(
            target_lengths=target_lengths,
            num_examples=args.num_examples,
            seed=args.seed,
        )

    # Write combined JSONL once (shared across all chunk_size runs)
    prompts_jsonl = work_dir / "benchmark_prompts.jsonl"
    with prompts_jsonl.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} prompts to {prompts_jsonl}", flush=True)

    # ------------------------------------------------------------------
    # Sweep over chunk sizes
    # ------------------------------------------------------------------
    all_raw_rows: list[dict] = []
    failed: list[int] = []

    for chunk_size in args.chunk_sizes:
        run_out_dir = work_dir / f"chunk{chunk_size}"
        run_out_dir.mkdir(exist_ok=True)

        try:
            profile_path = run_baseline(
                model=args.model,
                input_jsonl=str(prompts_jsonl),
                output_dir=str(run_out_dir),
                chunk_size=chunk_size,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
                extra_args=extra_baseline_args,
            )
            rows = parse_profiling(profile_path, manifest)
            all_raw_rows.extend(rows)
            print(
                f"  chunk_size={chunk_size}: collected {len(rows)} profiling rows",
                flush=True,
            )
        except Exception as e:
            print(f"  ERROR for chunk_size={chunk_size}: {e}", file=sys.stderr)
            failed.append(chunk_size)

    if not all_raw_rows:
        print("No profiling data collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Aggregate and write CSV
    # ------------------------------------------------------------------
    agg_rows = aggregate_rows(all_raw_rows)
    write_csv(agg_rows, args.output_csv)

    print(f"\n{'='*70}")
    print(f"Benchmark complete.")
    print(f"  Rows collected : {len(all_raw_rows)}")
    print(f"  Rows in CSV    : {len(agg_rows)}")
    print(f"  Output CSV     : {args.output_csv}")
    print(f"  Work dir       : {work_dir}")
    if failed:
        print(f"  Failed chunk sizes: {failed}", file=sys.stderr)

    # Print a quick preview table
    print()
    header = f"{'prompt_len':>12} {'chunk_sz':>9} {'n_ablations':>12} "
    header += f"{'total_wt(s)':>12} {'avg_wt(s)':>10} {'peak_vram(MB)':>14}"
    print(header)
    print("-" * len(header))
    for r in agg_rows:
        print(
            f"{r['prompt_length']:>12} {r['chunk_size']:>9} {r['num_ablations']:>12} "
            f"{r['total_wall_time_s']:>12.2f} {r['avg_time_per_ablation_s']:>10.3f} "
            f"{r['peak_vram_mb']:>14.1f}"
        )


if __name__ == "__main__":
    main()
