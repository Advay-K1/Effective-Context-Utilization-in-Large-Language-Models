#!/usr/bin/env python3
"""
Comprehensive Ablation Methods Benchmark
========================================
Compares three different context utilization measurement methods:

1. Physical Chunk Deletion (baseline) - chunk_deletion_baseline.py
2. Attention Mask Ablation (batched) - attention_mask_ablation.py  
3. Input×Gradient Attribution - inputxgrad_ablation.py

Measures:
- Wall-clock time efficiency
- Peak memory usage
- Computational complexity (forward passes)
- Quality of importance scores (correlation analysis)

Usage:
    # Compare all methods on sample prompts
    python benchmark_all_methods.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --input_file sample_prompts.jsonl \\
        --output_dir benchmark_results/

    # Sweep across different prompt lengths
    python benchmark_all_methods.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --lengths 512 1k 2k 4k \\
        --chunk_sizes 128 256 512 \\
        --num_examples 3 \\
        --output_dir benchmark_results/

    # Quick test with single method
    python benchmark_all_methods.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --methods inputxgrad \\
        --input_file sample_prompts.jsonl \\
        --output_dir test_results/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from generate_ruler_prompts import generate_example, make_token_counter, parse_length
    _CAN_GENERATE = True
except ImportError:
    _CAN_GENERATE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METHODS = {
    "deletion": {
        "script": "chunk_deletion_baseline.py",
        "output_file": "chunk_deletion_results.json",
        "profile_file": "profiling_summary.json",
        "description": "Physical chunk deletion (baseline)",
        "complexity": "O(m+1) forward passes"
    },
    "attention_mask": {
        "script": "attention_mask_ablation.py", 
        "output_file": "mask_ablation_results.json",
        "profile_file": "profiling_summary.json",
        "description": "Batched attention masking",
        "complexity": "O(ceil(m/B)+1) forward passes"
    },
    "inputxgrad": {
        "script": "inputxgrad_ablation.py",
        "output_file": "inputxgrad_results.json", 
        "profile_file": "profiling_summary.json",
        "description": "Input×Gradient attribution",
        "complexity": "O(1) forward + 1 backward pass"
    }
}

DEFAULT_METHODS = ["deletion", "attention_mask", "inputxgrad"]
DEFAULT_LENGTHS = ["512", "1k", "2k"]
DEFAULT_CHUNK_SIZES = [128, 256, 512]
DEFAULT_NUM_EXAMPLES = 1
DEFAULT_MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------
# Prompt generation (reuse from run_benchmark.py)
# ---------------------------------------------------------------------------

def generate_benchmark_prompts(
    target_lengths: List[int],
    num_examples: int,
    seed: int,
) -> tuple[List[Dict], List[tuple[int, int]]]:
    """Generate prompts for benchmarking."""
    if not _CAN_GENERATE:
        raise RuntimeError(
            "generate_ruler_prompts.py not found. Either add it or use --input_file."
        )
    
    import random
    rng = random.Random(seed)
    count_tokens = make_token_counter(None)
    used_labels: set = set()
    
    records: List[Dict] = []
    manifest: List[tuple[int, int]] = []
    
    for target_tokens in target_lengths:
        for _ in range(num_examples):
            ex = generate_example(
                target_tokens=target_tokens,
                position="middle",
                rng=rng,
                count_tokens=count_tokens,
                used_labels=used_labels,
            )
            idx = len(records)
            records.append({
                "prompt": ex["prompt"],
                "reference": ex["reference"],
                "task": ex["task"],
            })
            manifest.append((idx, target_tokens))
    
    return records, manifest


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_method(
    method: str,
    model: str,
    input_jsonl: str,
    output_dir: str,
    chunk_size: int,
    max_new_tokens: int,
    dtype: str,
    extra_args: List[str]
) -> Dict:
    """Run a single ablation method and return results."""
    method_info = METHODS[method]
    script_path = Path(__file__).parent / method_info["script"]
    
    method_output_dir = Path(output_dir) / f"{method}_chunk{chunk_size}"
    method_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, str(script_path),
        "--model", model,
        "--input_file", input_jsonl,
        "--output_dir", str(method_output_dir),
        "--chunk_size", str(chunk_size),
        "--max_new_tokens", str(max_new_tokens),
        "--dtype", dtype,
        *extra_args,
    ]
    
    # Add method-specific arguments
    if method == "attention_mask":
        cmd.extend(["--ablation_batch_size", "0"])  # Batch all chunks
    elif method == "inputxgrad":
        cmd.extend(["--attribution_method", "inputxgrad"])
    
    print(f"\n{'='*70}")
    print(f"Running {method_info['description']}")
    print(f"chunk_size={chunk_size}  output_dir={method_output_dir}")
    print(f"cmd: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"ERROR: {method} failed with code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    # Load results
    results_path = method_output_dir / method_info["output_file"]
    profile_path = method_output_dir / method_info["profile_file"]
    
    if not results_path.exists() or not profile_path.exists():
        print(f"ERROR: Expected output files not found for {method}")
        return None
    
    with open(results_path) as f:
        results_data = json.load(f)
    with open(profile_path) as f:
        profile_data = json.load(f)
    
    return {
        "method": method,
        "chunk_size": chunk_size,
        "results_data": results_data,
        "profile_data": profile_data,
        "script_runtime": end_time - start_time,
    }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def extract_importance_scores(method_result: Dict) -> List[List[float]]:
    """Extract chunk importance scores from method results."""
    method = method_result["method"]
    results_data = method_result["results_data"]
    
    all_scores = []
    for example in results_data:
        if method == "deletion":
            scores = [ci["influence_score"] for ci in example["chunk_influences"]]
        elif method == "attention_mask":
            scores = [ci["influence_score"] for ci in example["chunk_influences"]]
        elif method == "inputxgrad":
            scores = [ci["inputxgrad_importance"] for ci in example["chunk_influences"]]
        else:
            scores = []
        all_scores.append(scores)
    
    return all_scores


def compute_method_correlations(results: List[Dict]) -> Dict:
    """Compute correlations between different methods' importance scores."""
    correlations = {}
    
    # Extract all importance scores
    method_scores = {}
    for result in results:
        method = result["method"]
        method_scores[method] = extract_importance_scores(result)
    
    # Compute pairwise correlations
    methods = list(method_scores.keys())
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            scores1_flat = [score for example in method_scores[method1] for score in example]
            scores2_flat = [score for example in method_scores[method2] for score in example]
            
            # Ensure same length (should be if same prompts/chunks)
            min_len = min(len(scores1_flat), len(scores2_flat))
            scores1_flat = scores1_flat[:min_len]
            scores2_flat = scores2_flat[:min_len]
            
            if len(scores1_flat) > 1:
                pearson_r, pearson_p = pearsonr(scores1_flat, scores2_flat)
                spearman_r, spearman_p = spearmanr(scores1_flat, scores2_flat)
                
                correlations[f"{method1}_vs_{method2}"] = {
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n_points": len(scores1_flat),
                }
    
    return correlations


def analyze_efficiency(results: List[Dict]) -> Dict:
    """Analyze computational efficiency of different methods."""
    efficiency_data = {}
    
    for result in results:
        method = result["method"]
        profile_data = result["profile_data"]
        
        # Aggregate across examples
        times = []
        vrams = []
        forward_passes = []
        
        for example in profile_data["examples"]:
            times.append(example["total_wall_time_s"])
            vrams.append(example["max_peak_vram_mb"])
            
            # Estimate forward passes based on method
            if method == "deletion":
                forward_passes.append(example["total_ablation_runs"])
            elif method == "attention_mask":
                # Batched, fewer passes
                forward_passes.append(example.get("total_ablation_batches", 1) + 1)
            elif method == "inputxgrad":
                forward_passes.append(1)  # Only one forward pass
        
        efficiency_data[method] = {
            "mean_time_s": mean(times),
            "std_time_s": stdev(times) if len(times) > 1 else 0.0,
            "mean_vram_mb": mean(vrams),
            "std_vram_mb": stdev(vrams) if len(vrams) > 1 else 0.0,
            "mean_forward_passes": mean(forward_passes),
            "std_forward_passes": stdev(forward_passes) if len(forward_passes) > 1 else 0.0,
            "complexity": METHODS[method]["complexity"],
            "script_runtime_s": result["script_runtime"],
        }
    
    return efficiency_data


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_efficiency_plots(efficiency_data: Dict, output_dir: Path):
    """Create efficiency comparison plots."""
    methods = list(efficiency_data.keys())
    
    # Time comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Wall time
    times = [efficiency_data[m]["mean_time_s"] for m in methods]
    time_stds = [efficiency_data[m]["std_time_s"] for m in methods]
    axes[0, 0].bar(methods, times, yerr=time_stds, capsize=5)
    axes[0, 0].set_title("Wall Time Comparison")
    axes[0, 0].set_ylabel("Time (seconds)")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # VRAM usage
    vrams = [efficiency_data[m]["mean_vram_mb"] for m in methods]
    vram_stds = [efficiency_data[m]["std_vram_mb"] for m in methods]
    axes[0, 1].bar(methods, vrams, yerr=vram_stds, capsize=5)
    axes[0, 1].set_title("Peak VRAM Usage")
    axes[0, 1].set_ylabel("VRAM (MB)")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Forward passes
    passes = [efficiency_data[m]["mean_forward_passes"] for m in methods]
    pass_stds = [efficiency_data[m]["std_forward_passes"] for m in methods]
    axes[1, 0].bar(methods, passes, yerr=pass_stds, capsize=5)
    axes[1, 0].set_title("Forward Passes Required")
    axes[1, 0].set_ylabel("Number of Forward Passes")
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Script runtime (total including overhead)
    runtimes = [efficiency_data[m]["script_runtime_s"] for m in methods]
    axes[1, 1].bar(methods, runtimes)
    axes[1, 1].set_title("Total Script Runtime")
    axes[1, 1].set_ylabel("Runtime (seconds)")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_correlation_heatmap(correlations: Dict, output_dir: Path):
    """Create correlation heatmap between methods."""
    if not correlations:
        return
    
    # Extract correlation matrix
    methods = set()
    for key in correlations.keys():
        m1, m2 = key.split("_vs_")
        methods.add(m1)
        methods.add(m2)
    
    methods = sorted(list(methods))
    n = len(methods)
    
    pearson_matrix = np.eye(n)
    spearman_matrix = np.eye(n)
    
    for key, data in correlations.items():
        m1, m2 = key.split("_vs_")
        i, j = methods.index(m1), methods.index(m2)
        
        pearson_matrix[i, j] = data["pearson_r"]
        pearson_matrix[j, i] = data["pearson_r"]
        spearman_matrix[i, j] = data["spearman_r"]
        spearman_matrix[j, i] = data["spearman_r"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pearson correlation
    sns.heatmap(pearson_matrix, annot=True, xticklabels=methods, yticklabels=methods,
                center=0, cmap="RdBu_r", ax=axes[0])
    axes[0].set_title("Pearson Correlation")
    
    # Spearman correlation
    sns.heatmap(spearman_matrix, annot=True, xticklabels=methods, yticklabels=methods,
                center=0, cmap="RdBu_r", ax=axes[1])
    axes[1].set_title("Spearman Correlation")
    
    plt.tight_layout()
    plt.savefig(output_dir / "method_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main benchmark pipeline
# ---------------------------------------------------------------------------

def run_comprehensive_benchmark(
    methods: List[str],
    model: str,
    input_file: Optional[str],
    target_lengths: Optional[List[int]],
    chunk_sizes: List[int],
    num_examples: int,
    max_new_tokens: int,
    dtype: str,
    output_dir: Path,
    seed: int,
) -> Dict:
    """Run comprehensive benchmark across all methods."""
    
    # Prepare prompts
    if input_file:
        print(f"Using prompts from {input_file}")
        # Copy the input file to work directory
        import shutil
        work_prompts = output_dir / "benchmark_prompts.jsonl"
        shutil.copy(input_file, work_prompts)
        
        # Load for length estimation
        with open(input_file) as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]
        manifest = [(i, len(rec["prompt"]) // 4) for i, rec in enumerate(records)]
    else:
        if not target_lengths:
            raise ValueError("Must specify either --input_file or --lengths")
        
        print(f"Generating {num_examples} examples × {len(target_lengths)} lengths")
        records, manifest = generate_benchmark_prompts(target_lengths, num_examples, seed)
        
        work_prompts = output_dir / "benchmark_prompts.jsonl"
        with open(work_prompts, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    
    print(f"Using {len(records)} prompts, saved to {work_prompts}")
    
    # Run each method for each chunk size
    all_results = []
    failed_runs = []
    
    for chunk_size in chunk_sizes:
        for method in methods:
            print(f"\n{'='*80}")
            print(f"RUNNING: {method} with chunk_size={chunk_size}")
            print(f"{'='*80}")
            
            try:
                result = run_method(
                    method=method,
                    model=model,
                    input_jsonl=str(work_prompts),
                    output_dir=str(output_dir),
                    chunk_size=chunk_size,
                    max_new_tokens=max_new_tokens,
                    dtype=dtype,
                    extra_args=[],
                )
                
                if result:
                    all_results.append(result)
                    print(f"✓ SUCCESS: {method} chunk_size={chunk_size}")
                else:
                    failed_runs.append((method, chunk_size))
                    print(f"✗ FAILED: {method} chunk_size={chunk_size}")
                    
            except Exception as e:
                print(f"✗ ERROR: {method} chunk_size={chunk_size}: {e}")
                failed_runs.append((method, chunk_size))
    
    if not all_results:
        raise RuntimeError("All benchmark runs failed!")
    
    # Group results by chunk size for analysis
    results_by_chunk_size = {}
    for result in all_results:
        chunk_size = result["chunk_size"]
        if chunk_size not in results_by_chunk_size:
            results_by_chunk_size[chunk_size] = []
        results_by_chunk_size[chunk_size].append(result)
    
    # Analyze results for each chunk size
    benchmark_summary = {
        "model": model,
        "chunk_sizes": chunk_sizes,
        "methods": methods,
        "num_prompts": len(records),
        "max_new_tokens": max_new_tokens,
        "failed_runs": failed_runs,
        "results_by_chunk_size": {},
    }
    
    for chunk_size, chunk_results in results_by_chunk_size.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING CHUNK_SIZE={chunk_size}")
        print(f"{'='*60}")
        
        # Efficiency analysis
        efficiency_data = analyze_efficiency(chunk_results)
        correlations = compute_method_correlations(chunk_results)
        
        # Create plots for this chunk size
        chunk_output_dir = output_dir / f"analysis_chunk{chunk_size}"
        chunk_output_dir.mkdir(exist_ok=True)
        
        create_efficiency_plots(efficiency_data, chunk_output_dir)
        create_correlation_heatmap(correlations, chunk_output_dir)
        
        # Save detailed results
        chunk_analysis = {
            "chunk_size": chunk_size,
            "efficiency": efficiency_data,
            "correlations": correlations,
            "methods_completed": [r["method"] for r in chunk_results],
        }
        
        with open(chunk_output_dir / "analysis_summary.json", "w") as f:
            json.dump(chunk_analysis, f, indent=2)
        
        benchmark_summary["results_by_chunk_size"][chunk_size] = chunk_analysis
        
        # Print summary
        print("Efficiency Summary:")
        for method, data in efficiency_data.items():
            print(f"  {method:15s}: {data['mean_time_s']:6.2f}s  "
                  f"{data['mean_vram_mb']:6.0f}MB  "
                  f"{data['mean_forward_passes']:4.1f} passes")
        
        if correlations:
            print("\nMethod Correlations (Pearson r):")
            for pair, data in correlations.items():
                print(f"  {pair:25s}: r={data['pearson_r']:5.3f}  (p={data['pearson_p']:6.4f})")
    
    return benchmark_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark of context utilization methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or path")
    
    # Methods to compare
    parser.add_argument("--methods", nargs="+", choices=list(METHODS.keys()),
                        default=DEFAULT_METHODS,
                        help=f"Methods to benchmark. Default: {DEFAULT_METHODS}")
    
    # Prompt source
    parser.add_argument("--input_file", default=None,
                        help="JSONL file of prompts. If omitted, generates prompts.")
    parser.add_argument("--lengths", nargs="+", default=DEFAULT_LENGTHS,
                        help=f"Target prompt lengths if generating. Default: {DEFAULT_LENGTHS}")
    parser.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help=f"Examples per length. Default: {DEFAULT_NUM_EXAMPLES}")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt generation")
    
    # Benchmark parameters
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=DEFAULT_CHUNK_SIZES,
                        help=f"Chunk sizes to test. Default: {DEFAULT_CHUNK_SIZES}")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"Max tokens to generate. Default: {DEFAULT_MAX_NEW_TOKENS}")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")
    
    # Output
    parser.add_argument("--output_dir", default="benchmark_results",
                        help="Output directory for all results")
    
    args = parser.parse_args()
    
    # Parse lengths if not using input file
    target_lengths = None
    if not args.input_file:
        target_lengths = [parse_length(l) for l in args.lengths]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting comprehensive benchmark...")
    print(f"Methods: {args.methods}")
    print(f"Model: {args.model}")
    print(f"Chunk sizes: {args.chunk_sizes}")
    print(f"Output: {output_dir}")
    
    try:
        summary = run_comprehensive_benchmark(
            methods=args.methods,
            model=args.model,
            input_file=args.input_file,
            target_lengths=target_lengths,
            chunk_sizes=args.chunk_sizes,
            num_examples=args.num_examples,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            output_dir=output_dir,
            seed=args.seed,
        )
        
        # Save overall summary
        with open(output_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"Methods completed: {len([r for chunk_results in summary['results_by_chunk_size'].values() for r in chunk_results['methods_completed']])}")
        if summary["failed_runs"]:
            print(f"Failed runs: {summary['failed_runs']}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())