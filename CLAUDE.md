# Context Utilization Research — Claude Guide

## Project Overview

This is a research project (CS 498) measuring **effective context utilization in long-context LLMs** via chunk-level ablation. The current file (`chunk_deletion_baseline.py`) is the **naive baseline**: it physically deletes token chunks, regenerates, and measures how much each chunk influences the output.

The eventual goal is to replace physical deletion with attention-mask manipulation to avoid RoPE positional encoding shift artifacts and reduce compute from O(m+1) forward passes.

## Key Concepts

- **EUCR[λ]** — Effective Utilized Context Ratio: fraction of chunks whose ablation changes output beyond threshold λ
- **PWUP** — Position-Weighted Utilization Profile: (U_B, U_M, U_E) influence distribution across beginning/middle/end thirds
- **GUD** — Generation Utilization Drift: how the influential chunk set shifts across output stages
- **Chunk** — contiguous fixed-size (default 512) token window; last chunk may be shorter

## Codebase

- `chunk_deletion_baseline.py` — Main script. Runs ablation pipeline, computes metrics, profiles VRAM and wall time.
- `sample_prompts.jsonl` — Small test input (2 QA examples). Input format: `{"prompt": "...", "reference": "...", "task": "qa"}`
- `results/` — Output directory (gitignored). Contains `chunk_deletion_results.json` and `profiling_summary.json`.

## Running the Baseline

```bash
python chunk_deletion_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input_file sample_prompts.jsonl \
    --output_dir results/
```

Key flags: `--chunk_size` (default 512), `--compute_logprobs` (slower, adds logprob drop metric), `--dtype bfloat16` (default).

## Architecture Notes

- Model loaded via HuggingFace `transformers` with `device_map` and configurable dtype.
- Semantic similarity uses `sentence-transformers/all-MiniLM-L6-v2` (cosine sim on embeddings).
- Profiling resets `torch.cuda.peak_memory_stats` before each forward pass.
- Greedy decoding (`do_sample=False`) for reproducibility across ablation runs.

## Known Limitation (the whole point of this baseline)

Physically removing tokens shifts RoPE positions for all subsequent tokens — the model sees different positional encodings even for untouched content. This baseline exists to quantify the wall-time and VRAM cost of m+1 full forward passes, motivating the next step: attention-mask-only ablation.

## Next Steps

1. Profile this baseline across prompts of increasing length.
2. Use profiling numbers to motivate attention-mask optimization.
3. Implement mask-based ablation: zero attention to the target chunk while preserving positional encodings.
