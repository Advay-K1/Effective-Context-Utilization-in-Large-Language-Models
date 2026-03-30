# Physical Chunk Deletion Baseline

Naive ablation framework for measuring effective context utilization in long-context LLMs. This is the **baseline** implementation that physically removes chunks — your advisor recommends profiling this first to demonstrate why the attention mask optimization is necessary.

## Setup

```bash
pip install torch transformers sentence-transformers accelerate numpy
```

## Usage

```bash
# Basic run with default settings (chunk_size=512, greedy decoding)
python chunk_deletion_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input_file sample_prompts.jsonl \
    --output_dir results/

# Try different chunk sizes to study granularity sensitivity
python chunk_deletion_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input_file prompts.jsonl \
    --chunk_size 256 \
    --output_dir results/chunk256/

# With log-probability drop (slower but more informative)
python chunk_deletion_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input_file prompts.jsonl \
    --chunk_size 512 \
    --compute_logprobs \
    --output_dir results/with_logprobs/
```

## Input Format

JSONL file, one JSON object per line:

```json
{"prompt": "Your long prompt text...", "reference": "optional gold answer", "task": "qa"}
```

## Output

Two files in `--output_dir`:

- **`chunk_deletion_results.json`** — Per-example results including:
  - Baseline and ablated generated texts
  - Per-chunk influence scores (semantic similarity + optional logprob drop)
  - EUCR at multiple thresholds
  - PWUP (beginning / middle / end utilization)
  - GUD (generation utilization drift across output stages)

- **`profiling_summary.json`** — Wall-clock time and peak VRAM per run, which is the key data your advisor wants for the midterm review.

## Metrics Implemented

| Metric | What it measures |
|--------|-----------------|
| **EUCR[λ]** | Fraction of chunks whose ablation changes output beyond threshold λ |
| **PWUP** | (U_B, U_M, U_E) — how influence distributes across beginning/middle/end |
| **GUD** | How the set of influential chunks shifts across output stages |

## Known Limitation (the point of this baseline)

Physically deleting tokens shifts RoPE positional encodings for all subsequent tokens, meaning the model sees different positions even for unchanged content. This baseline quantifies how expensive the naive approach is (m+1 full forward passes) and motivates the attention-mask-only optimization as the next step.

## Next Steps

After profiling this baseline:
1. Record total wall time and peak VRAM for prompts of increasing length
2. Use those numbers to motivate switching to attention mask manipulation
3. Implement the mask-based ablation that zeros out attention to the target chunk while preserving positional encodings
