"""
CS 498 — Effective Context Utilization (v2 — cluster edition)
=============================================================
Four RoPE-aware context-importance methods for Qwen2.5-Instruct.

Usage
-----
# Minimal (uses all defaults):
python run_experiment.py

# Typical cluster run:
python run_experiment.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --context_lengths 512 1024 2048 4096 \
    --num_examples 5 \
    --output_dir /scratch/results_v2 \
    --methods substitution inputxgrad attn_agg mask4d

# Resume a partial run:
python run_experiment.py --resume --output_dir /scratch/results_v2

# Smoke-test (fast):
python run_experiment.py --context_lengths 512 --num_examples 1 --methods inputxgrad attn_agg

HuggingFace token (if model is gated):
  export HF_TOKEN=hf_...
  # or pass --hf_token hf_...
"""

import argparse
import gc
import json
import logging
import os
import random
import string
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed on a cluster node
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# 1.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CS 498 Context Utilization Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id",        default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--chunk_size",      type=int,   default=128)
    p.add_argument("--max_new_tokens",  type=int,   default=64)
    p.add_argument("--context_lengths", type=int,   nargs="+",
                   default=[512, 1024, 2048])
    p.add_argument("--needle_depths",   type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 0.9],
                   help="Needle positions as fraction of haystack length")
    p.add_argument("--num_examples",    type=int,   default=2,
                   help="Prompts per (context_length × needle_depth) combination")
    p.add_argument("--output_dir",      default="./results_v2")
    p.add_argument("--methods",         nargs="+",
                   choices=["substitution", "inputxgrad", "attn_agg", "mask4d"],
                   default=["substitution", "inputxgrad", "attn_agg", "mask4d"])
    p.add_argument("--hf_token",        default=None,
                   help="HuggingFace token (falls back to HF_TOKEN env var)")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--resume",          action="store_true",
                   help="Skip prompts that already have results in output_dir")
    p.add_argument("--dtype",           choices=["auto", "bfloat16", "float16"],
                   default="auto",
                   help="'auto' picks bfloat16 when supported, else float16")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Logging
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    return logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Model + tokenizer loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_id: str, dtype_arg: str, hf_token: str | None):
    """
    Load Qwen2.5-Instruct with the correct dtype and configure the tokenizer
    so that pad_token != eos_token (required for unambiguous attention masks).

    BUG FIX: The notebook never loaded the tokenizer in any visible cell, and
    set DTYPE="float16" globally even though Qwen2.5's RMSNorm overflows in
    float16 with eager attention.  We load bfloat16 by default.
    """
    # ── dtype ────────────────────────────────────────────────────────────
    if dtype_arg == "auto":
        load_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dtype_arg == "bfloat16":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float16

    token = hf_token or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        device_map="auto",          # spreads across all available GPUs automatically
        attn_implementation="eager",  # required for output_attentions + 4-D mask hooks
        token=token,
        trust_remote_code=True,
    )
    model.eval()

    # ── BUG FIX: pad_token == eos_token in stock Qwen2.5 ────────────────
    # HuggingFace cannot auto-infer the attention mask when these are equal,
    # which silently corrupts position indices.  Add a dedicated pad token.
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    # ── Collect all valid stop IDs ────────────────────────────────────────
    # BUG FIX: Without explicit eos_token_id the model runs to max_new_tokens
    # even after emitting <|im_end|>, producing garbage trailing tokens.
    stop_ids = []
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    if tokenizer.eos_token_id not in stop_ids:
        stop_ids.append(tokenizer.eos_token_id)
    eos_ids = sorted(set(stop_ids))

    return model, tokenizer, eos_ids


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Prompt helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_input_device(model) -> torch.device:
    """Return the device of the first parameter — correct even with device_map='auto'."""
    return next(model.parameters()).device


def encode_prompt(prompt_text: str, tokenizer, model) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrap prompt_text in the Qwen2.5-Instruct chat template and tokenise.
    Returns (input_ids, attention_mask), both shaped [1, seq_len].

    BUG FIX: The notebook defined encode_prompt *twice* with conflicting logic
    (one version inside the GENERATE_KWARGS cell, another outside it).  This
    single definition is authoritative.  It always uses the chat template so
    the model enters generation in instruction-following mode.
    """
    device = get_input_device(model)
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. "
            "Answer questions using only the provided text. "
            "Be concise and exact."
        )},
        {"role": "user", "content": prompt_text},
    ]
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(
        templated,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,   # template already added them
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def safe_generate(input_ids, attention_mask, model, tokenizer, eos_ids,
                  max_new_tokens, extra_kwargs=None):
    """
    Wrapper around model.generate() that:
      • always passes an explicit attention_mask
      • uses correct eos_token_id (im_end + endoftext)
      • never passes sampling params (top_p/top_k/temp) when greedy

    BUG FIX: notebook passed sampling params with do_sample=False, causing
    HuggingFace warnings and occasionally confusing the sampler.
    """
    kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
        **(extra_kwargs or {}),
    )
    with torch.no_grad():
        return model.generate(input_ids, attention_mask=attention_mask, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Prompt generation (NIAH)
# ═══════════════════════════════════════════════════════════════════════════

HAYSTACK = (
    "The researchers published their findings after months of analysis. "
    "Several key variables were identified that influenced the outcome significantly. "
    "The team conducted follow-up experiments to verify reproducibility across sites. "
    "Equipment calibration was performed at the start of each measurement session. "
    "Statistical significance was determined using a two-tailed t-test with α = 0.05. "
)


def make_needle_uuid():
    uid = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    needle   = f"The secret activation code is: {uid}."
    question = "What is the secret activation code?"
    return needle, question, uid


def make_needle_kv():
    val = random.randint(1000, 9999)
    key = random.choice(["system_port", "access_token", "threshold_value", "batch_index"])
    needle   = f"{key} = {val}"
    question = f"What is the value of {key}?"
    return needle, question, str(val)


NEEDLE_FACTORIES = [make_needle_uuid, make_needle_kv]


def build_prompt(target_tokens: int, depth_frac: float,
                 tokenizer, model,
                 needle_factory=None) -> dict:
    if needle_factory is None:
        needle_factory = random.choice(NEEDLE_FACTORIES)
    needle_text, question, reference = needle_factory()

    question_suffix = f"\n\nQuestion: {question}\nAnswer (be concise):"
    q_tokens      = len(tokenizer.encode(question_suffix, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(needle_text,     add_special_tokens=False))
    hay_budget    = target_tokens - q_tokens - needle_tokens - 10  # slack

    hay = HAYSTACK
    while len(tokenizer.encode(hay, add_special_tokens=False)) < hay_budget:
        hay += HAYSTACK

    hay_ids = tokenizer.encode(hay, add_special_tokens=False)[:hay_budget]
    hay = tokenizer.decode(hay_ids)

    words     = hay.split()
    insert_at = max(0, int(depth_frac * len(words)))
    words.insert(insert_at, needle_text)
    hay_with_needle = " ".join(words)

    prompt = hay_with_needle + question_suffix
    actual = encode_prompt(prompt, tokenizer, model)[0].shape[1]
    return {
        "prompt":    prompt,
        "reference": reference,
        "needle":    needle_text,
        "question":  question,
        "metadata":  {
            "target_tokens": target_tokens,
            "depth_frac":    depth_frac,
            "actual_tokens": actual,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Metrics
# ═══════════════════════════════════════════════════════════════════════════

def answer_correct(generated: str, reference: str) -> bool:
    return reference.strip().lower() in generated.strip().lower()


def compute_eucr(chunk_importances, threshold_fracs=(0.01, 0.05, 0.10)):
    """EUCR@λ = fraction of chunks whose normalised importance exceeds λ."""
    arr   = np.array(chunk_importances, dtype=float)
    total = arr.sum()
    if total == 0:
        return {str(lam): 0.0 for lam in threshold_fracs}
    norm  = arr / total
    return {f"{lam:.2f}": float((norm > lam).mean()) for lam in threshold_fracs}


def compute_pwup(chunk_importances):
    """Position-Weighted Utilization Profile: B/M/E thirds."""
    arr   = np.array(chunk_importances, dtype=float)
    n     = len(arr)
    total = arr.sum()
    if total == 0:
        return {"B": 0.0, "M": 0.0, "E": 0.0}
    norm  = arr / total
    t1, t2 = n // 3, 2 * n // 3
    return {
        "B": float(norm[:t1].mean()),
        "M": float(norm[t1:t2].mean()),
        "E": float(norm[t2:].mean()),
    }


def compute_gud(chunk_importances):
    """
    Gradient Utilization Decay — Spearman correlation between chunk index
    and importance.  Negative = recency bias; positive = primacy bias.
    """
    from scipy.stats import spearmanr
    arr = np.array(chunk_importances, dtype=float)
    if len(arr) < 3 or arr.std() == 0:
        return float("nan")
    r, _ = spearmanr(np.arange(len(arr)), arr)
    return float(r)


def needle_chunk_rank(chunk_importances, needle_chunk_idx):
    """Rank of the needle chunk by descending importance (1 = most important)."""
    arr  = np.array(chunk_importances, dtype=float)
    rank = int((-arr).argsort().tolist().index(needle_chunk_idx)) + 1
    return rank


def find_needle_chunk(ids_list, needle_tok_ids, chunk_size, seq_len):
    """Return chunk index containing the needle, falling back to midpoint."""
    needle_start = next(
        (i for i in range(len(ids_list) - len(needle_tok_ids) + 1)
         if ids_list[i : i + len(needle_tok_ids)] == needle_tok_ids),
        seq_len // 2,
    )
    return needle_start // chunk_size


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Substitution filler setup (lazy, needs tokenizer)
# ═══════════════════════════════════════════════════════════════════════════

def get_filler_id(tokenizer):
    """
    BUG FIX: The notebook used pad_token_id (= <|endoftext|>) as filler, which
    causes hidden-state collapse because the model recognises it as end-of-doc.
    We use '.' — a high-frequency, semantically neutral content token.
    """
    return tokenizer.encode(".", add_special_tokens=False)[0]


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Method A — Substitution ablation
# ═══════════════════════════════════════════════════════════════════════════

def substitution_ablation(prompt_record, model, tokenizer, eos_ids,
                           chunk_size, max_new_tokens):
    prompt    = prompt_record["prompt"]
    reference = prompt_record["reference"]
    needle    = prompt_record["needle"]
    filler_id = get_filler_id(tokenizer)

    ids, mask = encode_prompt(prompt, tokenizer, model)
    seq_len   = ids.shape[1]

    needle_tok_ids   = tokenizer.encode(needle, add_special_tokens=False)
    needle_chunk_idx = find_needle_chunk(ids[0].tolist(), needle_tok_ids, chunk_size, seq_len)
    num_chunks       = (seq_len + chunk_size - 1) // chunk_size

    base_out      = safe_generate(ids, mask, model, tokenizer, eos_ids, max_new_tokens)
    baseline_text = tokenizer.decode(base_out[0, seq_len:], skip_special_tokens=True).strip()
    baseline_ok   = answer_correct(baseline_text, reference)

    chunk_importances = []
    results           = []

    for ci in range(num_chunks):
        s      = ci * chunk_size
        e      = min(s + chunk_size, seq_len)
        ablated = ids.clone()
        ablated[0, s:e] = filler_id   # neutral '.' filler; positions unchanged

        out     = safe_generate(ablated, mask, model, tokenizer, eos_ids, max_new_tokens)
        gen     = tokenizer.decode(out[0, seq_len:], skip_special_tokens=True).strip()
        correct = answer_correct(gen, reference)
        changed = gen != baseline_text
        imp     = 1.0 if (baseline_ok and not correct) else (0.5 if changed else 0.0)
        chunk_importances.append(imp)
        results.append({
            "chunk_idx": ci, "chunk_start": s, "chunk_end": e,
            "output": gen, "correct": correct, "answer_changed": changed,
        })

    return {
        "method":             "substitution",
        "baseline_text":      baseline_text,
        "baseline_correct":   baseline_ok,
        "chunk_importances":  chunk_importances,
        "needle_chunk_idx":   needle_chunk_idx,
        "eucr":               compute_eucr(chunk_importances),
        "pwup":               compute_pwup(chunk_importances),
        "gud":                compute_gud(chunk_importances),
        "needle_rank":        needle_chunk_rank(chunk_importances, needle_chunk_idx),
        "ablation_results":   results,
        "metadata":           prompt_record["metadata"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Method B — InputXGrad
# ═══════════════════════════════════════════════════════════════════════════

def inputxgrad_importance(prompt_record, model, tokenizer, eos_ids,
                           chunk_size, max_new_tokens):
    prompt    = prompt_record["prompt"]
    reference = prompt_record["reference"]
    needle    = prompt_record["needle"]
    device    = get_input_device(model)

    ids, mask = encode_prompt(prompt, tokenizer, model)
    seq_len   = ids.shape[1]

    base_out    = safe_generate(ids, mask, model, tokenizer, eos_ids, max_new_tokens)
    answer_ids  = base_out[0, seq_len:]
    full_ids    = torch.cat([ids[0], answer_ids]).unsqueeze(0)
    full_mask   = torch.ones(1, full_ids.shape[1], dtype=torch.long, device=device)

    needle_tok_ids   = tokenizer.encode(needle, add_special_tokens=False)
    needle_chunk_idx = find_needle_chunk(ids[0].tolist(), needle_tok_ids, chunk_size, seq_len)

    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(full_ids).detach().requires_grad_(True)

    outputs = model(inputs_embeds=embeds, attention_mask=full_mask)
    logits  = outputs.logits   # [1, full_len, vocab]

    answer_logits = logits[0, seq_len - 1 : -1, :]
    loss = torch.nn.functional.cross_entropy(answer_logits, answer_ids)
    loss.backward()

    with torch.no_grad():
        importance = (embeds.grad[0, :seq_len] * embeds[0, :seq_len]).norm(dim=-1)
        importance = importance.cpu().float().numpy()

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunk_importances = [
        float(importance[ci * chunk_size : min((ci + 1) * chunk_size, seq_len)].mean())
        for ci in range(num_chunks)
    ]

    baseline_text    = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    baseline_correct = answer_correct(baseline_text, reference)

    return {
        "method":             "inputxgrad",
        "baseline_text":      baseline_text,
        "baseline_correct":   baseline_correct,
        "chunk_importances":  chunk_importances,
        "token_importances":  importance.tolist(),
        "needle_chunk_idx":   needle_chunk_idx,
        "eucr":               compute_eucr(chunk_importances),
        "pwup":               compute_pwup(chunk_importances),
        "gud":                compute_gud(chunk_importances),
        "needle_rank":        needle_chunk_rank(chunk_importances, needle_chunk_idx),
        "metadata":           prompt_record["metadata"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 10. Method C — Attention aggregation
# ═══════════════════════════════════════════════════════════════════════════

def attention_aggregation(prompt_record, model, tokenizer, eos_ids,
                           chunk_size, max_new_tokens, use_rollout=False):
    prompt    = prompt_record["prompt"]
    reference = prompt_record["reference"]
    needle    = prompt_record["needle"]
    question  = prompt_record["question"]

    ids, mask = encode_prompt(prompt, tokenizer, model)
    seq_len   = ids.shape[1]

    q_toks   = tokenizer.encode(question, add_special_tokens=False)
    q_len    = len(q_toks)
    q_start  = max(0, seq_len - q_len - 5)
    ctx_len  = q_start

    needle_tok_ids   = tokenizer.encode(needle, add_special_tokens=False)
    needle_chunk_idx = find_needle_chunk(ids[0].tolist(), needle_tok_ids, chunk_size, ctx_len)

    with torch.no_grad():
        outputs = model(ids, attention_mask=mask, output_attentions=True)

    # [n_layers, n_heads, seq_len, seq_len]
    all_attn = torch.stack(
        [a.squeeze(0).cpu().float() for a in outputs.attentions]
    )

    if not use_rollout:
        attn_q_to_ctx = all_attn[:, :, q_start:seq_len, :ctx_len]
        attn_to_ctx   = attn_q_to_ctx.mean(dim=(0, 1, 2)).numpy()
    else:
        T       = seq_len
        rollout = torch.eye(T)
        for layer_idx in range(all_attn.shape[0]):
            a = all_attn[layer_idx].mean(dim=0)
            a = a + torch.eye(T)
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)
            rollout = a @ rollout
        attn_to_ctx = rollout[q_start:seq_len, :ctx_len].mean(dim=0).numpy()

    num_chunks = (ctx_len + chunk_size - 1) // chunk_size
    chunk_attn = [
        float(attn_to_ctx[ci * chunk_size : min((ci + 1) * chunk_size, ctx_len)].mean())
        for ci in range(num_chunks)
    ]

    base_out         = safe_generate(ids, mask, model, tokenizer, eos_ids, max_new_tokens)
    baseline_text    = tokenizer.decode(base_out[0, seq_len:], skip_special_tokens=True).strip()
    baseline_correct = answer_correct(baseline_text, reference)

    return {
        "method":             "attn_agg" + ("_rollout" if use_rollout else ""),
        "baseline_text":      baseline_text,
        "baseline_correct":   baseline_correct,
        "chunk_importances":  chunk_attn,
        "needle_chunk_idx":   needle_chunk_idx,
        "eucr":               compute_eucr(chunk_attn),
        "pwup":               compute_pwup(chunk_attn),
        "gud":                compute_gud(chunk_attn),
        "needle_rank":        needle_chunk_rank(chunk_attn, needle_chunk_idx),
        "metadata":           prompt_record["metadata"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 11. Method D — 4-D mask ablation
# ═══════════════════════════════════════════════════════════════════════════

def mask4d_ablation(prompt_record, model, tokenizer, eos_ids,
                    chunk_size, max_new_tokens):
    """
    BUG FIX (major): The notebook's Method D called tokenizer.encode(prompt)
    directly — bypassing the chat template — and used pad_token_id=eos_token_id
    in its baseline generate() call.  Both bugs are fixed here:
      • encode_prompt() is used so the chat template is applied
      • safe_generate() is used for the baseline
      • The 4-D mask + manual decode loop is unchanged (it was correct)
    """
    prompt    = prompt_record["prompt"]
    reference = prompt_record["reference"]
    needle    = prompt_record["needle"]
    device    = get_input_device(model)
    dtype     = next(model.parameters()).dtype

    # BUG FIX: use encode_prompt (chat template) instead of raw encode
    ids, mask = encode_prompt(prompt, tokenizer, model)
    seq_len   = ids.shape[1]

    needle_tok_ids   = tokenizer.encode(needle, add_special_tokens=False)
    needle_chunk_idx = find_needle_chunk(ids[0].tolist(), needle_tok_ids, chunk_size, seq_len)
    num_chunks       = (seq_len + chunk_size - 1) // chunk_size

    # BUG FIX: use safe_generate for baseline (correct eos_ids + pad_token_id)
    base_out         = safe_generate(ids, mask, model, tokenizer, eos_ids, max_new_tokens)
    baseline_text    = tokenizer.decode(base_out[0, seq_len:], skip_special_tokens=True).strip()
    baseline_correct = answer_correct(baseline_text, reference)

    chunk_importances = []
    results           = []

    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, seq_len)

        # 4-D additive mask: causal lower-tri with columns [s:e] blocked.
        # Shape [1, 1, seq_len, seq_len]; additive (0=attend, -inf=block).
        # RoPE is applied to Q and K *before* this mask, so positions are valid.
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        causal[:, s:e] = False
        mask_4d = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
        mask_4d.masked_fill_(~causal, torch.finfo(dtype).min)

        # Prefill with the blocked mask → capture KV cache
        with torch.no_grad():
            prefill = model(
                ids,
                attention_mask=mask_4d,
                use_cache=True,
                return_dict=True,
            )
        past_kv = prefill.past_key_values
        cur_id  = prefill.logits[:, -1:, :].argmax(dim=-1)   # [1, 1]
        generated = [cur_id.item()]

        # Greedy decode from the blocked KV cache
        # (chunk ci is invisible in the cache; no further masking needed)
        for _ in range(max_new_tokens - 1):
            if cur_id.item() in eos_ids:
                break
            with torch.no_grad():
                step    = model(cur_id, past_key_values=past_kv,
                                use_cache=True, return_dict=True)
            past_kv = step.past_key_values
            cur_id  = step.logits[:, -1:, :].argmax(dim=-1)
            generated.append(cur_id.item())

        gen     = tokenizer.decode(generated, skip_special_tokens=True).strip()
        correct = answer_correct(gen, reference)
        changed = gen != baseline_text
        imp     = 1.0 if (baseline_correct and not correct) else (0.5 if changed else 0.0)
        chunk_importances.append(imp)
        results.append({
            "chunk_idx": ci, "chunk_start": s, "chunk_end": e,
            "output": gen, "correct": correct, "answer_changed": changed,
        })

        del past_kv, prefill
        torch.cuda.empty_cache()

    return {
        "method":             "mask4d",
        "baseline_text":      baseline_text,
        "baseline_correct":   baseline_correct,
        "chunk_importances":  chunk_importances,
        "needle_chunk_idx":   needle_chunk_idx,
        "eucr":               compute_eucr(chunk_importances),
        "pwup":               compute_pwup(chunk_importances),
        "gud":                compute_gud(chunk_importances),
        "needle_rank":        needle_chunk_rank(chunk_importances, needle_chunk_idx),
        "ablation_results":   results,
        "metadata":           prompt_record["metadata"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 12. Dispatch table
# ═══════════════════════════════════════════════════════════════════════════

def make_method_fn(model, tokenizer, eos_ids, chunk_size, max_new_tokens):
    """Return a dict mapping method name → callable(prompt_record) -> dict."""
    def sub(p):   return substitution_ablation(p, model, tokenizer, eos_ids, chunk_size, max_new_tokens)
    def ixg(p):   return inputxgrad_importance(p, model, tokenizer, eos_ids, chunk_size, max_new_tokens)
    def attn(p):  return attention_aggregation(p, model, tokenizer, eos_ids, chunk_size, max_new_tokens)
    def m4d(p):   return mask4d_ablation(p,       model, tokenizer, eos_ids, chunk_size, max_new_tokens)
    return {
        "substitution": sub,
        "inputxgrad":   ixg,
        "attn_agg":     attn,
        "mask4d":       m4d,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 13. Visualizations  (saved to disk; no plt.show() on cluster)
# ═══════════════════════════════════════════════════════════════════════════

def make_visualizations(all_results, output_dir, methods, chunk_size, eucr_key="0.05"):
    import pandas as pd

    rows = []
    for method, results_list in all_results.items():
        for r in results_list:
            m    = r.get("metadata", {})
            eucr = r.get("eucr") or {}
            pwup = r.get("pwup") or {}
            rows.append({
                "method":           method,
                "target_tokens":    m.get("target_tokens"),
                "depth_frac":       m.get("depth_frac"),
                "baseline_correct": r.get("baseline_correct"),
                "needle_rank":      r.get("needle_rank"),
                f"EUCR@{eucr_key}": eucr.get(eucr_key, np.nan),
                "PWUP_B":           pwup.get("B", np.nan),
                "PWUP_M":           pwup.get("M", np.nan),
                "PWUP_E":           pwup.get("E", np.nan),
                "GUD":              r.get("gud", np.nan),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    plt.rcParams.update({"font.size": 10,
                         "axes.spines.top": False,
                         "axes.spines.right": False})

    # ── Fig 1: Needle rank by depth × method ─────────────────────────────
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]
        for tl, grp in sub.groupby("target_tokens"):
            grp_s = grp.sort_values("depth_frac")
            ax.plot(grp_s["depth_frac"], grp_s["needle_rank"], marker="o", label=f"{tl} tok")
        ax.set_title(method)
        ax.set_xlabel("Needle depth (fraction)")
        ax.set_ylabel("Needle chunk rank (lower = better)")
        ax.legend(fontsize=8)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    plt.suptitle("Needle chunk rank vs depth — lower = model correctly identifies needle", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "needle_rank_by_depth.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: PWUP profiles ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    regions = ["PWUP_B", "PWUP_M", "PWUP_E"]
    labels  = ["Beginning", "Middle", "End"]
    x = np.arange(3)
    for ax, method in zip(axes, methods):
        sub   = df[df["method"] == method]
        means = [sub[r].mean() for r in regions]
        bars  = ax.bar(x, means, color=["#B5D4F4", "#378ADD", "#0C447C"])
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(method); ax.set_ylabel("Mean normalised importance")
        ax.set_ylim(0, max(means) * 1.3 + 0.01)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("PWUP — where in the context does each method find importance?", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pwup_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: EUCR vs context length ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        sub = df[df["method"] == method]
        grp = sub.groupby("target_tokens")[f"EUCR@{eucr_key}"].mean()
        ax.plot(grp.index, grp.values, marker="o", label=method)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel(f"EUCR @ λ={eucr_key}")
    ax.set_title("Effective context utilisation ratio vs context length")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eucr_vs_length.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 4: GUD ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    method_means = df.groupby("method")["GUD"].mean()
    colors = ["#5DCAA5" if v > 0 else "#D85A30" for v in method_means.values]
    ax.bar(method_means.index, method_means.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("GUD (Spearman corr: position vs importance)")
    ax.set_title("GUD: positive = primacy bias, negative = recency bias")
    for i, (name, val) in enumerate(method_means.items()):
        ax.text(i, val + (0.005 if val >= 0 else -0.02), f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gud_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 5: InputXGrad token-level heatmap ─────────────────────────────
    if "inputxgrad" in all_results and all_results["inputxgrad"]:
        shown = {}
        for r in all_results["inputxgrad"]:
            tl = r["metadata"]["target_tokens"]
            if tl not in shown:
                shown[tl] = r
        n = len(shown)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
        if n == 1:
            axes = [axes]
        for ax, (tl, r) in zip(axes, sorted(shown.items())):
            imps      = np.array(r["token_importances"])
            imps_norm = (imps - imps.min()) / (imps.max() - imps.min() + 1e-9)
            ax.imshow(imps_norm.reshape(1, -1), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_yticks([])
            ax.set_xlabel("Token position")
            ax.set_title(
                f"InputXGrad token importance — {tl} tokens  "
                f"(needle chunk {r['needle_chunk_idx']}  rank {r['needle_rank']})"
            )
            nc = r["needle_chunk_idx"]
            s  = nc * chunk_size
            e  = min(s + chunk_size, len(imps))
            ax.axvline(s, color="blue", linewidth=1.5, label="needle start")
            ax.axvline(e, color="blue", linewidth=1.5, linestyle="--")
        plt.suptitle("InputXGrad: per-token importance (red = high)", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "inputxgrad_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary = (
        df.groupby(["method", "target_tokens"])
          .agg(
              baseline_acc=("baseline_correct", "mean"),
              needle_rank=(  "needle_rank",      "mean"),
              eucr=(         f"EUCR@{eucr_key}", "mean"),
              pwup_b=(       "PWUP_B",           "mean"),
              pwup_m=(       "PWUP_M",           "mean"),
              pwup_e=(       "PWUP_E",           "mean"),
              gud=(          "GUD",              "mean"),
          )
          .round(3)
    )
    summary_file = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_file)
    print(summary.to_string())
    return summary_file


# ═══════════════════════════════════════════════════════════════════════════
# 14. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    logger = setup_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("CS 498 Context Utilization Experiment — cluster edition")
    logger.info("=" * 60)
    logger.info(f"model        : {args.model_id}")
    logger.info(f"methods      : {args.methods}")
    logger.info(f"ctx lengths  : {args.context_lengths}")
    logger.info(f"needle depths: {args.needle_depths}")
    logger.info(f"num_examples : {args.num_examples}")
    logger.info(f"chunk_size   : {args.chunk_size}")
    logger.info(f"output_dir   : {args.output_dir}")

    # ── GPU check ─────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found.  "
            "On a SLURM cluster, request a GPU with e.g. --gres=gpu:1"
        )
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        logger.info(f"GPU {i}: {props.name}  {total/1e9:.1f} GB total  {free/1e9:.1f} GB free")

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading model and tokenizer …")
    model, tokenizer, eos_ids = load_model_and_tokenizer(
        args.model_id, args.dtype, args.hf_token
    )
    logger.info(f"Model loaded — param dtype: {next(model.parameters()).dtype}")
    logger.info(f"pad_token_id : {tokenizer.pad_token_id}  ({tokenizer.pad_token!r})")
    logger.info(f"eos_ids      : {eos_ids}")
    logger.info(f"VRAM used    : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Smoke test ────────────────────────────────────────────────────────
    logger.info("Running smoke test …")
    test_ids, test_mask = encode_prompt(
        "The secret code is ALPHA-7. What is the secret code?",
        tokenizer, model,
    )
    test_out = safe_generate(test_ids, test_mask, model, tokenizer, eos_ids,
                              max_new_tokens=32)
    test_ans = tokenizer.decode(test_out[0, test_ids.shape[1]:], skip_special_tokens=True).strip()
    if "ALPHA-7" not in test_ans:
        raise RuntimeError(
            f"Smoke test FAILED — got: {test_ans!r}\n"
            "Check that MODEL_ID ends in -Instruct and bfloat16 loaded correctly."
        )
    logger.info(f"Smoke test passed — answer: {test_ans!r}")

    # ── Generate prompts ──────────────────────────────────────────────────
    prompts_file = os.path.join(args.output_dir, "prompts.jsonl")
    if args.resume and os.path.exists(prompts_file):
        logger.info(f"Loading existing prompts from {prompts_file}")
        with open(prompts_file) as f:
            all_prompts = [json.loads(line) for line in f]
    else:
        logger.info("Generating prompts …")
        random.seed(args.seed)
        all_prompts = []
        for tl in args.context_lengths:
            for d in args.needle_depths:
                for _ in range(args.num_examples):
                    all_prompts.append(build_prompt(tl, d, tokenizer, model))
        with open(prompts_file, "w") as f:
            for p in all_prompts:
                f.write(json.dumps(p) + "\n")
        logger.info(f"Generated {len(all_prompts)} prompts → {prompts_file}")

    total_configs = len(args.context_lengths) * len(args.needle_depths) * args.num_examples
    logger.info(
        f"Total configs: {len(args.context_lengths)} lengths × "
        f"{len(args.needle_depths)} depths × {args.num_examples} examples "
        f"= {total_configs} prompts"
    )

    # ── Load existing results (resume) ────────────────────────────────────
    results_file = os.path.join(args.output_dir, "all_results.json")
    if args.resume and os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
        logger.info(f"Resuming — loaded partial results from {results_file}")
    else:
        all_results = {m: [] for m in args.methods}

    # Build lookup of already-completed (prompt_idx, method) pairs
    done = {
        (r["prompt_idx"], method)
        for method, results_list in all_results.items()
        for r in results_list
        if "prompt_idx" in r
    }

    # ── Method dispatch ───────────────────────────────────────────────────
    method_fn = make_method_fn(model, tokenizer, eos_ids,
                                args.chunk_size, args.max_new_tokens)

    # ── Main experiment loop ──────────────────────────────────────────────
    for prompt_idx, p in enumerate(all_prompts):
        tl = p["metadata"]["target_tokens"]
        d  = p["metadata"]["depth_frac"]
        logger.info(
            f"\n[{prompt_idx + 1}/{len(all_prompts)}] "
            f"len={tl}  depth={d:.2f}  ref={p['reference']!r}"
        )

        for method in args.methods:
            if (prompt_idx, method) in done:
                logger.info(f"  [{method:14s}]  (skipped — already done)")
                continue

            t0 = time.time()
            try:
                res = method_fn[method](p)
                res["prompt_idx"] = prompt_idx
                all_results[method].append(res)
                elapsed = time.time() - t0
                logger.info(
                    f"  [{method:14s}]  "
                    f"needle_rank={res.get('needle_rank', '?')}  "
                    f"baseline_correct={res.get('baseline_correct', '?')}  "
                    f"({elapsed:.1f}s)"
                )
            except Exception as ex:
                logger.exception(f"  [{method:14s}]  ERROR: {ex}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        # Checkpoint after every prompt
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    logger.info(f"\nAll results saved → {results_file}")

    # ── Visualizations + summary ──────────────────────────────────────────
    logger.info("Generating visualizations …")
    try:
        import pandas as pd  # only needed here
        summary_file = make_visualizations(
            all_results, args.output_dir, args.methods, args.chunk_size
        )
        logger.info(f"Summary CSV → {summary_file}")
        logger.info(f"Figures saved to {args.output_dir}/")
    except ImportError:
        logger.warning("pandas not available — skipping visualizations")

    logger.info("Done.")


if __name__ == "__main__":
    main()