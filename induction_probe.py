"""
induction_probe.py
Mechanistic half of the study (RQ2), using plain `transformers` only.

WHY NOT transformer_lens: TransformerLens' NeoX weight converter reads
`model.embed_out`, which newer versions of `transformers` renamed. That
combination raises
    AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out'
for every Pythia model. Rather than pin fragile versions, this script uses
`output_attentions=True`, which is native to transformers and returns the
same attention matrices the induction score needs.

Two probes per model:

  (A) GENERIC INDUCTION SCORE - standard repeated-random-token test
      (Olsson et al., 2022). For a sequence [X, X], measure how much a head
      at position (len+i) attends to position (i+1): the token that FOLLOWED
      this token's first occurrence.

  (B) FORMAT INDUCTION SCORE - the same idea measured on the study's own
      few-shot arithmetic layout, at each "=" token attending back to the
      token after an earlier "=".
      Included because induction heads are established as the mechanism for
      in-context PATTERN CONTINUATION, not arithmetic computation. A purely
      generic score risks measuring a circuit unrelated to the accuracy
      curve; (B) measures induction over the format the task actually uses.
      Report both, and treat them as evidence about in-context format
      acquisition rather than about arithmetic ability itself.

Usage:
    python induction_probe.py --model EleutherAI/pythia-70m
    python induction_probe.py --model EleutherAI/pythia-6.9b --fp16
"""

import json
import argparse
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).parent / "results"
OUT_PATH = RESULTS_DIR / "induction_scores.jsonl"


def stack_attentions(model, input_ids):
    """-> tensor (n_layers, n_heads, seq, seq) on CPU, float32."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    # out.attentions: tuple of n_layers tensors, each (batch, heads, seq, seq)
    return torch.stack([a[0].float().cpu() for a in out.attentions], dim=0)


def generic_induction(model, device, seq_len=40, vocab_lo=100, vocab_hi=None):
    vocab_hi = vocab_hi or (model.config.vocab_size - 100)
    half = torch.randint(vocab_lo, vocab_hi, (1, seq_len), device=device)
    seq = torch.cat([half, half], dim=1)                    # [X, X]
    attn = stack_attentions(model, seq)                     # (L, H, S, S)

    n_layers, n_heads = attn.shape[0], attn.shape[1]
    scores = torch.zeros(n_layers, n_heads)
    for i in range(seq_len - 1):
        dest = seq_len + i        # position in the repeated half
        src = i + 1               # token after the first occurrence
        scores += attn[:, :, dest, src]
    return scores / (seq_len - 1)


def format_induction(model, tokenizer, device):
    text = ("4 + 1 = 5\n"
            "21 + 69 = 90\n"
            "(19 + 32) * 3 = 153\n"
            "8 + 9 =")
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    toks = tokenizer.convert_ids_to_tokens(ids[0])
    eq_pos = [i for i, t in enumerate(toks) if t.strip().replace("Ġ", "") == "="]
    if len(eq_pos) < 2:
        return None

    attn = stack_attentions(model, ids)
    n_layers, n_heads = attn.shape[0], attn.shape[1]
    scores = torch.zeros(n_layers, n_heads)
    count = 0
    for idx, dest in enumerate(eq_pos[1:], start=1):
        for prev in eq_pos[:idx]:
            src = prev + 1
            if src < attn.shape[-1]:
                scores += attn[:, :, dest, src]
                count += 1
    return scores / count if count else None


def probe(model_name, threshold, fp16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
    print(f"Loading {model_name} on {device} dtype={dtype} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()

    gen = generic_induction(model, device)
    fmt = format_induction(model, tokenizer, device)

    def summarise(scores, prefix):
        if scores is None:
            return {}
        best = (scores == scores.max()).nonzero()[0].tolist()
        return {
            f"{prefix}_max_score": float(scores.max()),
            f"{prefix}_best_layer_head": [int(x) for x in best],
            f"{prefix}_n_heads_above_threshold": int((scores > threshold).sum()),
            f"{prefix}_has_induction_head": bool((scores > threshold).sum() > 0),
            f"{prefix}_all_scores": scores.tolist(),
        }

    result = {"model": model_name, "threshold": threshold,
              "n_layers": int(gen.shape[0]), "n_heads": int(gen.shape[1])}
    result.update(summarise(gen, "generic"))
    result.update(summarise(fmt, "format"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"  generic: max={result['generic_max_score']:.3f} "
          f"at layer/head {result['generic_best_layer_head']}  "
          f"heads>{threshold}: {result['generic_n_heads_above_threshold']}")
    if "format_max_score" in result:
        print(f"  format : max={result['format_max_score']:.3f} "
              f"at layer/head {result['format_best_layer_head']}  "
              f"heads>{threshold}: {result['format_n_heads_above_threshold']}")
    print(f"  appended to {OUT_PATH}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Hugging Face name, e.g. EleutherAI/pythia-70m "
                         "(same format as inference.py)")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="induction score above which a head counts as an induction "
                         "head. 0.4 is a common convention; report the value used and "
                         "check robustness at 0.3 and 0.5.")
    ap.add_argument("--fp16", action="store_true", help="half precision (GPU only)")
    args = ap.parse_args()
    probe(args.model, args.threshold, args.fp16)


if __name__ == "__main__":
    main()