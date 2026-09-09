"""
induction_probe.py
Mechanistic half of the study (RQ2).

Two complementary probes are run per model:

  (A) GENERIC INDUCTION SCORE - the standard repeated-random-token test
      (Olsson et al., 2022). Detects heads that attend from a repeated
      token back to the token that FOLLOWED its first occurrence.
      This is the field-standard operationalisation and makes results
      comparable with prior work.

  (B) FORMAT INDUCTION SCORE - the same measurement, but on prompts with
      the actual few-shot arithmetic FORMAT ("3 + 4 = 7\n12 + 25 = 37\n...").

  Why (B) exists, and why it matters for the paper's claim:
  induction heads are established as the mechanism behind in-context
  PATTERN CONTINUATION, not arithmetic computation as such. A purely
  generic induction score therefore risks measuring a circuit that has
  little to do with the arithmetic accuracy curve, which would make any
  claimed alignment (or non-alignment) between the two hard to interpret.
  Probe (B) measures induction specifically over the task's own format,
  which is the mechanism plausibly gating few-shot arithmetic performance.

  Reporting BOTH, and being explicit that induction heads are a proxy for
  in-context format acquisition rather than for arithmetic ability itself,
  is the honest framing for the write-up.

Usage:
    python induction_probe.py --model pythia-70m
    python induction_probe.py --model pythia-1.4b

Note the model name here is the transformer_lens SHORT name
("pythia-70m"), not the Hugging Face repo path used by inference.py.
"""

import json
import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUT_PATH = RESULTS_DIR / "induction_scores.jsonl"


def generic_induction_scores(model, seq_len=30, batch_size=6, device="cpu"):
    """Standard repeated-random-token induction test -> (n_layers, n_heads)."""
    import torch

    vocab = model.cfg.d_vocab
    half = torch.randint(100, vocab - 100, (batch_size, seq_len), device=device)
    seq = torch.cat([half, half], dim=1)

    _, cache = model.run_with_cache(seq)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        attn = cache["pattern", layer]  # (batch, head, dest, src)
        vals = []
        for i in range(seq_len - 1):
            dest = seq_len + i      # position in the repeated half
            src = i + 1             # token after the first occurrence
            vals.append(attn[:, :, dest, src])       # (batch, head)
        scores[layer] = torch.stack(vals, 0).mean(0).mean(0)
    return scores


def format_induction_scores(model, tokenizer_text, device="cpu"):
    """
    Induction score measured over the task's own few-shot format.
    We build a prompt of repeated "<expr> = <ans>" lines and measure, at
    each '=' token, how much attention goes to the position just after a
    previous '=' -- i.e. the model using earlier examples to work out that
    a number follows '='.
    """
    import torch

    tokens = model.to_tokens(tokenizer_text)
    _, cache = model.run_with_cache(tokens)

    str_tokens = model.to_str_tokens(tokenizer_text)
    eq_positions = [i for i, t in enumerate(str_tokens) if t.strip() == "="]
    if len(eq_positions) < 2:
        return None

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        attn = cache["pattern", layer][0]  # (head, dest, src)
        vals = []
        for idx, dest in enumerate(eq_positions[1:], start=1):
            # attend back to the token right after each EARLIER '='
            for prev in eq_positions[:idx]:
                src = prev + 1
                if src < attn.shape[-1]:
                    vals.append(attn[:, dest, src])   # (head,)
        if vals:
            scores[layer] = torch.stack(vals, 0).mean(0)
    return scores


def probe(model_name, threshold):
    import torch
    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} into transformer_lens on {device} ...")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    generic = generic_induction_scores(model, device=device)

    fewshot_text = (
        "3 + 4 = 7\n"
        "12 + 25 = 37\n"
        "(5 + 6) * 2 = 22\n"
        "8 + 9 ="
    )
    fmt = format_induction_scores(model, fewshot_text, device=device)

    result = {
        "model": model_name,
        "threshold": threshold,
        "generic_max_score": float(generic.max()),
        "generic_best_layer_head": [int(x) for x in (generic == generic.max()).nonzero()[0]],
        "generic_n_heads_above_threshold": int((generic > threshold).sum()),
        "generic_has_induction_head": bool((generic > threshold).sum() > 0),
        "generic_all_scores": generic.tolist(),
    }
    if fmt is not None:
        result.update({
            "format_max_score": float(fmt.max()),
            "format_best_layer_head": [int(x) for x in (fmt == fmt.max()).nonzero()[0]],
            "format_n_heads_above_threshold": int((fmt > threshold).sum()),
            "format_has_induction_head": bool((fmt > threshold).sum() > 0),
            "format_all_scores": fmt.tolist(),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"  generic induction : max={result['generic_max_score']:.3f} "
          f"at layer/head {result['generic_best_layer_head']}  "
          f"heads>{threshold}: {result['generic_n_heads_above_threshold']}")
    if fmt is not None:
        print(f"  format  induction : max={result['format_max_score']:.3f} "
              f"at layer/head {result['format_best_layer_head']}  "
              f"heads>{threshold}: {result['format_n_heads_above_threshold']}")
    print(f"  appended to {OUT_PATH}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="transformer_lens name, e.g. pythia-70m")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="induction score above which a head counts as an induction head "
                         "(0.4 is a common convention; report your threshold in the paper "
                         "and check robustness at 0.3 and 0.5)")
    args = ap.parse_args()
    probe(args.model, args.threshold)


if __name__ == "__main__":
    main()
