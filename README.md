# Emergence Experiment - Run Guide

Pipeline for testing whether apparent emergent arithmetic ability reflects a
measurement artifact, a genuine mechanistic transition, or both.

## What changed in this version

Four substantive upgrades over the first version, each closing a weakness
that a reviewer would otherwise catch:

1. **Step-depth tiers k=1..5** (was k=1,2). The compounding prediction p^k
   only becomes sharply falsifiable across a range of depths. At p=0.7 the
   old design predicted 0.70 vs 0.49; the new one predicts 0.70 vs 0.17.
2. **Few-shot prompting** (`--shots`). Small models previously scored 0%
   partly because they did not recognise the task format at all (they
   continued with LaTeX and Python). Few-shot separates "cannot compute"
   from "does not know what is being asked", and makes the induction-head
   probe genuinely task-relevant.
3. **Wilson confidence intervals** on every accuracy point. With greedy
   decoding, repeat sampling gives zero variance, so the meaningful
   uncertainty is item-level binomial - now computed and plotted.
4. **Fair curve comparison.** The old power-law form was unbounded while
   accuracy is bounded in [0,1], so it lost to the sigmoid almost by
   construction. Replaced with a bounded saturating form. Verified on
   synthetic smooth data, where the old code called everything "sigmoid"
   and the new code correctly returns "smooth"/"inconclusive".

Also added: partial-step error detection (direct evidence of partial
algorithm acquisition), a format-specific induction probe, duplicate-row
protection, and `synthesize.py` for the final pre-registered verdict.

---

## Setup (on every machine)

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Generate the benchmark (ONCE, then copy the file)

```bash
python benchmark.py
```

Produces `data/tasks.json`: 180 items across 9 tiers, every answer
self-validated. **Copy this exact file to every machine** rather than
regenerating it, so all models see identical problems.

## 2. Inference (split across machines by RAM)

| Model | RAM | Machine |
|---|---|---|
| pythia-70m / 160m / 410m | <2 GB | your laptop |
| pythia-1b | ~5 GB | laptop or friend's |
| pythia-1.4b | ~7 GB | friend's machine |
| pythia-2.8b | ~14 GB | university system |

Run **both** prompt conditions for every model - the zero-shot vs few-shot
comparison is itself a result:

```bash
python inference.py --model EleutherAI/pythia-70m --shots 0
python inference.py --model EleutherAI/pythia-70m --shots 3
```

Re-running the same model appends duplicate rows; add `--overwrite` to
replace that model's previous rows instead.

Merging results from several machines:

```bash
cat laptop.jsonl friend.jsonl university.jsonl > results/raw_outputs.jsonl
```

## 3. Score errors

```bash
python scoring.py
```

Watch the `partial_step` column - it counts cases where the model computed
part of a multi-step expression correctly, which is the cleanest behavioral
evidence of partial algorithm acquisition.

## 4. Behavioral analysis

```bash
python analysis.py --shots 3
```

Gives per-tier accuracy with 95% CIs, the compounding-probability test, BIC
curve-shape classification, and plots in `results/plots/`.

## 5. Mechanistic probe

```bash
python induction_probe.py --model pythia-70m
python induction_probe.py --model pythia-1.4b
```

Note: transformer_lens uses short names (`pythia-70m`), while inference.py
uses HF repo paths (`EleutherAI/pythia-70m`). This is a library convention
difference, not an error.

## 6. Final verdict

```bash
python synthesize.py --shots 3
```

Applies the pre-registered decision rules (thresholds are constants at the
top of `synthesize.py` - state them in your methods section) and reports
mirage / mechanism / mixed / null.

---

## Verified in testing

- `benchmark.py`: 180 items generated, all ground truth self-validated,
  depth declarations cross-checked against operator counts.
- `analysis.py`: on synthetic data constructed to be *exactly* p^k
  compounding with smooth underlying scaling, the compounding test returned
  mean |residual| = 0.017 with the prediction inside the 95% CI for 24/24
  points, and the curve classifier no longer defaulted to "sigmoid".
- `scoring.py`: all eight error categories verified against hand-checked
  cases, including the operand-vs-partial-step distinction.
- `inference.py`, `induction_probe.py`, `synthesize.py`: syntax-verified.
  Not run end-to-end here (no Hugging Face access in the build sandbox).

## Known limitations to state in the paper

- **Induction heads are a proxy, not a direct measure of arithmetic
  ability.** They are established as the mechanism behind in-context
  pattern continuation (Olsson et al., 2022), not arithmetic computation.
  The format-induction probe narrows this gap but does not close it. A null
  mechanistic result means "this circuit did not change", not "nothing
  internal changed".
- **Scale ceiling.** Widely-cited emergent transitions occur beyond 10B
  parameters. Scope claims to early-stage scaling dynamics.
- **20 items per tier** gives fairly wide Wilson intervals. Raise with
  `--per_tier 40` if compute allows; report the CIs either way.
- **Step-independence assumption.** The p^k prediction assumes per-step
  errors are independent. Correlated errors would bias the prediction; the
  partial-step error counts give a partial check on this.

---

## Working across multiple machines

Each machine writes to its own results file so nothing collides in git:

```bash
# primary machine
python inference.py --model EleutherAI/pythia-1b --shots 3 --overwrite

# collaborator's machine (bigger RAM)
python inference.py --model EleutherAI/pythia-1.4b --shots 3 --overwrite --out raw_outputs_big.jsonl
```

`scoring.py` automatically reads **every** `results/raw_outputs*.jsonl`, so
after pulling a collaborator's commit you just run `python scoring.py` and
their data is included. No manual merging.

`data/tasks.json` is committed and must be identical on every machine — do not
regenerate it once runs have started, or results stop being comparable.

See `COLLABORATOR.md` for the instructions to hand to whoever runs the large
models.
