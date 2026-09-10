# Experiment Log

Chronological record of what was run, what it showed, and why the design
changed. Kept so that every methodological decision has a stated reason and
a dated result behind it, rather than being justified after the fact.

Each entry: **what changed → why → what it showed**.

---

## v1 — Initial benchmark (180 items, 5 tiers)

**Design.** 100 items, later 180: `single_digit`, `two_digit`, `three_digit`,
`carrying`, `multi_step`. Few-shot exemplars `3+4=7`, `12+25=37`,
`(5+6)*2=22`. Models 70M–1B on local CPU.

**Bug found: log-probability silently dropped.**
81% of `log_prob` values were `null`. Cause: the prompt ends in a space
(`"2 + 1 = "`), and BPE merges that trailing space with the answer's first
digit into one token. Comparing prompt-only vs prompt+answer token counts
therefore gave an answer length of 0, and the code returned `None`.
*Fix:* strip the trailing space before computing the baseline token count.
*Impact:* exact-match accuracy was unaffected; all log-prob analysis before
this point was invalid.

**Bug found: results appended across re-runs.**
Re-running a model appended a second copy of its rows rather than replacing
them, so one file held both the buggy and fixed runs (1000 rows where 500
were expected).
*Fix:* `--overwrite` flag that clears prior rows for a (model, shots) pair.

**Result.** 70M–410M scored 0/180. 1B scored 7/180 (3-shot) and 1/180
(0-shot). Generations at 70M reproduced the few-shot *format* exactly
(`answer\nnext expression =`) while getting every answer wrong.

---

## v2 — Exemplar contamination discovered

**Why the change.** A check of every "correct" answer showed 5 of 8 matched
an exemplar answer (7, 37, 22). Worse, the benchmark contained
`4 + 3 = 7` while an exemplar was `3 + 4 = 7` — the same operands, commuted.
The model was echoing, and the echo was being scored as correct.

**Change 1.** Exemplars replaced with values absent from the benchmark's
answer set: `23+46=69`, `34+94=128`, `(22+39)*4=244`.
**Result.** Accuracy across 70M–410M dropped from 8 correct to **0**.

**Change 2 (over-correction, then fixed).** The replacement exemplars were
all large (median answer 128) while `single_digit` answers span 2–18. This
primes the model toward large outputs and biases *against* the small-answer
tiers — the mirror image of the original problem.
*Fix:* a magnitude-balanced set, still collision-free:
`4+1=5`, `21+69=90`, `(19+32)*3=153`.

**Methodological finding worth reporting.** Few-shot exemplar choice
contaminated accuracy through two distinct mechanisms — direct answer
echoing, and magnitude priming that made small guesses land correctly. With
the original exemplars, apparent accuracy was 8/180; with collision-free
exemplars it was 0/180. Exemplar selection is not a neutral implementation
detail in small-model arithmetic evaluation.

---

## v3 — Step-depth axis added (180 → 220 items, 11 tiers)

**Why the change.** With only k=1 and k=2, the compounding prediction
`p^k` barely bends: at p=0.7 it predicts 0.70 vs 0.49. That is nearly
unfalsifiable. Across k=1..5 the same p predicts 0.70 vs 0.17 — a sharp,
testable separation.

**Change.** Two independent axes:
- *step depth* (`depth_1`..`depth_5`), operand sizes held roughly constant
  so only step count varies;
- *surface difficulty* (`single_digit`, `two_digit`, `three_digit`,
  `carrying`), all k=1, so digit count varies but step count does not.

**Change.** `--shots` flag added (0 and 3 run for every model). Zero-shot
generations at 410M continued into algebra prose and code
(`' -2*r. Let l(a) = -'`), showing a 0% score there reflects not recognising
the task, not inability to compute. Few-shot separates the two.

---

## v4 — Operation-aware compounding test

**Why the change.** The first compounding test estimated a single per-step
accuracy `p` from `depth_1` and predicted `p^k`. But `depth_1` is addition
only, while deeper tiers also require multiplication and subtraction. With
`p_add = 1.000` this predicted 100% accuracy at every depth — a meaningless
comparison across operations of different difficulty, and one that would
have been reported as a dramatic refutation of compounding for the wrong
reason.

**Change.** Added `mult_single` (`a * c`) and `sub_single` (`a - b`) tiers,
and recorded each depth item's operation sequence
(`depth_3 = [add, mult, add]`). The prediction is now the product along the
item's real sequence, e.g. `p_add * p_mult * p_add`.

**Validation.** On synthetic data built as an exact operation-wise product
with deliberately unequal per-operation difficulty (add 0.90, mult 0.50,
sub 0.80), the test returned mean residual 0.010 with every point inside
its 95% CI — i.e. it correctly recognises true compounding when present.

**Bug found: curve-fitting biased toward "sigmoid".**
The original comparison fitted an *unbounded* power law against accuracy,
which is bounded in [0,1]. The unbounded form cannot flatten near the
ceiling and therefore lost to the sigmoid almost by construction — on
synthetic smooth data it labelled every tier "sigmoid".
*Fix:* a bounded saturating form `L(1 - exp(-a x^b))`. The same synthetic
data then returned "smooth" and "inconclusive" at shallow depths.

**Bug found: `operations` never reached the analysis.**
`benchmark.py` wrote the operation sequence into `tasks.json`, but
`inference.py` did not copy it into its output records, so the
operation-aware test printed no rows at all.
*Fix:* `inference.py` now writes it; `scoring.py` also joins it from
`tasks.json`, so existing runs did not need repeating.

**New error category: `dropped_operand`.**
Inspecting `depth_2` failures showed a specific pattern — `(4+19)*5 → 95`,
i.e. `19*5` computed with the `+4` never bound in. Roughly half of all
`depth_2` errors at 1.4B and 2.8B. Previously filed as generic
`magnitude_error`, which obscured the strongest available evidence of
partial algorithm execution.

---

## v5 — Full sweep on GPU (70M–6.9B)

**Why the change.** CPU inference made 1.4B+ impractical. Kaggle's free GPU
runs the whole sweep in ~20 minutes. `--fp16` added so 6.9B fits in 16GB.

**Contamination caught by tooling.** `scoring.py` read both the old
180-item laptop results and the new 220-item GPU results. Task IDs are
reused across benchmark versions, so old rows were scored against the
current `tasks.json` — i.e. against the wrong expressions — and accuracies
were averaged over incompatible runs.
*Fix:* `scoring.py` now hard-fails on duplicate (model, shots, task) keys
rather than merging silently.

**Bug found: fp16 produced NaN attention.**
The induction probe sampled random token IDs up to `config.vocab_size`
(50304), but Pythia's config vocab is padded beyond the tokenizer's real
vocabulary (~50277); those embedding rows are untrained and produced NaN in
half precision. 20–50% of every attention matrix came back NaN.
*Consequence if unfixed:* 70M reported `max=0.063, 0 induction heads`. In
fp32 the same model reports `max=0.817, 5 induction heads`. The fp16 run
would have supported the opposite conclusion.
*Fix:* clamp sampling to the tokenizer's real vocabulary; NaN-safe argmax;
run the probe in fp32 wherever it fits (all models except 6.9B).

**Dependency removed.** `transformer_lens` failed on every Pythia model
(`'GPTNeoXForCausalLM' object has no attribute 'embed_out'` — newer
`transformers` renamed it). Rewritten using `output_attentions=True` from
`transformers` directly, removing the version dependency entirely.

---

## v5 results (clean data, 7 model sizes)

**Single-step, 3-shot.** `depth_1` (addition): 0, 0, 0, 0, 1.00, 1.00, 1.00.
`mult_single`: 0, 0, 0, 0.15, 0.90, 0.95, 1.00.

**Multi-step, 3-shot.** `depth_2`: at most 0.05 anywhere. `depth_3`,
`depth_4`, `depth_5`: exactly 0.000 at all seven sizes.

**Compounding.** At 6.9B, `p_add = 1.00` and `p_mult = 1.00`, so `(a+b)*c`
is predicted at 1.00. Measured 0.00. Mean |residual| = 0.261 over 28 points
against a pre-registered threshold of 0.05; prediction inside the 95% CI for
61% of points against a threshold of 70%. **The depth effect is not
explained by per-step error accumulation.**

**Induction heads.** Present at every scale including 70M (max 0.82,
5 heads above threshold), roughly flat at ~0.95 above that. Arithmetic does
not appear until 1.4B. Onset could not be localised — it lies below the
smallest model tested. **Induction-head formation is not the gate on
arithmetic ability.**

**Format-induction score declines with scale** (0.49 at 70M → ~0.30 from
410M on). Reported with caution: may reflect small models relying on literal
surface copying, or dilution across a larger head count. Needs a control
before being interpreted.

**Verdict against pre-registered rules: MIXED.** Neither existing account
predicts this configuration — per-step accuracy at ceiling, composition at
zero, and the candidate circuit present from the smallest model onward.

---

## v6 — Scratchpad condition (planned)

**Why.** The results establish *that* composition fails, not *why*. The
`dropped_operand` pattern suggests the failure is in binding an intermediate
result into the next operation, rather than in the arithmetic itself. A
prompt that supplies the intermediate result externally tests this directly:

    (4 + 19) * 5. First, 4 + 19 = 23. Then 23 * 5 =

- Correct under scratchpad but wrong without → the arithmetic is available;
  what fails is internal binding of intermediate results.
- Still wrong → the failure is deeper than composition.

Either outcome is informative, and this converts a descriptive result into a
causal one.

**Also planned.** `--per_tier 50` (from 20). Several tiers are claimed to be
exactly zero, but at n=20 the Wilson interval is [0, 0.16] — a true accuracy
of 15% cannot be excluded. At n=50 it tightens to [0, 0.07].