"""
synthesize.py
The final step: joins the behavioral branch (analysis.py) with the
mechanistic branch (induction_probe.py) and evaluates the result against
PRE-REGISTERED decision rules.

Run this LAST, after inference.py, scoring.py and induction_probe.py have
been run for every model size.

The decision rules below are stated in code, before results exist, so the
verdict is derived rather than chosen after seeing the data. State them in
the paper's methods section exactly as they appear here.

Usage:
    python synthesize.py --shots 3
"""

import json
import argparse
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
SCORED_PATH = RESULTS_DIR / "scored_outputs.jsonl"
INDUCTION_PATH = RESULTS_DIR / "induction_scores.jsonl"

# transformer_lens short name -> Hugging Face repo name
NAME_MAP = {
    "Qwen/Qwen3-0.6B-Base": "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base": "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
}

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analysis import MODEL_SIZE_MAP, load, aggregate, wilson_ci  # noqa: E402

# ---- pre-registered thresholds -------------------------------------------
COMPOUNDING_MAE_THRESHOLD = 0.05   # mean |residual| below this = "explained by compounding"
CI_COVERAGE_THRESHOLD = 0.70       # fraction of k>1 points whose CI contains the prediction
ACCURACY_JUMP_THRESHOLD = 0.15     # accuracy increase between adjacent sizes counting as a "jump"
# --------------------------------------------------------------------------


def compounding_summary(agg):
    """
    Operation-aware compounding summary, matching analysis.py exactly.
    (An earlier version used p_add**k, which mixed operations of different
    difficulty and reported a different MAE than analysis.py for the same
    data. Both now estimate one probability per OPERATION and multiply along
    each item's real operation sequence.)
    """
    OP_TIERS = {"add": "depth_1", "mult": "mult_single", "sub": "sub_single"}
    if any(t not in agg for t in OP_TIERS.values()):
        return None

    depth_tiers = sorted([t for t in agg if t.startswith("depth_") and t != "depth_1"],
                         key=lambda t: int(t.split("_")[1]))
    resid, inside, total = [], 0, 0
    for size in sorted(agg["depth_1"].keys()):
        p = {op: agg[t][size]["accuracy"] for op, t in OP_TIERS.items() if size in agg[t]}
        if len(p) < 3:
            continue
        for tier in depth_tiers:
            if size not in agg[tier]:
                continue
            d = agg[tier][size]
            ops = d.get("operations")
            if not ops:
                continue
            pred = 1.0
            for op in ops:
                pred *= p.get(op, 0.0)
            resid.append(abs(d["accuracy"] - pred))
            total += 1
            if d["ci_low"] <= pred <= d["ci_high"]:
                inside += 1
    if not resid:
        return None
    return {"mae": float(np.mean(resid)),
            "ci_coverage": inside / total if total else 0.0,
            "n_points": total}


def find_accuracy_jump(agg, tier):
    """Largest accuracy increase between adjacent model sizes for a tier."""
    if tier not in agg:
        return None
    sizes = sorted(agg[tier])
    best = None
    for a, b in zip(sizes, sizes[1:]):
        delta = agg[tier][b]["accuracy"] - agg[tier][a]["accuracy"]
        if best is None or delta > best["delta"]:
            best = {"from_size": a, "to_size": b, "delta": delta}
    return best


def load_induction():
    if not INDUCTION_PATH.exists():
        return {}
    out = {}
    for line in INDUCTION_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        hf = NAME_MAP.get(r["model"], r["model"])
        size = MODEL_SIZE_MAP.get(hf)
        if size:
            out[size] = r
    return out


def induction_onset(induction, key="format_has_induction_head"):
    """Smallest model size at which an induction head is detected."""
    sizes = sorted(s for s, r in induction.items() if r.get(key))
    return sizes[0] if sizes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--exclude_heldout", action="store_true")
    args = ap.parse_args()

    agg = aggregate(load(args.shots, args.exclude_heldout))
    induction = load_induction()

    print("=" * 78)
    print("SYNTHESIS - behavioral vs. mechanistic evidence")
    print("=" * 78)

    comp = compounding_summary(agg)
    if comp:
        print(f"\nCompounding test: mean|residual| = {comp['mae']:.3f} over {comp['n_points']} points;"
              f" prediction inside 95% CI for {comp['ci_coverage']:.0%} of them.")
        explained = (comp["mae"] < COMPOUNDING_MAE_THRESHOLD
                     and comp["ci_coverage"] >= CI_COVERAGE_THRESHOLD)
        print(f"  -> depth effect {'IS' if explained else 'is NOT'} adequately explained by compounding"
              f" (pre-registered thresholds: MAE<{COMPOUNDING_MAE_THRESHOLD},"
              f" coverage>={CI_COVERAGE_THRESHOLD:.0%})")
    else:
        explained = None
        print("\nCompounding test: insufficient data (need depth_1 plus deeper tiers).")

    print("\nLargest adjacent-size accuracy jump per depth tier:")
    jumps = {}
    for tier in sorted(t for t in agg if t.startswith("depth_")):
        j = find_accuracy_jump(agg, tier)
        if j:
            jumps[tier] = j
            marker = "  <-- exceeds jump threshold" if j["delta"] >= ACCURACY_JUMP_THRESHOLD else ""
            print(f"  {tier:10s} +{j['delta']:.3f} between {j['from_size']:.1e} "
                  f"and {j['to_size']:.1e}{marker}")

    if not induction:
        print("\nNo induction-head data found - run induction_probe.py for each model size.")
        print("Cannot evaluate RQ2 without it.")
        return

    gen_onset = induction_onset(induction, "generic_has_induction_head")
    fmt_onset = induction_onset(induction, "format_has_induction_head")
    print(f"\nInduction-head onset (smallest size where detected):")
    print(f"  generic probe: {f'{gen_onset:.1e}' if gen_onset else 'not detected at any tested size'}")
    print(f"  format  probe: {f'{fmt_onset:.1e}' if fmt_onset else 'not detected at any tested size'}")

    big_jumps = [j for j in jumps.values() if j["delta"] >= ACCURACY_JUMP_THRESHOLD]

    print("\n" + "-" * 78)
    print("VERDICT (against pre-registered rules)")
    print("-" * 78)

    if not big_jumps:
        print("No accuracy jump exceeding the pre-registered threshold was observed.")
        print("=> NULL RESULT for this scale range. Report honestly: the tested range")
        print("   (see model sizes above) sits below any detectable transition for this")
        print("   task. This still constrains where future work should look.")
        return

    onset = fmt_onset or gen_onset
    if onset is None:
        print("Accuracy jump observed, but no induction head detected at any tested size.")
        print("=> MIRAGE-CONSISTENT (behavioral jump without detectable circuit change),")
        print("   with the caveat that absence of THIS circuit is not absence of all")
        print("   mechanistic change.")
        return

    aligned = any(j["to_size"] == onset or j["from_size"] == onset for j in big_jumps)

    if explained and not aligned:
        print("Depth effect explained by compounding AND induction onset does not")
        print("coincide with the accuracy jump.")
        print("=> MIRAGE CONFIRMED, now with mechanistic evidence the metric-artifact")
        print("   literature has not previously supplied.")
    elif not explained and aligned:
        print("Depth effect NOT fully explained by compounding AND induction onset")
        print("coincides with the accuracy jump.")
        print("=> MECHANISM CONFIRMED: the jump is not purely a scoring artifact.")
    else:
        print("Mixed evidence: compounding explains part of the depth effect, but")
        print("circuit onset and behavioral jump do not align cleanly.")
        print("=> MIXED RESULT - the most novel outcome, since neither existing camp")
        print("   predicts partial decoupling. Report both components separately and")
        print("   avoid claiming a clean verdict either way.")


if __name__ == "__main__":
    main()