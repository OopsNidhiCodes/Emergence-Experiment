"""
analysis.py
Turns scored_outputs.jsonl into the paper's core statistical results.

Three analyses, mapping onto the two research questions:

 (1) RQ1a - COMPOUNDING TEST (the centrepiece)
     Estimate per-step accuracy p from the k=1 depth tier, then compare the
     compounding prediction p^k against measured accuracy at k=2..5.
     Small residuals => the "jump" is explained by multi-step scoring alone.

 (2) RQ1b - CURVE SHAPE
     Fit a SATURATING power law and a sigmoid to accuracy vs. model size and
     compare via BIC.
     Note: the earlier version fitted an unbounded power law (a*x^b) against
     accuracy, which is bounded in [0,1]. An unbounded form cannot flatten
     near the ceiling, so it loses to the sigmoid almost by construction.
     The saturating form L*(1 - exp(-a*x^b)) is bounded, making the
     comparison fair. This was a real bias in the earlier pipeline.

 (3) Wilson confidence intervals on every accuracy point, so scaling curves
     carry error bars. With greedy decoding the sampling variance is zero,
     so the meaningful uncertainty is ITEM-level (binomial over n items),
     not repeat-level.

Usage:
    python analysis.py
    python analysis.py --shots 3        # restrict to one prompt condition
    python analysis.py --exclude_heldout
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent / "results"
SCORED_PATH = RESULTS_DIR / "scored_outputs.jsonl"
PLOTS_DIR = RESULTS_DIR / "plots"

# Two model families. NOTE: Pythia is a controlled scaling suite (identical
# architecture, data and data order across sizes), so scale is cleanly
# isolated. Qwen3 is NOT — sizes differ in training compute and data volume.
# Qwen3 therefore serves as a cross-architecture robustness check on the
# PHENOMENON, not as a second controlled scaling analysis. Do not pool the
# two families in a single scaling curve.
MODEL_SIZE_MAP = {
    # --- Qwen3 base models (cross-architecture check) ---
    "Qwen/Qwen3-0.6B-Base": 0.6e9,
    "Qwen/Qwen3-1.7B-Base": 1.7e9,
    "Qwen/Qwen3-4B-Base": 4.0e9,
    "Qwen/Qwen3-8B-Base": 8.0e9,
    # --- Pythia (controlled scaling suite) ---
    "EleutherAI/pythia-70m": 70e6,
    "EleutherAI/pythia-160m": 160e6,
    "EleutherAI/pythia-410m": 410e6,
    "EleutherAI/pythia-1b": 1.0e9,
    "EleutherAI/pythia-1.4b": 1.4e9,
    "EleutherAI/pythia-2.8b": 2.8e9,
    "EleutherAI/pythia-6.9b": 6.9e9,
}


# ---------------- statistics helpers ----------------

def wilson_ci(n_correct, n_total, z=1.96):
    """Wilson score interval - reliable near 0 and 1, unlike normal approx."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_correct / n_total
    d = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / d
    half = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bic(y, yhat, n_params):
    n = len(y)
    rss = max(float(np.sum((np.asarray(y) - np.asarray(yhat)) ** 2)), 1e-12)
    return n * math.log(rss / n) + n_params * math.log(n)


def saturating_power(x_log, L, a, b):
    """
    Bounded, monotone alternative to an unbounded power law.
    x_log is log10(params); the exponent is clipped to avoid overflow
    during optimisation when the fitter explores large b.
    """
    expo = np.clip(-a * np.power(10.0, np.clip(b * x_log, -30, 30)), -700, 30)
    return L * (1.0 - np.exp(expo))


def sigmoid(x_log, L, k, x0):
    return L / (1.0 + np.exp(-k * (x_log - x0)))


# ---------------- data loading ----------------

def family_of(model_name):
    return "qwen" if model_name.lower().startswith("qwen") else "pythia"


def load(shots=None, exclude_heldout=False, scratchpad=None, family=None):
    if not SCORED_PATH.exists():
        raise SystemExit(f"No scored outputs at {SCORED_PATH}. Run inference.py then scoring.py.")
    records = []
    for line in SCORED_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if shots is not None and r.get("shots") != shots:
            continue
        if scratchpad is not None and bool(r.get("scratchpad")) != scratchpad:
            continue
        if family is not None and family_of(r["model"]) != family:
            continue
        if exclude_heldout and r.get("held_out"):
            continue
        records.append(r)
    if not records:
        raise SystemExit("No records matched the requested filters.")
    return records


def aggregate(records):
    """-> {tier: {size: {accuracy, lo, hi, n, mean_log_prob, num_steps}}}"""
    agg = defaultdict(lambda: defaultdict(lambda: {"c": 0, "n": 0, "lp": [], "lps": [], "k": None, "ops": None}))
    unknown = set()
    for r in records:
        size = MODEL_SIZE_MAP.get(r["model"])
        if size is None:
            unknown.add(r["model"])
            continue
        b = agg[r["tier"]][size]
        b["c"] += r["exact_match"]
        b["n"] += 1
        b["k"] = r["num_steps"]
        if r.get("operations"):
            b["ops"] = r["operations"]
        if r.get("log_prob") is not None:
            b["lp"].append(r["log_prob"])
        if r.get("log_prob_sum") is not None:
            b["lps"].append(r["log_prob_sum"])
    if unknown:
        print(f"WARNING: unknown model name(s) skipped: {unknown}")
        print("         add them to MODEL_SIZE_MAP at the top of analysis.py\n")

    out = {}
    for tier, by_size in agg.items():
        out[tier] = {}
        for size, b in by_size.items():
            lo, hi = wilson_ci(b["c"], b["n"])
            out[tier][size] = {
                "accuracy": b["c"] / b["n"] if b["n"] else 0.0,
                "ci_low": lo,
                "ci_high": hi,
                "n": b["n"],
                "mean_log_prob": float(np.mean(b["lp"])) if b["lp"] else None,
                "mean_log_prob_sum": float(np.mean(b["lps"])) if b["lps"] else None,
                "num_steps": b["k"],
                "operations": b["ops"],
            }
    return out


# ---------------- RQ1a: compounding test ----------------

def compounding_test(agg):
    """
    Operation-aware compounding test.

    The naive form of this test estimates a single per-step accuracy p and
    predicts p^k. That is only valid if every step is equally hard. It is
    NOT valid here: depth_1 is addition only, while depth_2+ also require
    multiplication and subtraction. Estimating p from addition alone and
    raising it to the k-th power compares different operations and produces
    a meaningless prediction (in an earlier run it predicted 100% at every
    depth because single-step addition was at ceiling).

    Instead we estimate a SEPARATE per-step accuracy for each operation from
    its own single-step tier:

        p_add   from depth_1      (a + b)
        p_mult  from mult_single  (a * c)
        p_sub   from sub_single   (a - b)

    and predict a depth-k item as the product over its actual operation
    sequence, e.g. depth_3 = ["add","mult","add"] -> p_add * p_mult * p_add.

    Residual = measured - predicted.
      near 0            -> the depth effect is compounding (metric artifact)
      large negative    -> the model fails FASTER than independent per-step
                           errors predict; something beyond compounding is
                           breaking (e.g. it cannot chain steps at all)
      large positive    -> better than compounding predicts
    """
    OP_TIERS = {"add": "depth_1", "mult": "mult_single", "sub": "sub_single"}

    missing = [t for t in OP_TIERS.values() if t not in agg]
    if missing:
        print(f"\nCannot run operation-aware compounding test - missing tier(s): {missing}")
        print("Re-generate the benchmark (python benchmark.py) and re-run inference.")
        return None

    depth_tiers = sorted(
        [t for t in agg if t.startswith("depth_") and t != "depth_1"],
        key=lambda t: int(t.split("_")[1]),
    )
    sizes = sorted(agg["depth_1"].keys())

    print("\n" + "=" * 78)
    print("RQ1a - OPERATION-AWARE COMPOUNDING TEST")
    print("=" * 78)
    print("Per-step accuracy is estimated PER OPERATION from its own single-step")
    print("tier, then multiplied along each item's actual operation sequence.\n")

    rows = []
    for size in sizes:
        p = {op: agg[tier][size]["accuracy"] for op, tier in OP_TIERS.items() if size in agg[tier]}
        if len(p) < 3:
            continue
        print(f"  model size {size:.1e}   p_add={p['add']:.3f}  p_mult={p['mult']:.3f}  p_sub={p['sub']:.3f}")
        print(f"    {'depth':>6} {'predicted':>10} {'measured':>10} {'residual':>10}   {'95% CI':>16}  ops")
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
            resid = d["accuracy"] - pred
            inside = d["ci_low"] <= pred <= d["ci_high"]
            ci = f"[{d['ci_low']:.2f},{d['ci_high']:.2f}]"
            flag = "" if inside else "  <-- outside CI"
            k = int(tier.split("_")[1])
            print(f"    {k:>6} {pred:>10.3f} {d['accuracy']:>10.3f} {resid:>+10.3f}   {ci:>16}{flag}  "
                  + "*".join(o[0] for o in ops))
            rows.append({"size": size, "k": k, "predicted": pred, "measured": d["accuracy"],
                         "residual": resid, "pred_in_ci": inside})
        print()

    if not rows:
        return None

    mae = float(np.mean([abs(r["residual"]) for r in rows]))
    inside = sum(r["pred_in_ci"] for r in rows)
    neg = sum(1 for r in rows if r["residual"] < -0.10)
    print(f"  Summary: mean |residual| = {mae:.3f}; prediction inside 95% CI for "
          f"{inside}/{len(rows)} points; {neg}/{len(rows)} points fail more than "
          f"0.10 BELOW prediction.")
    print("  Small MAE + high CI coverage  => depth effect is compounding (mirage-consistent).")
    print("  Large negative residuals      => failure is faster than compounding predicts;")
    print("                                   chaining itself is breaking, not just per-step error.")
    return rows


# ---------------- RQ1b: curve shape ----------------

def fit_curves(sizes, accs):
    x_log = np.log10(np.asarray(sizes, dtype=float))
    y = np.asarray(accs, dtype=float)
    res = {"better_fit": None, "bic_difference": None}

    try:
        p0 = [max(float(y.max()), 0.05), 1e-3, 0.1]
        popt, _ = curve_fit(saturating_power, x_log, y, p0=p0, maxfev=20000)
        b_pow = bic(y, saturating_power(x_log, *popt), 3)
        res["saturating_power"] = {"bic": b_pow}
    except Exception as exc:
        res["saturating_power"] = {"error": str(exc)}
        b_pow = None

    try:
        p0 = [max(float(y.max()), 0.05), 1.0, float(np.median(x_log))]
        popt, _ = curve_fit(sigmoid, x_log, y, p0=p0, maxfev=20000)
        b_sig = bic(y, sigmoid(x_log, *popt), 3)
        res["sigmoid"] = {"bic": b_sig}
    except Exception as exc:
        res["sigmoid"] = {"error": str(exc)}
        b_sig = None

    if b_pow is not None and b_sig is not None:
        diff = b_pow - b_sig
        res["bic_difference"] = diff
        if diff > 2:
            res["better_fit"] = "sigmoid (threshold-like)"
        elif diff < -2:
            res["better_fit"] = "saturating power law (smooth)"
        else:
            res["better_fit"] = "inconclusive (|dBIC| < 2)"
    return res


def curve_analysis(agg):
    print("\n" + "=" * 78)
    print("RQ1b - CURVE SHAPE (saturating power law vs. sigmoid, by BIC)")
    print("=" * 78)
    print("Both forms are bounded in [0,1], so neither is favoured by construction.\n")

    results = {}
    for tier in sorted(agg):
        sizes = sorted(agg[tier].keys())
        if len(sizes) < 4:
            print(f"  {tier:14s}: need >=4 model sizes to compare fits (have {len(sizes)}) - skipped")
            continue
        accs = [agg[tier][s]["accuracy"] for s in sizes]
        if max(accs) - min(accs) < 0.05:
            print(f"  {tier:14s}: accuracy is flat across scale (range {max(accs)-min(accs):.3f}) - "
                  f"no curve to classify")
            continue
        fit = fit_curves(sizes, accs)
        results[tier] = fit
        d = fit["bic_difference"]
        print(f"  {tier:14s}: {fit['better_fit']}"
              + (f"   (dBIC = {d:+.2f})" if d is not None else ""))
    return results


# ---------------- plots ----------------

def make_plots(agg):
    """
    Four figures, written to results/plots/:
      fig1_depth_accuracy   - accuracy vs scale, one line per step depth
      fig2_compounding      - measured vs operation-aware predicted accuracy
      fig3_logprob          - continuous metric vs scale (the mirage control)
      fig4_surface_vs_depth - single-step difficulty vs step count
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping plots)")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    depth_tiers = sorted([t for t in agg if t.startswith("depth_")],
                         key=lambda t: int(t.split("_")[1]))

    def errbars(tier):
        sizes = sorted(agg[tier].keys())
        acc = [agg[tier][s]["accuracy"] for s in sizes]
        lo = [agg[tier][s]["accuracy"] - agg[tier][s]["ci_low"] for s in sizes]
        hi = [agg[tier][s]["ci_high"] - agg[tier][s]["accuracy"] for s in sizes]
        return sizes, acc, [lo, hi]

    # ---- fig 1: accuracy vs scale by depth ----
    if depth_tiers:
        plt.figure(figsize=(6.5, 4.2))
        for tier in depth_tiers:
            k = int(tier.split("_")[1])
            s, a, e = errbars(tier)
            plt.errorbar(s, a, yerr=e, fmt="o-", capsize=3, label=f"k={k}")
        plt.xscale("log"); plt.ylim(-0.05, 1.05)
        plt.xlabel("Model size (parameters)")
        plt.ylabel("Exact-match accuracy")
        plt.title("Accuracy vs. scale by step depth (95% Wilson CIs)")
        plt.grid(alpha=0.3); plt.legend(title="steps", fontsize=8)
        plt.savefig(PLOTS_DIR / "fig1_depth_accuracy.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ---- fig 2: measured vs operation-aware prediction ----
    OP_TIERS = {"add": "depth_1", "mult": "mult_single", "sub": "sub_single"}
    if all(t in agg for t in OP_TIERS.values()):
        sizes = sorted(agg["depth_1"].keys())
        plt.figure(figsize=(6.5, 4.2))
        for tier in depth_tiers:
            if tier == "depth_1":
                continue
            k = int(tier.split("_")[1])
            xs, meas, pred = [], [], []
            for s in sizes:
                if s not in agg[tier]:
                    continue
                ops = agg[tier][s].get("operations")
                if not ops:
                    continue
                p = 1.0
                ok = True
                for op in ops:
                    tier_for_op = OP_TIERS[op]
                    if s not in agg.get(tier_for_op, {}):
                        ok = False; break
                    p *= agg[tier_for_op][s]["accuracy"]
                if not ok:
                    continue
                xs.append(s); meas.append(agg[tier][s]["accuracy"]); pred.append(p)
            if xs:
                line, = plt.plot(xs, meas, "o-", label=f"k={k} measured")
                plt.plot(xs, pred, "--", color=line.get_color(), alpha=0.6,
                         label=f"k={k} predicted")
        plt.xscale("log"); plt.ylim(-0.05, 1.05)
        plt.xlabel("Model size (parameters)")
        plt.ylabel("Exact-match accuracy")
        plt.title("Measured vs. operation-aware compounding prediction")
        plt.grid(alpha=0.3); plt.legend(fontsize=7, ncol=2)
        plt.savefig(PLOTS_DIR / "fig2_compounding.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ---- fig 3: log-probability ----
    plt.figure(figsize=(6.5, 4.2))
    plotted = 0
    for tier in sorted(agg):
        sizes = sorted(agg[tier].keys())
        lps = [agg[tier][s]["mean_log_prob"] for s in sizes]
        if any(v is None for v in lps):
            continue
        plt.plot(sizes, lps, "o-", label=tier); plotted += 1
    if plotted:
        plt.xscale("log")
        plt.xlabel("Model size (parameters)")
        plt.ylabel("Mean log-probability of correct answer")
        plt.title("Continuous metric vs. scale")
        plt.grid(alpha=0.3); plt.legend(fontsize=7, ncol=2)
        plt.savefig(PLOTS_DIR / "fig3_logprob.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- fig 4: surface difficulty vs step depth ----
    surface = [t for t in ["single_digit", "two_digit", "three_digit", "carrying",
                           "mult_single", "sub_single"] if t in agg]
    if surface and depth_tiers:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for tier in surface:
            s, a, e = errbars(tier)
            axes[0].errorbar(s, a, yerr=e, fmt="o-", capsize=3, label=tier)
        axes[0].set_title("Single-step tiers (k=1, varying surface difficulty)")
        for tier in depth_tiers:
            k = int(tier.split("_")[1])
            s, a, e = errbars(tier)
            axes[1].errorbar(s, a, yerr=e, fmt="o-", capsize=3, label=f"k={k}")
        axes[1].set_title("Depth tiers (varying step count)")
        for ax in axes:
            ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Model size (parameters)")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
        axes[0].set_ylabel("Exact-match accuracy")
        plt.savefig(PLOTS_DIR / "fig4_surface_vs_depth.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"\nPlots written to {PLOTS_DIR}")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p.name}")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=None,
                    help="restrict analysis to one prompt condition (0 or 3)")
    ap.add_argument("--exclude_heldout", action="store_true")
    ap.add_argument("--scratchpad", dest="scratchpad", action="store_true", default=None,
                    help="analyse ONLY scratchpad rows")
    ap.add_argument("--no_scratchpad", dest="scratchpad", action="store_false",
                    help="analyse ONLY standard (non-scratchpad) rows")
    ap.add_argument("--family", choices=["pythia", "qwen"], default=None,
                    help="restrict to one model family. The families are NOT "
                         "pooled by default in interpretation: Pythia is a "
                         "controlled scaling suite, Qwen3 is not.")
    args = ap.parse_args()

    records = load(args.shots, args.exclude_heldout, args.scratchpad, args.family)
    agg = aggregate(records)

    print("=" * 78)
    print("ACCURACY BY TIER AND MODEL SIZE (with 95% Wilson CIs)")
    print("=" * 78)
    for tier in sorted(agg):
        print(f"\n  {tier}  (k={list(agg[tier].values())[0]['num_steps']})")
        for size in sorted(agg[tier]):
            d = agg[tier][size]
            lp = f"{d['mean_log_prob']:.3f}" if d["mean_log_prob"] is not None else "n/a"
            print(f"    {size:>9.1e}  acc={d['accuracy']:.3f} "
                  f"[{d['ci_low']:.2f},{d['ci_high']:.2f}]  n={d['n']}  mean_logprob={lp}")

    compounding_test(agg)
    curve_analysis(agg)
    make_plots(agg)


if __name__ == "__main__":
    main()