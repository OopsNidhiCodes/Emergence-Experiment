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

MODEL_SIZE_MAP = {
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

def load(shots=None, exclude_heldout=False):
    if not SCORED_PATH.exists():
        raise SystemExit(f"No scored outputs at {SCORED_PATH}. Run inference.py then scoring.py.")
    records = []
    for line in SCORED_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if shots is not None and r.get("shots") != shots:
            continue
        if exclude_heldout and r.get("held_out"):
            continue
        records.append(r)
    if not records:
        raise SystemExit("No records matched the requested filters.")
    return records


def aggregate(records):
    """-> {tier: {size: {accuracy, lo, hi, n, mean_log_prob, num_steps}}}"""
    agg = defaultdict(lambda: defaultdict(lambda: {"c": 0, "n": 0, "lp": [], "k": None}))
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
        if r.get("log_prob") is not None:
            b["lp"].append(r["log_prob"])
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
                "num_steps": b["k"],
            }
    return out


# ---------------- RQ1a: compounding test ----------------

def compounding_test(agg):
    """Compare measured accuracy at depth k against p(k=1)^k, per model size."""
    if "depth_1" not in agg:
        print("No depth_1 tier found - cannot estimate per-step accuracy.")
        return None

    depth_tiers = sorted(
        [t for t in agg if t.startswith("depth_")],
        key=lambda t: int(t.split("_")[1]),
    )
    sizes = sorted(agg["depth_1"].keys())

    print("\n" + "=" * 78)
    print("RQ1a - COMPOUNDING-PROBABILITY TEST")
    print("=" * 78)
    print("Residual = measured - predicted.  |residual| near 0 supports the")
    print("metric-artifact account: the drop with depth is just p^k.\n")

    rows = []
    for size in sizes:
        p_step = agg["depth_1"][size]["accuracy"]
        print(f"  model size {size:.1e}   per-step accuracy p = {p_step:.3f}")
        print(f"    {'depth':>6} {'predicted':>10} {'measured':>10} {'residual':>10}   {'95% CI':>16}")
        for tier in depth_tiers:
            k = int(tier.split("_")[1])
            if size not in agg[tier]:
                continue
            d = agg[tier][size]
            pred = p_step ** k
            resid = d["accuracy"] - pred
            ci = f"[{d['ci_low']:.2f},{d['ci_high']:.2f}]"
            flag = ""
            # is the prediction inside the measured CI?
            if not (d["ci_low"] <= pred <= d["ci_high"]) and k > 1:
                flag = "  <-- outside CI"
            print(f"    {k:>6} {pred:>10.3f} {d['accuracy']:>10.3f} {resid:>+10.3f}   {ci:>16}{flag}")
            rows.append({"size": size, "k": k, "predicted": pred,
                         "measured": d["accuracy"], "residual": resid,
                         "pred_in_ci": bool(d["ci_low"] <= pred <= d["ci_high"])})
        print()

    multi = [r for r in rows if r["k"] > 1]
    if multi:
        mae = float(np.mean([abs(r["residual"]) for r in multi]))
        inside = sum(r["pred_in_ci"] for r in multi)
        print(f"  Summary over k>1 points: mean |residual| = {mae:.3f}; "
              f"prediction inside 95% CI for {inside}/{len(multi)} points.")
        print("  Interpretation: many points inside CI and small MAE => the depth")
        print("  effect is compounding, not an extra mechanism. Systematically")
        print("  positive residuals => the model does better than compounding")
        print("  predicts, implying something beyond per-step probability.")
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping plots)")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: depth tiers, measured vs compounding prediction
    depth_tiers = sorted([t for t in agg if t.startswith("depth_")],
                         key=lambda t: int(t.split("_")[1]))
    if "depth_1" in agg and len(depth_tiers) > 1:
        plt.figure(figsize=(7, 4.5))
        for tier in depth_tiers:
            sizes = sorted(agg[tier].keys())
            accs = [agg[tier][s]["accuracy"] for s in sizes]
            los = [agg[tier][s]["accuracy"] - agg[tier][s]["ci_low"] for s in sizes]
            his = [agg[tier][s]["ci_high"] - agg[tier][s]["accuracy"] for s in sizes]
            plt.errorbar(sizes, accs, yerr=[los, his], fmt="o-", capsize=3, label=f"measured {tier}")
        # predicted curves
        sizes = sorted(agg["depth_1"].keys())
        for tier in depth_tiers:
            k = int(tier.split("_")[1])
            if k == 1:
                continue
            preds = [agg["depth_1"][s]["accuracy"] ** k for s in sizes]
            plt.plot(sizes, preds, "--", alpha=0.5, label=f"predicted k={k}")
        plt.xscale("log")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Model size (parameters)")
        plt.ylabel("Exact-match accuracy")
        plt.title("Measured vs. compounding-predicted accuracy by step depth")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        plt.savefig(PLOTS_DIR / "compounding_test.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Plot 2: accuracy vs log-probability per tier
    plt.figure(figsize=(7, 4.5))
    for tier in sorted(agg):
        sizes = sorted(agg[tier].keys())
        lps = [agg[tier][s]["mean_log_prob"] for s in sizes]
        if any(v is None for v in lps):
            continue
        plt.plot(sizes, lps, "o-", label=tier)
    plt.xscale("log")
    plt.xlabel("Model size (parameters)")
    plt.ylabel("Mean log-probability of correct answer")
    plt.title("Continuous metric vs. model size")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.savefig(PLOTS_DIR / "log_prob_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots written to {PLOTS_DIR}")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=None,
                    help="restrict analysis to one prompt condition (0 or 3)")
    ap.add_argument("--exclude_heldout", action="store_true")
    args = ap.parse_args()

    records = load(args.shots, args.exclude_heldout)
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
