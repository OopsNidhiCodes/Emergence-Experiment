"""
benchmark.py
Generates and validates the graded arithmetic benchmark.

DESIGN RATIONALE
----------------
The central test of RQ1 is whether an apparent accuracy "jump" is explained
by compounding probability: P(all k steps correct) ~= p_step^k. That test
only has teeth if the benchmark spans a RANGE of step-depths k. With only
k=1 and k=2, the predicted curve barely bends and the prediction is nearly
unfalsifiable.

This benchmark therefore has two independent axes:

  AXIS 1 - step depth (the compounding axis, drives RQ1)
      depth_1  : a + b                      (k=1)
      depth_2  : (a + b) * c                (k=2)
      depth_3  : (a + b) * c + d            (k=3)
      depth_4  : ((a + b) * c + d) - e      (k=4)
      depth_5  : ((((a + b) * c) + d) - e) * f  (k=5)

  AXIS 2 - single-step surface difficulty (controls for "harder digits"
           rather than "more steps"; all k=1)
      single_digit, two_digit, three_digit, carrying

Holding k=1 fixed while varying digit count lets you show that any jump on
the depth axis is about step COUNT, not about arithmetic being harder.

Ground truth is always computed by Python itself, never hand-typed.

Usage:
    python benchmark.py                  # generate data/tasks.json
    python benchmark.py --check          # re-validate an existing tasks.json
    python benchmark.py --per_tier 20    # items per tier (default 20)
"""

import json
import random
import argparse
from pathlib import Path

random.seed(42)  # fixed seed -> identical benchmark on every machine

DATA_PATH = Path(__file__).parent / "data" / "tasks.json"


# ----------------------------------------------------------------------
# AXIS 2: single-step surface difficulty (all k = 1)
# ----------------------------------------------------------------------

def _simple_add(tier, n, lo, hi, force_carry=False):
    items, made = [], 0
    while made < n:
        a, b = random.randint(lo, hi), random.randint(lo, hi)
        if force_carry and (a % 10) + (b % 10) < 10:
            continue
        items.append({
            "id": f"{tier}_{made:02d}",
            "tier": tier,
            "axis": "surface_difficulty",
            "expression": f"{a} + {b}",
            "answer": str(a + b),
            "num_steps": 1,
        })
        made += 1
    return items


def make_single_digit(n):
    return _simple_add("single_digit", n, 1, 9)


def make_two_digit(n):
    return _simple_add("two_digit", n, 10, 99)


def make_three_digit(n):
    return _simple_add("three_digit", n, 100, 999)


def make_carrying(n):
    return _simple_add("carrying", n, 10, 99, force_carry=True)


# ----------------------------------------------------------------------
# AXIS 1: step depth (the compounding axis)
# ----------------------------------------------------------------------

def make_depth(k, n):
    """
    Builds k-step expressions. Operands are kept small (single / low-double
    digit) so surface difficulty stays roughly constant as k grows -- the
    ONLY thing changing across depth tiers should be the number of
    sequential steps required.
    """
    items = []
    for i in range(n):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        c = random.randint(2, 9)
        d = random.randint(2, 20)
        e = random.randint(2, 20)
        f = random.randint(2, 5)

        if k == 1:
            expr = f"{a} + {b}"
        elif k == 2:
            expr = f"({a} + {b}) * {c}"
        elif k == 3:
            expr = f"(({a} + {b}) * {c}) + {d}"
        elif k == 4:
            expr = f"((({a} + {b}) * {c}) + {d}) - {e}"
        elif k == 5:
            expr = f"(((({a} + {b}) * {c}) + {d}) - {e}) * {f}"
        else:
            raise ValueError(f"unsupported depth {k}")

        items.append({
            "id": f"depth_{k}_{i:02d}",
            "tier": f"depth_{k}",
            "axis": "step_depth",
            "expression": expr,
            "answer": str(eval(expr)),  # safe: expression built here, not user input
            "num_steps": k,
        })
    return items


# ----------------------------------------------------------------------
# Prompt formatting (zero-shot vs few-shot)
# ----------------------------------------------------------------------

# Exemplars are chosen so that NONE of their answers appears anywhere in the
# 180-item benchmark, and none is a commutation of a benchmark item. This
# matters: with the original exemplars (3+4=7, 12+25=37, (5+6)*2=22) small
# models frequently echoed an exemplar answer, and because the benchmark
# contained items with those same answers (e.g. "4 + 3 = 7"), those echoes
# were scored as CORRECT. That inflated per-step accuracy, which in turn
# feeds the compounding prediction in analysis.py.
# Verified collision-free against data/tasks.json (seed 42).
FEWSHOT_EXEMPLARS = [
    ("4 + 1", "5"),
    ("21 + 69", "90"),
    ("(19 + 32) * 3", "153"),
]


def format_prompt(expression, shots=0):
    """
    shots=0 -> "12 + 25 ="
    shots>0 -> solved examples first, so the model can infer the task
               format from context instead of guessing it.

    Why this matters: at small scale a model may score 0% simply because it
    does not recognise that a numeric answer is wanted (it continues with
    LaTeX, code, prose...). Few-shot prompting separates "cannot do
    arithmetic" from "does not know what is being asked". It also makes the
    RQ2 induction-head probe far more relevant, since in-context format
    inference is precisely what induction heads are known to support.
    """
    prefix = ""
    for expr, ans in FEWSHOT_EXEMPLARS[:shots]:
        prefix += f"{expr} = {ans}\n"
    return f"{prefix}{expression} ="


# ----------------------------------------------------------------------

def build_benchmark(per_tier):
    tasks = []
    # Axis 2 - surface difficulty (k=1 throughout)
    tasks += make_single_digit(per_tier)
    tasks += make_two_digit(per_tier)
    tasks += make_three_digit(per_tier)
    tasks += make_carrying(per_tier)
    # Axis 1 - step depth
    for k in [1, 2, 3, 4, 5]:
        tasks += make_depth(k, per_tier)
    return tasks


def validate(tasks):
    """Independently re-derive every answer and confirm it matches."""
    errors = []
    for t in tasks:
        try:
            recomputed = str(eval(t["expression"]))
        except Exception as exc:
            errors.append((t["id"], f"could not evaluate: {exc}"))
            continue
        if recomputed != t["answer"]:
            errors.append((t["id"], f"mismatch: stored={t['answer']} recomputed={recomputed}"))
        if t.get("axis") == "step_depth":
            n_ops = sum(t["expression"].count(op) for op in ["+", "*", "-"])
            if n_ops != t["num_steps"]:
                errors.append((t["id"], f"depth mismatch: num_steps={t['num_steps']} operators={n_ops}"))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="validate an existing tasks.json instead of regenerating")
    parser.add_argument("--per_tier", type=int, default=20, help="items per tier")
    parser.add_argument("--holdout_frac", type=float, default=0.2,
                        help="fraction of EACH tier reserved as held-out")
    args = parser.parse_args()

    if args.check:
        tasks = json.loads(DATA_PATH.read_text())
    else:
        tasks = build_benchmark(args.per_tier)

        # Hold out a fraction WITHIN EACH TIER so the held-out set stays
        # balanced across tiers instead of being drawn from just one.
        by_tier = {}
        for t in tasks:
            by_tier.setdefault(t["tier"], []).append(t)
        n_hold = 0
        for tier_items in by_tier.values():
            k = int(round(len(tier_items) * args.holdout_frac))
            for t in tier_items:
                t["held_out"] = False
            for t in (tier_items[-k:] if k else []):
                t["held_out"] = True
                n_hold += 1

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_PATH.write_text(json.dumps(tasks, indent=2))
        print(f"Wrote {len(tasks)} tasks across {len(by_tier)} tiers to {DATA_PATH}")
        print(f"Held-out items: {n_hold}")

    errors = validate(tasks)
    if errors:
        print(f"VALIDATION FAILED on {len(errors)} item(s):")
        for tid, msg in errors:
            print(f"  {tid}: {msg}")
        raise SystemExit(1)

    print(f"All {len(tasks)} ground-truth answers validated correctly.")
    tiers = {}
    for t in tasks:
        tiers[t["tier"]] = tiers.get(t["tier"], 0) + 1
    print("Tiers:", ", ".join(f"{k}({v})" for k, v in sorted(tiers.items())))


if __name__ == "__main__":
    main()
