"""
scoring.py
Classifies every model output in results/raw_outputs.jsonl into an error
type, and writes results/scored_outputs.jsonl.

Error taxonomy (the point of this is RQ2: a model making STRUCTURED errors
has partially learned the algorithm; a model making unstructured errors has
not learned it at all):

  correct            - exact match
  no_answer          - produced no number at all (does not understand the task)
  copied_operand     - echoed a number from the question instead of computing
  copied_exemplar    - echoed an ANSWER from the few-shot examples (strong
                       evidence of format-copying without computation)
  dropped_operand    - applied a later operation to only ONE addend, dropping
                       the other, e.g. (4+19)*5 -> 95 because it computed
                       19*5 and never added the 4. This is a SPECIFIC
                       algorithmic failure (the multiplication executes but
                       the addition result is never bound as its operand),
                       and is much stronger evidence of partial algorithm
                       acquisition than a generic magnitude miss.
  partial_step       - matches the result of a CORRECT PREFIX of the expression
                       (e.g. computed (a+b) but stopped before "* c").
                       This is the strongest evidence of partial algorithm
                       acquisition, and only exists for k>1 items.
  carry_error        - off by an exact power of 10 (classic missed carry)
  near_miss          - off by <= 9 (digit slip)
  magnitude_error    - right order of magnitude, wrong value
  random             - none of the above

Usage:
    python scoring.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"
SCORED_PATH = RESULTS_DIR / "scored_outputs.jsonl"


def raw_files():
    """
    Every results/raw_outputs*.jsonl file. Each machine writes its own
    (via inference.py --out), so results committed from different machines
    merge here without ever colliding in git.
    """
    return sorted(RESULTS_DIR.glob("raw_outputs*.jsonl"))
TASKS_PATH = Path(__file__).parent / "data" / "tasks.json"

# Answers appearing in the few-shot exemplars (see FEWSHOT_EXEMPLARS in
# benchmark.py). Small models frequently just echo one of these instead of
# computing anything. Without this category such echoes get scored as
# "near_miss" or "magnitude_error", which wrongly implies partial arithmetic
# competence when no arithmetic occurred at all.
try:
    from benchmark import FEWSHOT_EXEMPLARS
    EXEMPLAR_ANSWERS = {int(a) for _, a in FEWSHOT_EXEMPLARS}
except Exception:
    EXEMPLAR_ANSWERS = {7, 37, 22}


def prefix_values(expression):
    """
    All intermediate values obtainable by evaluating a correct PREFIX of the
    expression. For "((2 + 18) * 5) + 13" this yields {20, 100, 113}.
    Used to detect a model that did some steps correctly then stopped.
    """
    vals = set()
    # progressively strip trailing "<op> <number>" groups and unwrap parens
    expr = expression
    for _ in range(6):
        try:
            vals.add(int(eval(expr)))
        except Exception:
            pass
        stripped = re.sub(r"\s*[\+\-\*]\s*\d+\s*$", "", expr).strip()
        if stripped == expr:
            break
        expr = stripped
        # unwrap a fully-enclosing paren pair if present
        if expr.startswith("(") and expr.endswith(")"):
            expr = expr[1:-1].strip()
    return vals


def operand_values(expression):
    return {int(m) for m in re.findall(r"\d+", expression)}


def dropped_operand_values(expression):
    """
    Values obtained by applying the outer operation to only ONE of the two
    addends inside the innermost parenthesis, ignoring the other.
    For "(4 + 19) * 5" this returns {20, 95} (4*5 and 19*5).
    """
    m = re.search(r"\((\d+) \+ (\d+)\)\s*\*\s*(\d+)", expression)
    if not m:
        return set()
    a, b, c = (int(x) for x in m.groups())
    return {a * c, b * c}


def classify(true_answer, predicted, expression):
    if predicted is None:
        return "no_answer"
    try:
        true_val = int(true_answer)
        pred_val = int(predicted)
    except (TypeError, ValueError):
        return "no_answer"

    if true_val == pred_val:
        return "correct"

    operands = operand_values(expression)
    # A "partial step" must be the RESULT of at least one computation, so
    # exclude values that are simply operands copied from the question --
    # otherwise stripping the expression all the way down to a bare number
    # would mislabel operand-copying as partial algorithm acquisition.
    prefixes = prefix_values(expression) - {true_val} - operands
    if pred_val in prefixes:
        return "partial_step"

    if pred_val in dropped_operand_values(expression):
        return "dropped_operand"

    if pred_val in operands:
        return "copied_operand"

    # Checked BEFORE the numeric-distance categories: an echoed exemplar
    # answer can land close to the true value by chance, and must not be
    # mistaken for a near miss.
    if pred_val in EXEMPLAR_ANSWERS:
        return "copied_exemplar"

    diff = abs(true_val - pred_val)
    if diff in (1, 10, 100, 1000, 10000):
        return "carry_error"
    if diff <= 9:
        return "near_miss"
    if true_val != 0 and 0.5 <= abs(pred_val / true_val) <= 2.0:
        return "magnitude_error"
    return "random"


def main():
    files = raw_files()
    if not files:
        print(f"No raw_outputs*.jsonl found in {RESULTS_DIR}. Run inference.py first.")
        return
    print("Reading:")
    for f in files:
        n = sum(1 for line in f.read_text().splitlines() if line.strip())
        print(f"  {f.name}  ({n} rows)")
    print()

    expr_by_id, ops_by_id = {}, {}
    if TASKS_PATH.exists():
        for t in json.loads(TASKS_PATH.read_text()):
            expr_by_id[t["id"]] = t["expression"]
            if t.get("operations"):
                ops_by_id[t["id"]] = t["operations"]

    records = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    # warn about duplicate rows (same model+shots+task run twice)
    seen = defaultdict(int)
    for r in records:
        seen[(r["model"], r.get("shots"), r["task_id"])] += 1
    dupes = sum(1 for v in seen.values() if v > 1)
    if dupes:
        print("=" * 70)
        print(f"STOPPING: {dupes} (model, shots, task) combinations appear more than once.")
        print("=" * 70)
        print("This usually means results from DIFFERENT benchmark versions are mixed")
        print("in results/. Task IDs are reused across benchmark versions, so old rows")
        print("would be scored against the CURRENT tasks.json - i.e. against the wrong")
        print("expressions - and accuracies would be averaged over incompatible runs.")
        print()
        print("Files read:")
        for f in files:
            print(f"  {f.name}")
        print()
        print("Delete the stale file(s), keep only the current benchmark's results,")
        print("and re-run. Nothing has been written.")
        raise SystemExit(1)

    for r in records:
        expr = expr_by_id.get(r["task_id"], "")
        r["error_type"] = classify(r["true_answer"], r["predicted_answer"], expr)
        # Attach the operation sequence from tasks.json. inference.py does not
        # write this field, so joining it here lets the operation-aware
        # compounding test run without re-running inference.
        if r["task_id"] in ops_by_id:
            r["operations"] = ops_by_id[r["task_id"]]

    SCORED_PATH.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    summary = defaultdict(lambda: defaultdict(int))
    for r in records:
        summary[(r["model"], r.get("shots"), r["tier"])][r["error_type"]] += 1

    cols = ["correct", "partial_step", "dropped_operand", "carry_error",
            "near_miss", "magnitude_error", "copied_operand", "copied_exemplar",
            "no_answer", "random"]
    print(f"{'model':<26}{'sh':>3} {'tier':<13}" + "".join(f"{c[:9]:>11}" for c in cols))
    for (model, shots, tier), counts in sorted(summary.items()):
        short = model.split("/")[-1]
        print(f"{short:<26}{shots if shots is not None else '-':>3} {tier:<13}"
              + "".join(f"{counts.get(c, 0):>11}" for c in cols))

    print(f"\nWrote {len(records)} scored records to {SCORED_PATH}")
    print("\nNote: 'partial_step' and 'dropped_operand' are the key RQ2 columns.")
    print("Both count cases where the model executed PART of the algorithm correctly:")
    print("  partial_step    - stopped after a correct prefix of the expression")
    print("  dropped_operand - ran the outer operation on one addend only")
    print("These are structured failures, not guessing, and their rise with scale is")
    print("behavioral evidence of partial algorithm acquisition.")


if __name__ == "__main__":
    main()