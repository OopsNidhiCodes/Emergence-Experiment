"""
inference.py
Runs the benchmark through one Pythia model size and records, per item:
  - exact_match (0/1)
  - mean token log-probability of the correct answer (continuous metric)
  - the raw generated text (for later error-type classification)

Needs a real machine with internet access to Hugging Face and enough RAM
for the chosen model size.

Usage:
    python inference.py --model EleutherAI/pythia-70m --shots 3
    python inference.py --model EleutherAI/pythia-1.4b --shots 3

Run once per (model, shots) combination. Results are APPENDED to
results/raw_outputs.jsonl, so different model sizes can be run on
different machines and the files concatenated afterwards.

IMPORTANT: because results are appended, re-running the SAME model twice
duplicates its rows. Use --overwrite to clear previous rows for this
(model, shots) pair before writing.
"""

import json
import argparse
from pathlib import Path

from benchmark import format_prompt

DATA_PATH = Path(__file__).parent / "data" / "tasks.json"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_OUT = "raw_outputs.jsonl"


def load_tasks():
    return json.loads(DATA_PATH.read_text())


def get_answer_log_prob(model, tokenizer, prompt, answer, device):
    """
    Mean log-probability the model assigns to the correct answer's tokens,
    conditioned on the prompt. This is the smooth/continuous metric that
    the mirage critique argues should be used alongside exact-match.

    Tokenisation note: the answer is tokenised as " <answer>" (with a
    leading space) and appended to the prompt token IDs directly, rather
    than re-tokenising the concatenated string. Re-tokenising the joined
    string lets BPE merge the boundary characters, which previously made
    the computed answer length collapse to zero and silently discarded
    ~80% of log-probs.
    """
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    answer_ids = tokenizer(" " + answer, return_tensors="pt").input_ids

    if answer_ids.shape[1] == 0:
        return None

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)
    answer_len = answer_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits

    log_probs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for j in range(answer_len):
        # token at position (prompt_len + j) is predicted by logits at (prompt_len + j - 1)
        pos = prompt_ids.shape[1] + j - 1
        tok_id = full_ids[0, prompt_ids.shape[1] + j]
        total += log_probs[0, pos, tok_id].item()

    return total / answer_len


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=12):
    import torch

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy -> deterministic, comparable across sizes
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def extract_answer(text):
    """
    Take the first number on the first non-empty line of the continuation.
    Stopping at the line break matters for few-shot prompting, where the
    model often continues with further invented examples after answering.
    """
    import re
    first_line = text.strip().split("\n")[0]
    match = re.search(r"-?\d+", first_line)
    return match.group(0) if match else None


def drop_existing(out_path, model_name, shots):
    """Remove previously-written rows for this (model, shots) pair."""
    if not out_path.exists():
        return 0
    kept, dropped = [], 0
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("model") == model_name and r.get("shots") == shots:
            dropped += 1
        else:
            kept.append(line)
    out_path.write_text("\n".join(kept) + ("\n" if kept else ""))
    return dropped


def run_model(model_name, shots, overwrite, out_name=DEFAULT_OUT, fp16=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
    if fp16 and device != "cuda":
        print("  (--fp16 ignored: half precision needs a GPU)")
    print(f"Loading {model_name} on {device} dtype={dtype} (shots={shots}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    tasks = load_tasks()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name

    if overwrite:
        n = drop_existing(out_path, model_name, shots)
        if n:
            print(f"  removed {n} previous rows for this model/shots pair")

    n_correct = 0
    with open(out_path, "a") as f:
        for idx, t in enumerate(tasks, 1):
            prompt = format_prompt(t["expression"], shots=shots)
            generated = generate_answer(model, tokenizer, prompt, device)
            predicted = extract_answer(generated)
            exact_match = int(predicted == t["answer"]) if predicted is not None else 0
            n_correct += exact_match
            log_prob = get_answer_log_prob(model, tokenizer, prompt, t["answer"], device)

            f.write(json.dumps({
                "model": model_name,
                "shots": shots,
                "dtype": str(dtype).replace("torch.", ""),
                "task_id": t["id"],
                "tier": t["tier"],
                "axis": t["axis"],
                "num_steps": t["num_steps"],
                "operations": t.get("operations"),
                "held_out": t["held_out"],
                "prompt": prompt,
                "true_answer": t["answer"],
                "generated_text": generated,
                "predicted_answer": predicted,
                "exact_match": exact_match,
                "log_prob": log_prob,
            }) + "\n")

            if idx % 20 == 0:
                print(f"  {idx}/{len(tasks)} done  (running accuracy {n_correct/idx:.1%})")

    print(f"Finished {model_name}: {n_correct}/{len(tasks)} correct ({n_correct/len(tasks):.1%})")
    print(f"Appended to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. EleutherAI/pythia-70m")
    parser.add_argument("--shots", type=int, default=3,
                        help="number of in-context examples (0 = zero-shot). "
                             "Run BOTH 0 and 3 for each model: the comparison is "
                             "itself a result, and few-shot is what makes the "
                             "induction-head probe task-relevant.")
    parser.add_argument("--overwrite", action="store_true",
                        help="clear previous rows for this (model, shots) pair first")
    parser.add_argument("--fp16", action="store_true",
                        help="load in half precision (GPU only). Halves memory, so "
                             "pythia-6.9b fits on a 16GB card. Log-probs shift slightly "
                             "vs fp32, so use the SAME setting for every model in a sweep.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output filename inside results/. Use a DIFFERENT name on "
                             "each machine (e.g. --out raw_outputs_friend.jsonl) so that "
                             "results committed from different machines never collide in git. "
                             "analysis.py reads every results/raw_outputs*.jsonl automatically.")
    args = parser.parse_args()
    run_model(args.model, args.shots, args.overwrite, args.out, args.fp16)


if __name__ == "__main__":
    main()