# Instructions for running the large models

You are running **pythia-1.4b** and **pythia-2.8b**, which need more RAM than
the primary machine has. Everything else is already done.

## Requirements

- Python 3.9+
- ~8 GB free RAM for 1.4b, ~16 GB for 2.8b
- ~15 GB free disk (model downloads are cached)
- A stable internet connection for the first run of each model

## Setup

```bash
git clone <REPO_URL>
cd emergence_experiment
python -m venv venv
source venv/bin/activate        # Windows PowerShell: venv\Scripts\activate
pip install -r requirements.txt
```

## Important: do NOT regenerate the benchmark

`data/tasks.json` is committed to the repo and must be used exactly as-is.
Do **not** run `python benchmark.py` — that would regenerate the task file and
your results would no longer be comparable with the runs from the other
machine.

## Run these six commands

Use `--out raw_outputs_big.jsonl` on every command. This writes to a separate
file so your results never conflict with the other machine's in git.

```bash
python inference.py --model EleutherAI/pythia-1.4b --shots 3 --overwrite --out raw_outputs_big.jsonl
python inference.py --model EleutherAI/pythia-1.4b --shots 0 --overwrite --out raw_outputs_big.jsonl
python inference.py --model EleutherAI/pythia-2.8b --shots 3 --overwrite --out raw_outputs_big.jsonl
python inference.py --model EleutherAI/pythia-2.8b --shots 0 --overwrite --out raw_outputs_big.jsonl
```

Expect this to take a while on CPU — roughly 1-2 hours for 1.4b and 3-6 hours
for 2.8b, depending on the machine. It prints progress every 20 items. If it
gets interrupted, just re-run the same command; `--overwrite` clears the partial
rows for that model first.

If 2.8b runs out of memory, skip it and report back — 1.4b alone is still
valuable.

## Optional: the mechanistic probe

Only if `pip install transformer_lens` succeeded:

```bash
python induction_probe.py --model pythia-1.4b
python induction_probe.py --model pythia-2.8b
```

Note the different naming convention here (`pythia-1.4b`, not
`EleutherAI/pythia-1.4b`) — that is a library convention, not a typo.

## Send the results back

```bash
git add results/raw_outputs_big.jsonl results/induction_scores.jsonl
git commit -m "Add 1.4b and 2.8b results"
git push
```

That's it. Do not run `scoring.py` or `analysis.py` — those are run on the
primary machine once all results are collected.
