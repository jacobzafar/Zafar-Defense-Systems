# Frozen evaluation harness

This is the **only** place in this repo that produces accuracy metrics
(AP@0.5, small-object recall, false-alarm rate, latency, track
continuity) for a detector backend. `detector/train.py`'s own
`training_report.json` deliberately does not compute any of these — see
`docs/DECISIONS.md` entry #9.

## The eval set must be frozen

"Frozen" means: once an eval set is finalized, its images and labels must
never change, and it must never be used to train or fine-tune anything.
Comparing results across runs/models is only meaningful if the eval set
underneath them is identical.

This is enforced two ways, not just documented:

1. **A `.frozen` marker file.** `eval/schema.py`'s `assert_not_for_training()`
   is called by `detector/train.py` before it loads any dataset — if the
   target directory contains a `.frozen` file, training refuses to start
   with a clear error. Create the marker with
   `eval.schema.freeze_eval_set(eval_set_dir)` once an eval set is final.
2. **A manifest checksum.** `eval/schema.py`'s `compute_manifest_checksum()`
   hashes `sequences.json`. Record that checksum in your eval config's
   `expected_checksum` field; the harness (`eval/harness.py`) verifies it
   on every run and fails loudly if it doesn't match — catching
   accidental edits to the frozen set. Note this hashes the label file,
   not image pixel bytes — it catches label/structure tampering, not a
   swapped image file with the same name.

If an eval set directory has no `.frozen` marker, the harness still runs
(useful while you're still assembling one) but prints a warning that
results aren't safely comparable yet.

## Directory layout

See the module docstring in `eval/schema.py` for the exact
`sequences.json` schema. In short: `<eval_set_dir>/images/` +
`<eval_set_dir>/sequences.json`, where each sequence is an ordered list of
frames (so track continuity can be scored), and each frame has zero or
more `{"bbox": [x,y,w,h], "category": "drone"}`-style boxes. A frame with
no "drone" box is a hard-negative frame (background/bird/clutter),
which is what the false-alarm-rate metric evaluates against.

## Running it

```bash
python -m eval.harness --config path/to/eval_config.yaml
```

See `EvalConfig` in `eval/harness.py` for every field (eval set path,
detector config dict, tracker config dict, IoU threshold, small-object
area threshold, output paths). Produces:

- a JSON metric card (default `eval/output/metric_card.json`)
- a Markdown report (default `eval/output/REPORT.md`) with a "Baseline
  comparison" section — an empty table row for you to paste published
  baseline numbers into, with a citation, so this run's numbers can be
  read next to a known reference rather than in isolation.

`eval/output/` is gitignored — every run regenerates it, and committing a
stale metric card risks it being mistaken for a current, validated result.

## What this harness does NOT do

- It does not claim any number is good or bad — that's a judgment call
  for whoever reads the report next to the baseline comparison section.
- It does not benchmark on real drone footage by default — no real eval
  set ships with this repo (see `docs/known-limitations.md`). A tiny
  synthetic fixture exists purely so this harness itself is testable
  (`tests/test_eval_harness.py`) — never mistake its output for a real
  accuracy result.
- It does not select or recommend a "best" backend automatically. It
  reports numbers; comparing them against your own bar (or the pasted
  baseline) is a human decision.
