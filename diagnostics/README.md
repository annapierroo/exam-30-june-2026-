# Diagnostics

Utilities for inspecting policy decisions from legal observations.

## Files

- `decision_log.py`: records one complete game as a sequence of decisions.
- `views.py`: filters decision logs into readable subsets.

## Decision Log

`record_decision_log` stores, for each decision:

- player id;
- policy name;
- legal observation before the action;
- legal actions;
- action probabilities;
- chosen action;
- action-selection mode;
- public outcome after the action.

The log stores public outcomes and legal observations only.

Record one decision trace from a checkpoint with:

```bash
python scripts/record_diagnostics.py \
  --checkpoint experiments/results/checkpoint.json \
  --output experiments/results/decisions.json
```

Print it with:

```bash
python scripts/show_diagnostics.py experiments/results/decisions.json --show-probs
```

Or inspect it visually by opening:

```text
diagnostics_ui/visual_diagnostics.html
```

Then load `experiments/results/decisions.json` with the file picker.

## Views

`views.py` provides filters for:

- player;
- policy name;
- trick position;
- partner currently leading;
- opponent currently leading;
- rich tricks;
- chosen actions below a probability threshold.
