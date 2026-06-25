# Diagnostics

Utilities for inspecting policy decisions from legal observations.

## Files

- `decision_log.py`: records one complete game as a sequence of decisions.
- `neural_interactions.py`: estimates local feature and pairwise interaction
  sensitivities for neural checkpoints.
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

## Neural Interactions

`scripts/analyze_neural_interactions.py` loads a neural checkpoint, replays the
fixed evaluation suite, and measures:

- feature sensitivity: mean absolute local gradient of the neural logit;
- pair interaction sensitivity: mean absolute local cross-partial derivative of
  the neural logit for selected feature pairs.

Run it with:

```bash
python scripts/analyze_neural_interactions.py \
  --checkpoint experiments/results/checkpoint.json \
  --games 100 \
  --output experiments/results/neural_interactions.json
```

The report explains local neural logits, not causal game outcomes. Engineered
interaction features such as `briscola_x_punti_presa` appear as ordinary input
features in the feature-sensitivity table.

## Views

`views.py` provides filters for:

- player;
- policy name;
- trick position;
- partner currently leading;
- opponent currently leading;
- rich tricks;
- chosen actions below a probability threshold.
