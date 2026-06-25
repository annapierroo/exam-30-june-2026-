# Diagnostics UI

Open `visual_diagnostics.html` and load a diagnostics JSON file.

The UI accepts:

- decision logs produced by `scripts/record_diagnostics.py`;
- neural interaction reports produced by `scripts/analyze_neural_interactions.py`.

Use `Hands on` to reveal every player's hand during debugging.

Start a local server from the project root:

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

## Self-Play Diagnostic Game

Generate a game where the learner plays with and against copies of itself:

```bash
.venv/bin/python scripts/record_diagnostics.py \
  --checkpoint experiments/results/checkpoint.json \
  --partner learner \
  --opponents learner \
  --output experiments/results/diagnostics_self_play.json
```

Open:

```text
http://127.0.0.1:8765/diagnostics_ui/visual_diagnostics.html?json=/experiments/results/diagnostics_self_play.json
```

## Learner vs Perfect Heuristic

Generate a game where the learner plays with a perfect heuristic partner against
perfect heuristic opponents:

```bash
.venv/bin/python scripts/record_diagnostics.py \
  --checkpoint experiments/results/checkpoint.json \
  --partner perfect \
  --opponents perfect \
  --output experiments/results/diagnostics_vs_perfect.json
```

Open:

```text
http://127.0.0.1:8765/diagnostics_ui/visual_diagnostics.html?json=/experiments/results/diagnostics_vs_perfect.json
```

## Neural Interaction Report

Generate an aggregate neural interaction report:

```bash
.venv/bin/python scripts/analyze_neural_interactions.py \
  --checkpoint experiments/results/checkpoint.json \
  --games 100 \
  --output experiments/results/neural_interactions_games_100.json
```

Open:

```text
http://127.0.0.1:8765/diagnostics_ui/visual_diagnostics.html?json=/experiments/results/neural_interactions_games_100.json
```

This view shows aggregate feature sensitivities and mixed interactions for the
neural logits. It includes a global interaction table and separate tables for
each interaction order available in the report. It does not replace the
decision-log view for individual games.

## Available Policies

Use `--partner` to choose the learner's teammate and `--opponents` to choose
both opponents.

Available values:

```text
learner
random
greedy
heuristic
advanced
perfect
```

Example: learner partner vs perfect heuristic opponents:

```bash
.venv/bin/python scripts/record_diagnostics.py \
  --checkpoint experiments/results/checkpoint.json \
  --partner learner \
  --opponents perfect \
  --output experiments/results/diagnostics_learner_partner_vs_perfect.json
```

## Reproducible Initial Games

Use these options to choose the initial game:

```text
--seed-ambiente      controls deck shuffle, briscola, and initial hands
--seed-policy        controls stochastic policy choices
--primo-giocatore-id controls who starts the first trick
```

Example:

```bash
.venv/bin/python scripts/record_diagnostics.py \
  --checkpoint experiments/results/checkpoint.json \
  --partner perfect \
  --opponents perfect \
  --seed-ambiente 123 \
  --seed-policy 456 \
  --primo-giocatore-id 0 \
  --output experiments/results/diagnostics_custom_seed.json
```

Use the same three values across different commands to compare games with the
same initial deck, briscola, and hands.
