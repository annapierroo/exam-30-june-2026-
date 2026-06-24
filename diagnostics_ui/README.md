# Diagnostics UI

Open `visual_diagnostics.html` and load a diagnostics JSON file.

The UI expects the diagnostics JSON produced by `scripts/record_diagnostics.py`.
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
