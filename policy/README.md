# policy

This folder contains policy interfaces and baseline policies for four-player
Briscola.

Policies act only from a legal `Osservazione` and return legal cards from the
current hand.

## Interface

All policies implement the minimal `Policy` protocol:

- `name`: policy identifier;
- `action_probabilities(osservazione)`: probability assigned to each legal card;
- `select_action(osservazione, rng, greedy=False)`: selected legal card.

## Policies

- `RandomPolicy`: uniform random choice among legal cards.
- `GreedyPolicy`: myopic baseline; takes with the least costly sufficient card,
  otherwise discards the least costly card.
- `HeuristicPolicy`: minimal team-aware baseline; when the partner is taking,
  avoids spending valuable cards.
- `AdvancedHeuristicPolicy`: explicit rule-based heuristic for richer team-aware
  play. It separates cases by current winner, player position in the trick, and
  trick value.

## Features

- `BriscolaFeatureExtractor`: builds numeric features from a legal
  `Osservazione` and a legal candidate card. It is intended for learnable
  policies such as a future linear softmax policy, and does not use hidden game
  state.

## Tests

Run policy tests from the repository root:

```bash
python3 -B -m unittest discover -s policy/tests
```
