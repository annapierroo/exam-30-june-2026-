#!/usr/bin/env python3
"""CLI entrypoint to inspect local neural feature interactions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python scripts/analyze_neural_interactions.py` without installation.
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics import (
    analyze_neural_action_samples,
    collect_neural_action_samples,
    default_interaction_pairs,
    neural_interaction_report_to_dict,
)
from evaluation import default_evaluation_suite, make_evaluation_cases
from evaluate import learner_from_checkpoint, load_checkpoint
from policy import NeuralSoftmaxPolicy


def parse_args() -> argparse.Namespace:
    """Read neural interaction diagnostic parameters from the CLI."""

    parser = argparse.ArgumentParser(
        description="Analyze local feature interactions in a neural checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-ambiente-start", type=int, default=100_000)
    parser.add_argument("--seed-policy-start", type=int, default=200_000)
    parser.add_argument("--learner-giocatore-id", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--chosen-only", action="store_true")
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        default=None,
        help="Feature pair as feature_a:feature_b. Repeat for custom pairs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games deve essere positivo")
    if args.max_decisions < 0:
        parser.error("--max-decisions deve essere non negativo")
    if args.top <= 0:
        parser.error("--top deve essere positivo")
    return args


def main() -> None:
    args = parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    learner = learner_from_checkpoint(checkpoint)
    if not isinstance(learner, NeuralSoftmaxPolicy):
        raise ValueError("La diagnostica delle interazioni richiede una policy neural")

    cases = make_evaluation_cases(
        games=args.games,
        seed_ambiente_start=args.seed_ambiente_start,
        seed_policy_start=args.seed_policy_start,
    )
    samples = collect_neural_action_samples(
        learner_policy=learner,
        suite=default_evaluation_suite(),
        cases=cases,
        learner_giocatore_id=args.learner_giocatore_id,
        greedy=not args.stochastic,
        chosen_only=args.chosen_only,
        max_decisions=args.max_decisions or None,
    )
    interaction_pairs = (
        tuple(args.pair)
        if args.pair
        else default_interaction_pairs(learner.feature_extractor.feature_names)
    )
    analysis = analyze_neural_action_samples(
        policy=learner,
        samples=samples,
        interaction_pairs=interaction_pairs,
    )

    output_path = args.output or default_output_path(args.checkpoint, args.games)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            neural_interaction_report_to_dict(
                checkpoint_path=str(args.checkpoint),
                checkpoint_update_index=checkpoint["update_index"],
                games_per_scenario=args.games,
                greedy=not args.stochastic,
                chosen_only=args.chosen_only,
                samples=samples,
                analysis=analysis,
                top_n=args.top,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print_report_summary(analysis, top_n=args.top)
    print(f"saved_report={output_path}")


def parse_pair(value: str) -> tuple[str, str]:
    """Parse one CLI feature pair."""

    parts = value.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            "Feature pairs must use the feature_a:feature_b format"
        )
    return parts[0], parts[1]


def default_output_path(checkpoint_path: Path, games: int) -> Path:
    """Save the report next to the evaluated checkpoint."""

    return checkpoint_path.parent / f"neural_interactions_games_{games}.json"


def print_report_summary(analysis, *, top_n: int) -> None:
    """Print compact interaction and feature tables."""

    print(f"samples={analysis.sample_count}")
    print("\nTop feature sensitivities")
    print(f"{'feature':42} {'strength':>12} {'mean_abs_grad':>14} {'std':>9}")
    for attribution in analysis.feature_attributions[:top_n]:
        print(
            f"{attribution.feature:42} "
            f"{attribution.weighted_strength:12.6f} "
            f"{attribution.mean_abs_gradient:14.6f} "
            f"{attribution.feature_std:9.4f}"
        )

    print("\nTop pair interactions")
    print(
        f"{'feature_a':32} {'feature_b':32} "
        f"{'strength':>12} {'mean_abs_cross':>15}"
    )
    for attribution in analysis.interaction_attributions[:top_n]:
        print(
            f"{attribution.feature_a:32} "
            f"{attribution.feature_b:32} "
            f"{attribution.weighted_strength:12.6f} "
            f"{attribution.mean_abs_cross_partial:15.6f}"
        )


if __name__ == "__main__":
    main()
