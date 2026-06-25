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
    DEFAULT_MAX_INTERACTION_CANDIDATES,
    analyze_neural_action_samples,
    collect_neural_action_samples,
    default_interactions,
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
    parser.add_argument("--max-interaction-order", type=int, default=3)
    parser.add_argument(
        "--max-interaction-candidates",
        type=int,
        default=DEFAULT_MAX_INTERACTION_CANDIDATES,
        help=(
            "Maximum default interaction candidates to generate. Increase this "
            "explicitly for high-order scans."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--interaction",
        action="append",
        type=parse_interaction,
        default=None,
        help=(
            "Custom interaction as feature_a:feature_b[:feature_c...]. "
            "Repeat for multiple custom interactions."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games deve essere positivo")
    if args.max_decisions < 0:
        parser.error("--max-decisions deve essere non negativo")
    if args.top <= 0:
        parser.error("--top deve essere positivo")
    if args.max_interaction_order < 2:
        parser.error("--max-interaction-order deve essere almeno 2")
    if args.max_interaction_candidates <= 0:
        parser.error("--max-interaction-candidates deve essere positivo")
    if args.pair and args.interaction:
        parser.error("--pair e --interaction sono alternativi")
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
    interaction_candidates = (
        tuple(args.interaction or args.pair)
        if args.interaction or args.pair
        else default_interactions(
            learner.feature_extractor.feature_names,
            max_order=args.max_interaction_order,
            max_candidates=args.max_interaction_candidates,
        )
    )
    analysis = analyze_neural_action_samples(
        policy=learner,
        samples=samples,
        interaction_candidates=interaction_candidates,
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

    parts = parse_interaction(value)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Feature pairs must use the feature_a:feature_b format"
        )
    return parts[0], parts[1]


def parse_interaction(value: str) -> tuple[str, ...]:
    """Parse one CLI feature interaction."""

    parts = tuple(part for part in value.split(":") if part)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "Interactions must contain at least 2 features separated by ':'"
        )
    if len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError("Interaction features must be distinct")
    return parts


def default_output_path(checkpoint_path: Path, games: int) -> Path:
    """Save the report next to the evaluated checkpoint."""

    return checkpoint_path.parent / f"neural_interactions_games_{games}.json"


def print_report_summary(analysis, *, top_n: int) -> None:
    """Print compact interaction and feature tables."""

    print(f"samples={analysis.sample_count}")
    print(f"candidate_interactions={len(analysis.interaction_candidates)}")
    print_order_counts(analysis)
    print("\nTop feature sensitivities")
    print(f"{'feature':42} {'strength':>12} {'mean_abs_grad':>14} {'std':>9}")
    for attribution in analysis.feature_attributions[:top_n]:
        print(
            f"{attribution.feature:42} "
            f"{attribution.weighted_strength:12.6f} "
            f"{attribution.mean_abs_gradient:14.6f} "
            f"{attribution.feature_std:9.4f}"
        )

    print("\nTop interactions")
    print(
        f"{'features':74} {'order':>5} "
        f"{'strength':>12} {'mean_abs_partial':>16}"
    )
    for attribution in analysis.interaction_attributions[:top_n]:
        print(
            f"{format_interaction(attribution.features):74} "
            f"{len(attribution.features):5d} "
            f"{attribution.weighted_strength:12.6f} "
            f"{attribution.mean_abs_partial:16.6f}"
        )
    print_top_interactions_by_order(analysis, top_n=top_n)


def print_order_counts(analysis) -> None:
    """Print how many candidate interactions were evaluated per order."""

    counts: dict[int, int] = {}
    for candidate in analysis.interaction_candidates:
        counts[len(candidate)] = counts.get(len(candidate), 0) + 1
    print(
        "interaction_orders="
        + ", ".join(f"{order}:{count}" for order, count in sorted(counts.items()))
    )


def print_top_interactions_by_order(analysis, *, top_n: int) -> None:
    """Print order-specific interaction tables sorted by raw mixed partials."""

    grouped: dict[int, list] = {}
    for attribution in analysis.interaction_attributions:
        grouped.setdefault(len(attribution.features), []).append(attribution)

    for order, rows in sorted(grouped.items()):
        print(f"\nTop interactions of order {order} by mean_abs_partial")
        print(
            f"{'features':74} {'order':>5} "
            f"{'strength':>12} {'mean_abs_partial':>16}"
        )
        for attribution in sorted(
            rows,
            key=lambda item: item.mean_abs_partial,
            reverse=True,
        )[:top_n]:
            print(
                f"{format_interaction(attribution.features):74} "
                f"{len(attribution.features):5d} "
                f"{attribution.weighted_strength:12.6f} "
                f"{attribution.mean_abs_partial:16.6f}"
            )


def format_interaction(features: tuple[str, ...]) -> str:
    """Return a readable interaction label."""

    return " x ".join(features)


if __name__ == "__main__":
    main()
