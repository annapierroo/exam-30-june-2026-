#!/usr/bin/env python3
"""Record one visualizable diagnostic game from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python scripts/record_diagnostics.py` without installing the project.
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics import decision_log_to_dict, record_decision_log
from policy import (
    AdvancedHeuristicPolicy,
    GreedyPolicy,
    HeuristicPolicy,
    PerfectHeuristicPolicy,
    Policy,
    RandomPolicy,
)
from scripts.evaluate import load_checkpoint, learner_from_checkpoint


POLICY_CHOICES = ("learner", "random", "greedy", "heuristic", "advanced", "perfect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one Briscola diagnostic game for the visual UI.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learner-giocatore-id", type=int, default=0)
    parser.add_argument("--partner", choices=POLICY_CHOICES, default="perfect")
    parser.add_argument("--opponents", choices=POLICY_CHOICES, default="perfect")
    parser.add_argument("--seed-ambiente", type=int, default=100_000)
    parser.add_argument("--seed-policy", type=int, default=200_000)
    parser.add_argument("--primo-giocatore-id", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.learner_giocatore_id not in range(4):
        parser.error("--learner-giocatore-id deve essere tra 0 e 3")
    if args.primo_giocatore_id not in range(4):
        parser.error("--primo-giocatore-id deve essere tra 0 e 3")
    return args


def main() -> None:
    args = parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    learner = learner_from_checkpoint(checkpoint)
    policies_by_player = policies_for_game(
        learner=learner,
        learner_giocatore_id=args.learner_giocatore_id,
        partner_policy=args.partner,
        opponent_policy=args.opponents,
    )
    log = record_decision_log(
        policies_by_player=policies_by_player,
        seed_ambiente=args.seed_ambiente,
        seed_policy=args.seed_policy,
        primo_giocatore_id=args.primo_giocatore_id,
        greedy=not args.stochastic,
        focus_giocatore_id=args.learner_giocatore_id,
    )

    report = {
        "kind": "briscola_rl_4players_decision_diagnostics",
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_update_index": checkpoint["update_index"],
        "learner_giocatore_id": args.learner_giocatore_id,
        "partner_policy": args.partner,
        "opponent_policy": args.opponents,
        "policy_assignment": {
            str(giocatore_id): policy.name
            for giocatore_id, policy in policies_by_player.items()
        },
        "decision_log": decision_log_to_dict(log),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved_diagnostics={args.output}")


def policies_for_game(
    *,
    learner: Policy,
    learner_giocatore_id: int,
    partner_policy: str,
    opponent_policy: str,
) -> dict[int, Policy]:
    partner_id = (learner_giocatore_id + 2) % 4
    policies: dict[int, Policy] = {}
    for giocatore_id in range(4):
        if giocatore_id == learner_giocatore_id:
            policies[giocatore_id] = learner
        elif giocatore_id == partner_id:
            policies[giocatore_id] = make_policy(
                partner_policy,
                f"{partner_policy}_partner",
                learner=learner,
            )
        else:
            policies[giocatore_id] = make_policy(
                opponent_policy,
                f"{opponent_policy}_opponent_p{giocatore_id}",
                learner=learner,
            )
    return policies


def make_policy(policy_name: str, name: str, *, learner: Policy) -> Policy:
    if policy_name == "learner":
        if not hasattr(learner, "copy"):
            raise ValueError("La policy learner non supporta copy()")
        return learner.copy(name=name)
    if policy_name == "random":
        return RandomPolicy(name=name)
    if policy_name == "greedy":
        return GreedyPolicy(name=name)
    if policy_name == "heuristic":
        return HeuristicPolicy(name=name)
    if policy_name == "advanced":
        return AdvancedHeuristicPolicy(name=name)
    if policy_name == "perfect":
        return PerfectHeuristicPolicy(name=name)
    raise ValueError(f"Policy non supportata: {policy_name}")


if __name__ == "__main__":
    main()
