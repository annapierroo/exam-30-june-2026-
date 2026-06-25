"""Local interaction diagnostics for trained neural Briscola policies."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from dataclasses import dataclass
from math import comb

import numpy as np
import torch

from evaluation import EvaluationCase, EvaluationSuite
from game.rules import NUMERO_GIOCATORI
from policy import NeuralSoftmaxPolicy, Policy

from .decision_log import record_decision_log


DEFAULT_INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("carta_briscola", "punti_presa"),
    ("carta_briscola", "carta_prende"),
    ("carta_briscola", "posizione_quarto"),
    ("carta_briscola", "fase_finale"),
    ("carta_briscola", "briscole_che_battono_non_osservate"),
    ("carta_prende", "punti_presa"),
    ("carta_prende", "posizione_quarto"),
    ("carta_liscia", "carta_prende"),
    ("carta_carico", "compagno_sta_prendendo"),
    ("carta_carico", "avversario_deve_giocare"),
    ("carta_rischiosa", "avversari_dopo"),
    ("punti_carta", "compagno_sta_prendendo"),
    ("punti_carta", "avversari_dopo"),
    ("differenza_punteggio", "fase_finale"),
    ("squadra_indietro", "fase_finale"),
    ("mazzo_vuoto", "carta_briscola"),
    ("mazzo_vuoto", "carta_carico"),
)

DEFAULT_INTERACTION_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(feature for pair in DEFAULT_INTERACTION_PAIRS for feature in pair)
)
MIN_INTERACTION_ORDER = 2
DEFAULT_MAX_INTERACTION_CANDIDATES = 10_000


@dataclass(frozen=True)
class NeuralActionSample:
    """One legal action scored by a neural policy inside a recorded decision."""

    scenario_name: str
    game_index: int
    step_index: int
    action_id: str
    chosen: bool
    probability: float
    logit: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class FeatureAttribution:
    """Aggregate local logit sensitivity for one input feature."""

    feature: str
    mean_gradient: float
    mean_abs_gradient: float
    rms_gradient: float
    feature_std: float
    weighted_strength: float
    samples: int


@dataclass(frozen=True)
class InteractionAttribution:
    """Aggregate local mixed-partial sensitivity for one feature interaction."""

    features: tuple[str, ...]
    mean_partial: float
    mean_abs_partial: float
    rms_partial: float
    feature_stds: tuple[float, ...]
    weighted_strength: float
    samples: int


@dataclass(frozen=True)
class NeuralInteractionAnalysis:
    """Feature and interaction diagnostics for one neural policy."""

    sample_count: int
    feature_attributions: tuple[FeatureAttribution, ...]
    interaction_attributions: tuple[InteractionAttribution, ...]
    interaction_candidates: tuple[tuple[str, ...], ...]


def collect_neural_action_samples(
    *,
    learner_policy: NeuralSoftmaxPolicy,
    suite: EvaluationSuite,
    cases: list[EvaluationCase],
    learner_giocatore_id: int = 0,
    greedy: bool = True,
    chosen_only: bool = False,
    max_decisions: int | None = None,
) -> list[NeuralActionSample]:
    """Collect neural logits and features from learner decisions in fixed games."""

    if max_decisions is not None and max_decisions <= 0:
        raise ValueError("max_decisions deve essere positivo o None")

    samples: list[NeuralActionSample] = []
    learner_decisions = 0
    for scenario in suite.scenarios:
        for game_index, case in enumerate(cases):
            if max_decisions is not None and learner_decisions >= max_decisions:
                return samples

            log = record_decision_log(
                policies_by_player=_policies_by_player(
                    learner_policy=learner_policy,
                    compagno_policy=scenario.compagno_policy,
                    avversario_successivo_policy=(
                        scenario.avversario_successivo_policy
                    ),
                    avversario_precedente_policy=(
                        scenario.avversario_precedente_policy
                    ),
                    learner_giocatore_id=learner_giocatore_id,
                ),
                seed_ambiente=case.seed_ambiente,
                seed_policy=case.seed_policy,
                primo_giocatore_id=case.primo_giocatore_id,
                greedy=greedy,
                focus_giocatore_id=learner_giocatore_id,
            )
            for record in log.records:
                if record.giocatore_id != learner_giocatore_id:
                    continue
                if max_decisions is not None and learner_decisions >= max_decisions:
                    return samples
                learner_decisions += 1
                samples.extend(
                    _samples_from_record(
                        learner_policy=learner_policy,
                        scenario_name=scenario.name,
                        game_index=game_index,
                        record=record,
                        chosen_only=chosen_only,
                    )
                )
    return samples


def analyze_neural_action_samples(
    *,
    policy: NeuralSoftmaxPolicy,
    samples: list[NeuralActionSample],
    interaction_pairs: tuple[tuple[str, str], ...] | None = None,
    interaction_candidates: tuple[tuple[str, ...], ...] | None = None,
    interaction_features: tuple[str, ...] | None = None,
    max_interaction_order: int = 3,
) -> NeuralInteractionAnalysis:
    """Estimate local feature and interaction sensitivities on neural logits."""

    if not samples:
        raise ValueError("Serve almeno un sample neurale da analizzare")

    feature_names = tuple(policy.feature_extractor.feature_names)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    candidates = _resolve_interaction_candidates(
        feature_names=feature_names,
        interaction_pairs=interaction_pairs,
        interaction_candidates=interaction_candidates,
        interaction_features=interaction_features,
        max_interaction_order=max_interaction_order,
    )
    _validate_interaction_candidates(candidates, feature_index)

    gradients_by_feature: list[list[float]] = [[] for _ in feature_names]
    values_by_feature: list[list[float]] = [[] for _ in feature_names]
    mixed_partials: dict[tuple[str, ...], list[float]] = {
        candidate: [] for candidate in candidates
    }
    candidate_indices = {
        candidate: tuple(feature_index[feature] for feature in candidate)
        for candidate in candidates
    }

    policy.eval()
    for sample in samples:
        gradient, sample_cross_partials = _local_logit_derivatives(
            policy=policy,
            features=sample.features,
            candidate_indices=candidate_indices,
        )
        for index, value in enumerate(sample.features):
            values_by_feature[index].append(value)
            gradients_by_feature[index].append(float(gradient[index]))
        for candidate, value in sample_cross_partials.items():
            mixed_partials[candidate].append(value)

    feature_attributions = tuple(
        sorted(
            (
                _feature_attribution(
                    feature=feature,
                    gradients=gradients_by_feature[index],
                    values=values_by_feature[index],
                )
                for index, feature in enumerate(feature_names)
            ),
            key=lambda attribution: attribution.weighted_strength,
            reverse=True,
        )
    )
    interaction_attributions = tuple(
        sorted(
            (
                _interaction_attribution(
                    candidate=candidate,
                    values=mixed_partials[candidate],
                    feature_values=tuple(
                        values_by_feature[index]
                        for index in candidate_indices[candidate]
                    ),
                )
                for candidate in candidates
            ),
            key=lambda attribution: attribution.weighted_strength,
            reverse=True,
        )
    )

    return NeuralInteractionAnalysis(
        sample_count=len(samples),
        feature_attributions=feature_attributions,
        interaction_attributions=interaction_attributions,
        interaction_candidates=candidates,
    )


def default_interaction_pairs(
    feature_names: tuple[str, ...] | list[str],
) -> tuple[tuple[str, str], ...]:
    """Return default pairs that exist in the active feature extractor."""

    available = set(feature_names)
    return tuple(
        pair
        for pair in DEFAULT_INTERACTION_PAIRS
        if pair[0] in available and pair[1] in available
    )


def default_interaction_features(
    feature_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Return default feature candidates that exist in the active extractor."""

    available = set(feature_names)
    return tuple(
        feature
        for feature in DEFAULT_INTERACTION_FEATURES
        if feature in available
    )


def default_interactions(
    feature_names: tuple[str, ...] | list[str],
    *,
    max_order: int = 3,
    max_candidates: int | None = DEFAULT_MAX_INTERACTION_CANDIDATES,
) -> tuple[tuple[str, ...], ...]:
    """Build default interaction candidates across orders 2..max_order."""

    return build_interaction_candidates(
        features=default_interaction_features(feature_names),
        max_order=max_order,
        max_candidates=max_candidates,
    )


def build_interaction_candidates(
    *,
    features: tuple[str, ...] | list[str],
    max_order: int,
    max_candidates: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Build ordered feature combinations for mixed-partial diagnostics."""

    unique_features = tuple(dict.fromkeys(features))
    _validate_max_interaction_order(max_order, feature_count=len(unique_features))
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates deve essere positivo o None")

    candidates: list[tuple[str, ...]] = []
    for order in range(MIN_INTERACTION_ORDER, max_order + 1):
        next_count = comb(len(unique_features), order)
        if (
            max_candidates is not None
            and len(candidates) + next_count > max_candidates
        ):
            raise ValueError(
                "Troppe interazioni candidate: "
                f"{len(candidates) + next_count} supera il limite {max_candidates}"
            )
        candidates.extend(combinations(unique_features, order))
    return tuple(candidates)


def neural_interaction_report_to_dict(
    *,
    checkpoint_path: str,
    checkpoint_update_index: int,
    games_per_scenario: int,
    greedy: bool,
    chosen_only: bool,
    samples: list[NeuralActionSample],
    analysis: NeuralInteractionAnalysis,
    top_n: int,
) -> dict:
    """Serialize diagnostics into a compact JSON-compatible report."""

    scenario_counts = Counter(sample.scenario_name for sample in samples)
    interaction_order_summary = _interaction_order_summary(
        analysis.interaction_attributions
    )
    candidate_order_counts = _order_counts(analysis.interaction_candidates)
    max_evaluated_order = max(candidate_order_counts) if candidate_order_counts else 0
    nonzero_orders = [
        order
        for order, summary in interaction_order_summary.items()
        if summary["nonzero_mean_abs_partial_count"] > 0
    ]
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_update_index": checkpoint_update_index,
        "games_per_scenario": games_per_scenario,
        "greedy": greedy,
        "chosen_only": chosen_only,
        "sample_count": analysis.sample_count,
        "samples_by_scenario": dict(sorted(scenario_counts.items())),
        "interaction_config": {
            "min_order": MIN_INTERACTION_ORDER,
            "max_order": max_evaluated_order,
            "max_evaluated_order": max_evaluated_order,
            "max_nonzero_order": max(nonzero_orders) if nonzero_orders else None,
            "candidate_count": len(analysis.interaction_candidates),
            "evaluated_interaction_count": len(analysis.interaction_attributions),
            "candidate_order_counts": {
                str(order): count
                for order, count in sorted(candidate_order_counts.items())
            },
            "candidate_features": sorted(
                {
                    feature
                    for candidate in analysis.interaction_candidates
                    for feature in candidate
                }
            ),
        },
        "metric_notes": {
            "feature_strength": (
                "mean_abs_gradient times observed feature standard deviation"
            ),
            "interaction_strength": (
                "mean_abs_mixed_partial times observed feature standard deviations"
            ),
            "per_order_interactions": (
                "per-order tables are sorted by mean_abs_partial to avoid comparing "
                "different orders through the standard-deviation product penalty"
            ),
            "logit_scope": (
                "scores explain local neural logits, not causal game outcomes"
            ),
        },
        "top_features": [
            _feature_attribution_to_dict(attribution)
            for attribution in analysis.feature_attributions[:top_n]
        ],
        "top_interactions": [
            _interaction_attribution_to_dict(attribution)
            for attribution in analysis.interaction_attributions[:top_n]
        ],
        "top_interactions_by_order": _top_interactions_by_order(
            analysis.interaction_attributions,
            top_n=top_n,
        ),
        "interaction_order_summary": {
            str(order): summary
            for order, summary in sorted(interaction_order_summary.items())
        },
        "all_interactions": [
            _interaction_attribution_to_dict(attribution)
            for attribution in analysis.interaction_attributions
        ],
    }


def _samples_from_record(
    *,
    learner_policy: NeuralSoftmaxPolicy,
    scenario_name: str,
    game_index: int,
    record,
    chosen_only: bool,
) -> list[NeuralActionSample]:
    with torch.no_grad():
        cards, logits = learner_policy.action_logits_tensor(record.osservazione)
        probabilities = torch.softmax(logits, dim=0)
    samples: list[NeuralActionSample] = []
    for carta, logit, probability in zip(cards, logits, probabilities):
        chosen = carta == record.azione
        if chosen_only and not chosen:
            continue
        samples.append(
            NeuralActionSample(
                scenario_name=scenario_name,
                game_index=game_index,
                step_index=record.step_index,
                action_id=carta.id,
                chosen=chosen,
                probability=float(probability.detach().item()),
                logit=float(logit.detach().item()),
                features=tuple(
                    learner_policy.feature_extractor.extract(
                        record.osservazione,
                        carta,
                    )
                ),
            )
        )
    return samples


def _policies_by_player(
    *,
    learner_policy: NeuralSoftmaxPolicy,
    compagno_policy: Policy,
    avversario_successivo_policy: Policy,
    avversario_precedente_policy: Policy,
    learner_giocatore_id: int,
) -> dict[int, Policy]:
    return {
        learner_giocatore_id: learner_policy,
        (learner_giocatore_id + 1) % NUMERO_GIOCATORI: (
            avversario_successivo_policy
        ),
        (learner_giocatore_id + 2) % NUMERO_GIOCATORI: compagno_policy,
        (learner_giocatore_id + 3) % NUMERO_GIOCATORI: (
            avversario_precedente_policy
        ),
    }


def _resolve_interaction_candidates(
    *,
    feature_names: tuple[str, ...],
    interaction_pairs: tuple[tuple[str, str], ...] | None,
    interaction_candidates: tuple[tuple[str, ...], ...] | None,
    interaction_features: tuple[str, ...] | None,
    max_interaction_order: int,
) -> tuple[tuple[str, ...], ...]:
    if interaction_candidates is not None and interaction_pairs is not None:
        raise ValueError("interaction_candidates e interaction_pairs sono alternativi")
    if interaction_candidates is not None:
        return interaction_candidates
    if interaction_pairs is not None:
        return tuple(tuple(pair) for pair in interaction_pairs)
    return build_interaction_candidates(
        features=interaction_features or default_interaction_features(feature_names),
        max_order=max_interaction_order,
    )


def _validate_interaction_candidates(
    candidates: tuple[tuple[str, ...], ...],
    feature_index: dict[str, int],
) -> None:
    if not candidates:
        raise ValueError("Serve almeno una interazione da analizzare")
    for candidate in candidates:
        if len(candidate) < MIN_INTERACTION_ORDER:
            raise ValueError("Le interazioni devono contenere almeno due feature")
        if len(set(candidate)) != len(candidate):
            raise ValueError("Le feature di una interazione devono essere diverse")
        for feature in candidate:
            if feature not in feature_index:
                raise ValueError(f"Feature non trovata: {feature}")


def _validate_max_interaction_order(max_order: int, *, feature_count: int) -> None:
    if max_order < MIN_INTERACTION_ORDER:
        raise ValueError(f"max_order deve essere almeno {MIN_INTERACTION_ORDER}")
    if feature_count < MIN_INTERACTION_ORDER:
        raise ValueError("Servono almeno due feature candidate")
    if max_order > feature_count:
        raise ValueError(
            f"max_order deve essere al massimo il numero di feature candidate ({feature_count})"
        )


def _local_logit_derivatives(
    *,
    policy: NeuralSoftmaxPolicy,
    features: tuple[float, ...],
    candidate_indices: dict[tuple[str, ...], tuple[int, ...]],
) -> tuple[np.ndarray, dict[tuple[str, ...], float]]:
    # The current neural policy is a one-hidden-layer tanh MLP. Closed-form
    # derivatives keep this diagnostic fast enough for full evaluation suites.
    with torch.no_grad():
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        hidden_pre_activation = policy.hidden_layer(feature_tensor)
        hidden_activation = torch.tanh(hidden_pre_activation)
        output_weights = policy.output_layer.weight.squeeze(0)
        first_factor = output_weights * (1.0 - hidden_activation.pow(2))
        hidden_weights = policy.hidden_layer.weight
        gradient = torch.mv(hidden_weights.t(), first_factor)

        derivative_factors = {
            order: output_weights * _tanh_derivative(hidden_activation, order)
            for order in sorted({len(indices) for indices in candidate_indices.values()})
        }
        mixed_partials = {
            candidate: float(
                torch.sum(
                    derivative_factors[len(indices)]
                    * _hidden_weight_product(hidden_weights, indices)
                ).item()
            )
            for candidate, indices in candidate_indices.items()
        }

    return gradient.cpu().numpy(), mixed_partials


def _tanh_derivative(hidden_activation: torch.Tensor, order: int) -> torch.Tensor:
    if order < 1:
        raise ValueError("order deve essere positivo")

    result = torch.zeros_like(hidden_activation)
    for power, coefficient in enumerate(_tanh_derivative_coefficients(order)):
        if coefficient == 0.0:
            continue
        if power == 0:
            result = result + coefficient
        else:
            result = result + coefficient * hidden_activation.pow(power)
    return result


@lru_cache(maxsize=None)
def _tanh_derivative_coefficients(order: int) -> tuple[float, ...]:
    # P_1(t) = 1 - t^2, with t = tanh(z). Higher derivatives follow
    # P_{n+1}(t) = (1 - t^2) * dP_n(t)/dt.
    coefficients = (1.0, 0.0, -1.0)
    for _ in range(1, order):
        coefficients = _next_tanh_derivative_coefficients(coefficients)
    return coefficients


def _next_tanh_derivative_coefficients(
    coefficients: tuple[float, ...],
) -> tuple[float, ...]:
    next_coefficients = [0.0 for _ in range(len(coefficients) + 1)]
    for power, coefficient in enumerate(coefficients):
        if power == 0 or coefficient == 0.0:
            continue
        derivative_coefficient = power * coefficient
        next_coefficients[power - 1] += derivative_coefficient
        next_coefficients[power + 1] -= derivative_coefficient
    return tuple(next_coefficients)


def _hidden_weight_product(
    hidden_weights: torch.Tensor,
    indices: tuple[int, ...],
) -> torch.Tensor:
    product = torch.ones_like(hidden_weights[:, indices[0]])
    for index in indices:
        product = product * hidden_weights[:, index]
    return product


def _feature_attribution(
    *,
    feature: str,
    gradients: list[float],
    values: list[float],
) -> FeatureAttribution:
    gradient_array = np.asarray(gradients, dtype=np.float64)
    feature_std = _std(values)
    mean_abs_gradient = float(np.mean(np.abs(gradient_array)))
    return FeatureAttribution(
        feature=feature,
        mean_gradient=float(np.mean(gradient_array)),
        mean_abs_gradient=mean_abs_gradient,
        rms_gradient=_rms(gradient_array),
        feature_std=feature_std,
        weighted_strength=mean_abs_gradient * feature_std,
        samples=len(gradients),
    )


def _interaction_attribution(
    *,
    candidate: tuple[str, ...],
    values: list[float],
    feature_values: tuple[list[float], ...],
) -> InteractionAttribution:
    value_array = np.asarray(values, dtype=np.float64)
    feature_stds = tuple(_std(values) for values in feature_values)
    mean_abs_partial = float(np.mean(np.abs(value_array)))
    return InteractionAttribution(
        features=candidate,
        mean_partial=float(np.mean(value_array)),
        mean_abs_partial=mean_abs_partial,
        rms_partial=_rms(value_array),
        feature_stds=feature_stds,
        weighted_strength=mean_abs_partial * _product(feature_stds),
        samples=len(values),
    )


def _product(values: tuple[float, ...]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _order_counts(candidates: tuple[tuple[str, ...], ...]) -> dict[int, int]:
    counts = Counter(len(candidate) for candidate in candidates)
    return dict(sorted(counts.items()))


def _interaction_order_summary(
    attributions: tuple[InteractionAttribution, ...],
) -> dict[int, dict[str, float | int]]:
    grouped: dict[int, list[InteractionAttribution]] = defaultdict(list)
    for attribution in attributions:
        grouped[len(attribution.features)].append(attribution)

    summary: dict[int, dict[str, float | int]] = {}
    for order, rows in sorted(grouped.items()):
        summary[order] = {
            "candidate_count": len(rows),
            "nonzero_weighted_strength_count": sum(
                1 for row in rows if row.weighted_strength > 0.0
            ),
            "nonzero_mean_abs_partial_count": sum(
                1 for row in rows if row.mean_abs_partial > 0.0
            ),
            "max_weighted_strength": max(row.weighted_strength for row in rows),
            "max_mean_abs_partial": max(row.mean_abs_partial for row in rows),
        }
    return summary


def _top_interactions_by_order(
    attributions: tuple[InteractionAttribution, ...],
    *,
    top_n: int,
) -> dict[str, list[dict]]:
    grouped: dict[int, list[InteractionAttribution]] = defaultdict(list)
    for attribution in attributions:
        grouped[len(attribution.features)].append(attribution)

    return {
        str(order): [
            _interaction_attribution_to_dict(attribution)
            for attribution in sorted(
                rows,
                key=lambda attribution: attribution.mean_abs_partial,
                reverse=True,
            )[:top_n]
        ]
        for order, rows in sorted(grouped.items())
    }


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _feature_attribution_to_dict(attribution: FeatureAttribution) -> dict:
    return {
        "feature": attribution.feature,
        "weighted_strength": attribution.weighted_strength,
        "mean_abs_gradient": attribution.mean_abs_gradient,
        "mean_gradient": attribution.mean_gradient,
        "rms_gradient": attribution.rms_gradient,
        "feature_std": attribution.feature_std,
        "samples": attribution.samples,
    }


def _interaction_attribution_to_dict(
    attribution: InteractionAttribution,
) -> dict:
    payload = {
        "features": list(attribution.features),
        "order": len(attribution.features),
        "weighted_strength": attribution.weighted_strength,
        "mean_abs_partial": attribution.mean_abs_partial,
        "mean_partial": attribution.mean_partial,
        "rms_partial": attribution.rms_partial,
        "feature_stds": list(attribution.feature_stds),
        "samples": attribution.samples,
    }
    if len(attribution.features) >= 2:
        payload["feature_a"] = attribution.features[0]
        payload["feature_b"] = attribution.features[1]
        payload["mean_abs_cross_partial"] = attribution.mean_abs_partial
        payload["mean_cross_partial"] = attribution.mean_partial
    return payload
