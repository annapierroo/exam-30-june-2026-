"""Local interaction diagnostics for trained neural Briscola policies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

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
    """Aggregate local cross-partial sensitivity for one feature pair."""

    feature_a: str
    feature_b: str
    mean_cross_partial: float
    mean_abs_cross_partial: float
    rms_cross_partial: float
    feature_a_std: float
    feature_b_std: float
    weighted_strength: float
    samples: int


@dataclass(frozen=True)
class NeuralInteractionAnalysis:
    """Feature and pairwise interaction diagnostics for one neural policy."""

    sample_count: int
    feature_attributions: tuple[FeatureAttribution, ...]
    interaction_attributions: tuple[InteractionAttribution, ...]


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
) -> NeuralInteractionAnalysis:
    """Estimate local feature and interaction sensitivities on neural logits."""

    if not samples:
        raise ValueError("Serve almeno un sample neurale da analizzare")

    feature_names = tuple(policy.feature_extractor.feature_names)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    pairs = interaction_pairs or default_interaction_pairs(feature_names)
    _validate_interaction_pairs(pairs, feature_index)

    gradients_by_feature: list[list[float]] = [[] for _ in feature_names]
    values_by_feature: list[list[float]] = [[] for _ in feature_names]
    cross_partials: dict[tuple[str, str], list[float]] = {
        pair: [] for pair in pairs
    }
    pair_indices = {
        pair: (feature_index[pair[0]], feature_index[pair[1]]) for pair in pairs
    }

    policy.eval()
    for sample in samples:
        gradient, sample_cross_partials = _local_logit_derivatives(
            policy=policy,
            features=sample.features,
            pair_indices=pair_indices,
        )
        for index, value in enumerate(sample.features):
            values_by_feature[index].append(value)
            gradients_by_feature[index].append(float(gradient[index]))
        for pair, value in sample_cross_partials.items():
            cross_partials[pair].append(value)

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
                    pair=pair,
                    values=cross_partials[pair],
                    feature_a_values=values_by_feature[pair_indices[pair][0]],
                    feature_b_values=values_by_feature[pair_indices[pair][1]],
                )
                for pair in pairs
            ),
            key=lambda attribution: attribution.weighted_strength,
            reverse=True,
        )
    )

    return NeuralInteractionAnalysis(
        sample_count=len(samples),
        feature_attributions=feature_attributions,
        interaction_attributions=interaction_attributions,
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
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_update_index": checkpoint_update_index,
        "games_per_scenario": games_per_scenario,
        "greedy": greedy,
        "chosen_only": chosen_only,
        "sample_count": analysis.sample_count,
        "samples_by_scenario": dict(sorted(scenario_counts.items())),
        "metric_notes": {
            "feature_strength": (
                "mean_abs_gradient times observed feature standard deviation"
            ),
            "interaction_strength": (
                "mean_abs_cross_partial times both observed feature standard deviations"
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


def _validate_interaction_pairs(
    pairs: tuple[tuple[str, str], ...],
    feature_index: dict[str, int],
) -> None:
    if not pairs:
        raise ValueError("Serve almeno una coppia di feature da analizzare")
    for first, second in pairs:
        if first not in feature_index:
            raise ValueError(f"Feature non trovata: {first}")
        if second not in feature_index:
            raise ValueError(f"Feature non trovata: {second}")
        if first == second:
            raise ValueError("Le feature di una coppia devono essere diverse")


def _local_logit_derivatives(
    *,
    policy: NeuralSoftmaxPolicy,
    features: tuple[float, ...],
    pair_indices: dict[tuple[str, str], tuple[int, int]],
) -> tuple[np.ndarray, dict[tuple[str, str], float]]:
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

        second_factor = (
            output_weights
            * (-2.0 * hidden_activation * (1.0 - hidden_activation.pow(2)))
        )
        cross_partials = {
            pair: float(
                torch.sum(
                    second_factor
                    * hidden_weights[:, first_index]
                    * hidden_weights[:, second_index]
                ).item()
            )
            for pair, (first_index, second_index) in pair_indices.items()
        }

    return gradient.cpu().numpy(), cross_partials


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
    pair: tuple[str, str],
    values: list[float],
    feature_a_values: list[float],
    feature_b_values: list[float],
) -> InteractionAttribution:
    value_array = np.asarray(values, dtype=np.float64)
    feature_a_std = _std(feature_a_values)
    feature_b_std = _std(feature_b_values)
    mean_abs_cross_partial = float(np.mean(np.abs(value_array)))
    return InteractionAttribution(
        feature_a=pair[0],
        feature_b=pair[1],
        mean_cross_partial=float(np.mean(value_array)),
        mean_abs_cross_partial=mean_abs_cross_partial,
        rms_cross_partial=_rms(value_array),
        feature_a_std=feature_a_std,
        feature_b_std=feature_b_std,
        weighted_strength=mean_abs_cross_partial * feature_a_std * feature_b_std,
        samples=len(values),
    )


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
    return {
        "feature_a": attribution.feature_a,
        "feature_b": attribution.feature_b,
        "weighted_strength": attribution.weighted_strength,
        "mean_abs_cross_partial": attribution.mean_abs_cross_partial,
        "mean_cross_partial": attribution.mean_cross_partial,
        "rms_cross_partial": attribution.rms_cross_partial,
        "feature_a_std": attribution.feature_a_std,
        "feature_b_std": attribution.feature_b_std,
        "samples": attribution.samples,
    }
