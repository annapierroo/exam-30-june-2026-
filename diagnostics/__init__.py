"""Diagnostics utilities for inspecting Briscola policy decisions."""

from .decision_log import (
    DecisionLog,
    DecisionOutcome,
    DecisionRecord,
    decision_log_to_dict,
    record_decision_log,
)
from .neural_interactions import (
    DEFAULT_INTERACTION_PAIRS,
    FeatureAttribution,
    InteractionAttribution,
    NeuralActionSample,
    NeuralInteractionAnalysis,
    analyze_neural_action_samples,
    collect_neural_action_samples,
    default_interaction_pairs,
    neural_interaction_report_to_dict,
)
from .views import (
    records_by_trick_position,
    records_chosen_with_probability_below,
    records_for_player,
    records_for_policy,
    records_on_rich_trick,
    records_with_opponent_leading,
    records_with_partner_leading,
)

__all__ = [
    "DecisionLog",
    "DecisionOutcome",
    "DecisionRecord",
    "DEFAULT_INTERACTION_PAIRS",
    "FeatureAttribution",
    "InteractionAttribution",
    "NeuralActionSample",
    "NeuralInteractionAnalysis",
    "analyze_neural_action_samples",
    "collect_neural_action_samples",
    "default_interaction_pairs",
    "decision_log_to_dict",
    "neural_interaction_report_to_dict",
    "record_decision_log",
    "records_by_trick_position",
    "records_chosen_with_probability_below",
    "records_for_player",
    "records_for_policy",
    "records_on_rich_trick",
    "records_with_opponent_leading",
    "records_with_partner_leading",
]
