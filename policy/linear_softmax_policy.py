"""Linear softmax policy for learnable Briscola agents."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from game.cards import Carta
from game.observation import Osservazione

from .base import FeatureExtractor
from .features import BriscolaFeatureExtractor


@dataclass
class LinearSoftmaxPolicy:
    """Linear action preferences followed by a masked softmax over legal cards."""

    theta: Sequence[float] | np.ndarray
    feature_extractor: FeatureExtractor = field(
        default_factory=BriscolaFeatureExtractor
    )
    name: str = "linear_softmax"

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=np.float32).copy()
        self._validate_theta_size()

    @classmethod
    def initialize(
        cls,
        feature_extractor: FeatureExtractor | None = None,
        rng: random.Random | None = None,
        scale: float = 0.01,
        name: str = "linear_softmax",
    ) -> LinearSoftmaxPolicy:
        """Initialize small random parameters matching the feature dimension."""

        feature_extractor = feature_extractor or BriscolaFeatureExtractor()
        rng = rng or random.Random()
        theta = np.asarray(
            [rng.uniform(-scale, scale) for _ in range(feature_extractor.size())],
            dtype=np.float32,
        )
        return cls(theta=theta, feature_extractor=feature_extractor, name=name)

    def copy(self, name: str | None = None) -> LinearSoftmaxPolicy:
        """Copy parameters while sharing the immutable feature definition."""

        return LinearSoftmaxPolicy(
            theta=np.array(self.theta, dtype=np.float32, copy=True),
            feature_extractor=self.feature_extractor,
            name=name or self.name,
        )

    def action_preferences(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Compute linear preferences for legal cards only."""

        features_by_card = self._features_by_card(osservazione)
        return self._preferences_from_features(features_by_card)

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Apply a numerically stable softmax to legal action preferences."""

        preferences = self.action_preferences(osservazione)
        return self._probabilities_from_preferences(preferences)

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        """Select argmax when greedy, otherwise sample from the softmax."""

        probabilities = self.action_probabilities(osservazione)
        if greedy:
            return max(probabilities, key=probabilities.get)

        threshold = rng.random()
        cumulative = 0.0
        last_card = next(iter(probabilities))
        for carta, probability in probabilities.items():
            cumulative += probability
            last_card = carta
            if threshold <= cumulative:
                return carta
        return last_card

    def log_probability(self, osservazione: Osservazione, action: Carta) -> float:
        """Return log pi(action | observation) for a legal action."""

        probabilities = self.action_probabilities(osservazione)
        if action not in probabilities:
            raise ValueError("Action is not legal")
        return math.log(max(probabilities[action], 1e-15))

    def grad_log_probability(
        self,
        osservazione: Osservazione,
        action: Carta,
    ) -> np.ndarray:
        """Return grad log pi(action | observation) for linear softmax."""

        features_by_card = self._features_by_card(osservazione)
        preferences = self._preferences_from_features(features_by_card)
        probabilities = self._probabilities_from_preferences(preferences)
        if action not in probabilities:
            raise ValueError("Action is not legal")

        action_features = features_by_card[action]
        expected_features = np.zeros_like(self.theta, dtype=np.float32)

        for carta, probability in probabilities.items():
            expected_features += np.float32(probability) * features_by_card[carta]

        return (action_features - expected_features).astype(np.float32)

    def entropy(self, osservazione: Osservazione) -> float:
        """Return the entropy of the legal-action distribution."""

        probabilities = self.action_probabilities(osservazione)
        return entropy_from_probabilities(probabilities.values())

    def grad_entropy(self, osservazione: Osservazione) -> np.ndarray:
        """Return grad H(pi(. | observation)) for linear softmax."""

        features_by_card = self._features_by_card(osservazione)
        preferences = self._preferences_from_features(features_by_card)
        probabilities = self._probabilities_from_preferences(preferences)
        entropy = entropy_from_probabilities(probabilities.values())
        gradient = np.zeros_like(self.theta, dtype=np.float32)

        for carta, probability in probabilities.items():
            log_probability = math.log(max(probability, 1e-15))
            scale = np.float32(probability * (-log_probability - entropy))
            gradient += scale * features_by_card[carta]

        return gradient.astype(np.float32)

    def apply_gradient(
        self,
        gradient: Sequence[float] | np.ndarray,
        learning_rate: float,
    ) -> None:
        """Apply a gradient step."""

        gradient_array = np.asarray(gradient, dtype=np.float32)
        if gradient_array.shape != self.theta.shape:
            raise ValueError("Gradient size must match theta size")
        self.theta += np.float32(learning_rate) * gradient_array

    def _features(self, osservazione: Osservazione, carta: Carta) -> np.ndarray:
        features = np.asarray(
            self.feature_extractor.extract(osservazione, carta),
            dtype=np.float32,
        )
        if features.shape != self.theta.shape:
            raise ValueError("Feature vector size must match theta size")
        return features

    def _features_by_card(self, osservazione: Osservazione) -> dict[Carta, np.ndarray]:
        return {
            carta: self._features(osservazione, carta)
            for carta in osservazione.azioni_legali
        }

    def _preferences_from_features(
        self,
        features_by_card: dict[Carta, np.ndarray],
    ) -> dict[Carta, float]:
        return {
            carta: dot(self.theta, features)
            for carta, features in features_by_card.items()
        }

    def _probabilities_from_preferences(
        self,
        preferences: dict[Carta, float],
    ) -> dict[Carta, float]:
        if not preferences:
            raise ValueError("No legal actions available")

        cards = list(preferences)
        preference_values = np.asarray(
            [preferences[carta] for carta in cards],
            dtype=np.float32,
        )
        exp_values = np.exp(preference_values - np.max(preference_values))
        probabilities = exp_values / np.sum(exp_values)
        return {
            carta: float(probability)
            for carta, probability in zip(cards, probabilities)
        }

    def _validate_theta_size(self) -> None:
        expected_shape = (self.feature_extractor.size(),)
        if self.theta.shape != expected_shape:
            raise ValueError(
                f"Theta shape {self.theta.shape} does not match "
                f"feature shape {expected_shape}"
            )


def dot(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        raise ValueError("Vectors must have the same size")
    return float(np.dot(left_array, right_array))


def add_scaled_in_place(
    target: np.ndarray,
    source: Sequence[float] | np.ndarray,
    scale: float,
) -> None:
    source_array = np.asarray(source, dtype=np.float32)
    if target.shape != source_array.shape:
        raise ValueError("Vectors must have the same size")
    target += np.float32(scale) * source_array


def vector_norm(vector: Sequence[float] | np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))


def entropy_from_probabilities(probabilities: Iterable[float]) -> float:
    return float(
        -sum(
            probability * math.log(max(probability, 1e-15))
            for probability in probabilities
        )
    )
