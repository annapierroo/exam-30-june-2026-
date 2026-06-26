"""Shared policy interface."""

from __future__ import annotations

import random
from typing import Protocol

from game.cards import Carta
from game.observation import Osservazione


class Policy(Protocol):
    """Minimal interface for policies that act from legal observations."""

    name: str

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Return a probability for each legal action."""
        ...

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        """Select one legal action from the observation."""
        ...


class FeatureExtractor(Protocol):
    """Minimal interface shared by selectable policy feature schemas."""

    feature_names: list[str]

    def size(self) -> int:
        """Return the fixed vector dimension."""
        ...

    def extract(self, osservazione: Osservazione, carta: Carta) -> list[float]:
        """Extract one legal-action feature vector."""
        ...
