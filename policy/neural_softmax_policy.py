"""PyTorch neural softmax policy for learnable Briscola agents."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from game.cards import Carta
from game.observation import Osservazione

from .base import FeatureExtractor
from .features import BriscolaFeatureExtractor
from .linear_softmax_policy import vector_norm


class NeuralSoftmaxPolicy(nn.Module):
    """One-hidden-layer PyTorch policy with a masked softmax over legal cards."""

    def __init__(
        self,
        theta: Sequence[float] | np.ndarray | None = None,
        feature_extractor: FeatureExtractor | None = None,
        hidden_size: int = 64,
        name: str = "neural_softmax",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size deve essere positivo")

        self.feature_extractor = feature_extractor or BriscolaFeatureExtractor()
        self.hidden_size = int(hidden_size)
        self.name = name
        input_size = self.feature_extractor.size()

        self.hidden_layer = nn.Linear(input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self._zero_parameters()
        if theta is not None:
            self.load_flat_parameters(theta)

    @classmethod
    def initialize(
        cls,
        feature_extractor: FeatureExtractor | None = None,
        rng: random.Random | None = None,
        hidden_size: int = 64,
        scale: float = 0.01,
        name: str = "neural_softmax",
    ) -> NeuralSoftmaxPolicy:
        """Initialize small random MLP parameters matching the feature dimension."""

        feature_extractor = feature_extractor or BriscolaFeatureExtractor()
        rng = rng or random.Random()
        policy = cls(
            feature_extractor=feature_extractor,
            hidden_size=hidden_size,
            name=name,
        )
        values = np.asarray(
            [
                rng.uniform(-scale, scale)
                for _ in range(
                    policy.parameter_count(feature_extractor.size(), hidden_size)
                )
            ],
            dtype=np.float32,
        )
        policy.load_flat_parameters(values)
        return policy

    @staticmethod
    def parameter_count(input_size: int, hidden_size: int) -> int:
        """Return the number of flat parameters for the configured MLP."""

        if input_size <= 0:
            raise ValueError("input_size deve essere positivo")
        if hidden_size <= 0:
            raise ValueError("hidden_size deve essere positivo")
        return hidden_size * input_size + hidden_size + hidden_size + 1

    @property
    def theta(self) -> np.ndarray:
        """Return a flat NumPy copy for checkpoint and snapshot compatibility."""

        with torch.no_grad():
            chunks = [
                parameter.detach().cpu().numpy().reshape(-1)
                for parameter in self._ordered_parameters()
            ]
        return np.concatenate(chunks).astype(np.float32)

    @theta.setter
    def theta(self, values: Sequence[float] | np.ndarray) -> None:
        self.load_flat_parameters(values)

    def copy(self, name: str | None = None) -> NeuralSoftmaxPolicy:
        """Copy parameters while sharing the immutable feature definition."""

        return NeuralSoftmaxPolicy(
            theta=self.theta,
            feature_extractor=self.feature_extractor,
            hidden_size=self.hidden_size,
            name=name or self.name,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits from a batch of feature vectors."""

        hidden = torch.tanh(self.hidden_layer(features))
        return self.output_layer(hidden).squeeze(-1)

    def action_preferences(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Compute neural preferences for legal cards only."""

        with torch.no_grad():
            cards, logits = self.action_logits_tensor(osservazione)
            values = logits.detach().cpu().numpy()
        return {carta: float(value) for carta, value in zip(cards, values)}

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Apply a numerically stable softmax to legal action preferences."""

        with torch.no_grad():
            cards, logits = self.action_logits_tensor(osservazione)
            probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
        return {
            carta: float(probability)
            for carta, probability in zip(cards, probabilities)
        }

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

        return float(self.log_probability_tensor(osservazione, action).detach().item())

    def log_probability_tensor(
        self,
        osservazione: Osservazione,
        action: Carta,
    ) -> torch.Tensor:
        """Return a differentiable log pi(action | observation) tensor."""

        cards, logits = self.action_logits_tensor(osservazione)
        if action not in cards:
            raise ValueError("Action is not legal")
        action_index = cards.index(action)
        return F.log_softmax(logits, dim=0)[action_index]

    def action_logits_tensor(
        self,
        osservazione: Osservazione,
    ) -> tuple[list[Carta], torch.Tensor]:
        """Return legal cards and differentiable logits in matching order."""

        cards = list(osservazione.azioni_legali)
        if not cards:
            raise ValueError("No legal actions available")
        feature_batch = torch.stack(
            [self._feature_tensor(osservazione, carta) for carta in cards],
        )
        return cards, self.forward(feature_batch)

    def load_flat_parameters(self, values: Sequence[float] | np.ndarray) -> None:
        """Load parameters from the flat checkpoint/snapshot representation."""

        values_array = np.asarray(values, dtype=np.float32)
        expected_shape = (
            self.parameter_count(self.feature_extractor.size(), self.hidden_size),
        )
        if values_array.shape != expected_shape:
            raise ValueError(
                f"Theta shape {values_array.shape} does not match "
                f"neural parameter shape {expected_shape}"
            )

        offset = 0
        with torch.no_grad():
            for parameter in self._ordered_parameters():
                size = parameter.numel()
                chunk = values_array[offset : offset + size].reshape(parameter.shape)
                parameter.copy_(torch.from_numpy(chunk))
                offset += size

    def _feature_tensor(self, osservazione: Osservazione, carta: Carta) -> torch.Tensor:
        features = np.asarray(
            self.feature_extractor.extract(osservazione, carta),
            dtype=np.float32,
        )
        expected_shape = (self.feature_extractor.size(),)
        if features.shape != expected_shape:
            raise ValueError("Feature vector size must match extractor size")
        return torch.from_numpy(features)

    def _ordered_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (
            self.hidden_layer.weight,
            self.hidden_layer.bias,
            self.output_layer.weight,
            self.output_layer.bias,
        )

    def _zero_parameters(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


def neural_vector_norm(policy: NeuralSoftmaxPolicy) -> float:
    """Return the norm of a neural policy parameter vector."""

    return vector_norm(policy.theta)
