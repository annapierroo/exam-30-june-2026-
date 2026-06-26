"""Shared non-interaction feature schema for aligned policy experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from game.cards import Carta
from game.observation import Osservazione

from .features import (
    DEFAULT_ATOMIC_FEATURE_NAMES as BASE_ATOMIC_FEATURE_NAMES,
    DEFAULT_INTERACTION_FEATURE_NAMES as BASE_INTERACTION_FEATURE_NAMES,
    BriscolaFeatureExtractor,
)
from .new_feature_set import (
    DEFAULT_ATOMIC_FEATURE_NAMES as NEW_ATOMIC_FEATURE_NAMES,
    DEFAULT_INTERACTION_FEATURE_NAMES as NEW_INTERACTION_FEATURE_NAMES,
    NEW_CONTEXT_FACTOR_FEATURE_NAMES,
    NewFeatureSetExtractor,
)


def _ordered_unique(names: tuple[str, ...]) -> tuple[str, ...]:
    """Keep the first occurrence of each feature name."""

    return tuple(dict.fromkeys(names))


COMMON_ATOMIC_FEATURE_NAMES = _ordered_unique(
    BASE_ATOMIC_FEATURE_NAMES
    + NEW_ATOMIC_FEATURE_NAMES
    + NEW_CONTEXT_FACTOR_FEATURE_NAMES
)

BASE_ALIGNED_FEATURE_NAMES = (
    COMMON_ATOMIC_FEATURE_NAMES + BASE_INTERACTION_FEATURE_NAMES
)
NEW_ALIGNED_FEATURE_NAMES = (
    COMMON_ATOMIC_FEATURE_NAMES + NEW_INTERACTION_FEATURE_NAMES
)

_BASE_FEATURE_NAMES = set(
    BASE_ATOMIC_FEATURE_NAMES + BASE_INTERACTION_FEATURE_NAMES,
)
_NEW_FEATURE_NAMES = set(
    NEW_ATOMIC_FEATURE_NAMES
    + NEW_CONTEXT_FACTOR_FEATURE_NAMES
    + NEW_INTERACTION_FEATURE_NAMES,
)
_KNOWN_FEATURE_NAMES = _BASE_FEATURE_NAMES | _NEW_FEATURE_NAMES
_INTERACTION_FEATURE_NAMES = set(
    BASE_INTERACTION_FEATURE_NAMES + NEW_INTERACTION_FEATURE_NAMES,
)


@dataclass
class AlignedFeatureExtractor:
    """Compose aligned vectors from the two established feature extractors."""

    feature_names: list[str] = field(
        default_factory=lambda: list(BASE_ALIGNED_FEATURE_NAMES),
    )
    _base_feature_names: list[str] = field(init=False, repr=False)
    _new_feature_names: list[str] = field(init=False, repr=False)
    _base_extractor: BriscolaFeatureExtractor = field(init=False, repr=False)
    _new_extractor: NewFeatureSetExtractor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("Le feature allineate non possono contenere duplicati")

        unknown_names = set(self.feature_names) - _KNOWN_FEATURE_NAMES
        if unknown_names:
            raise ValueError(
                f"Feature allineate sconosciute: {sorted(unknown_names)}",
            )

        self._base_feature_names = [
            name for name in self.feature_names if name in _BASE_FEATURE_NAMES
        ]
        self._new_feature_names = [
            name
            for name in self.feature_names
            if name not in _BASE_FEATURE_NAMES and name in _NEW_FEATURE_NAMES
        ]
        self._base_extractor = BriscolaFeatureExtractor(
            feature_names=self._base_feature_names,
        )
        self._new_extractor = NewFeatureSetExtractor(
            feature_names=self._new_feature_names,
        )

    def size(self) -> int:
        """Return the feature vector size."""

        return len(self.feature_names)

    @property
    def atomic_feature_names(self) -> tuple[str, ...]:
        """Return the active non-interaction feature names."""

        atomic_names = set(COMMON_ATOMIC_FEATURE_NAMES)
        return tuple(name for name in self.feature_names if name in atomic_names)

    @property
    def interaction_feature_names(self) -> tuple[str, ...]:
        """Return the active engineered interaction feature names."""

        return tuple(
            name for name in self.feature_names if name in _INTERACTION_FEATURE_NAMES
        )

    def extract(self, osservazione: Osservazione, carta: Carta) -> list[float]:
        """Extract one vector without reimplementing either legacy formula set."""

        values = dict(
            zip(
                self._base_feature_names,
                self._base_extractor.extract(osservazione, carta),
                strict=True,
            ),
        )
        values.update(
            zip(
                self._new_feature_names,
                self._new_extractor.extract(osservazione, carta),
                strict=True,
            ),
        )
        return [float(values[name]) for name in self.feature_names]
