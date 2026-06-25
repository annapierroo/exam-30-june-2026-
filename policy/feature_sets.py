"""Named feature extractor selection for train/evaluate scripts."""

from __future__ import annotations

from typing import Literal

from .features import (
    DEFAULT_ATOMIC_FEATURE_NAMES as BASE_ATOMIC_FEATURE_NAMES,
    DEFAULT_INTERACTION_FEATURE_NAMES as BASE_INTERACTION_FEATURE_NAMES,
    BriscolaFeatureExtractor,
)
from .new_feature_set import (
    DEFAULT_ATOMIC_FEATURE_NAMES as NEW_ATOMIC_FEATURE_NAMES,
    DEFAULT_INTERACTION_FEATURE_NAMES as NEW_INTERACTION_FEATURE_NAMES,
    NewFeatureSetExtractor,
)


FeatureSetName = Literal[
    "base",
    "base_atomic",
    "base_interactions_only",
    "new_interactions",
    "new_atomic",
    "new_interactions_only",
]
FEATURE_SET_NAMES = {
    "base",
    "base_atomic",
    "base_interactions_only",
    "new_interactions",
    "new_atomic",
    "new_interactions_only",
}


def build_feature_extractor(
    feature_set: FeatureSetName,
) -> BriscolaFeatureExtractor | NewFeatureSetExtractor:
    """Build the extractor associated with a named feature set."""

    if feature_set == "base":
        return BriscolaFeatureExtractor()
    if feature_set == "base_atomic":
        return BriscolaFeatureExtractor(feature_names=list(BASE_ATOMIC_FEATURE_NAMES))
    if feature_set == "base_interactions_only":
        return BriscolaFeatureExtractor(
            feature_names=list(BASE_INTERACTION_FEATURE_NAMES),
        )
    if feature_set == "new_interactions":
        return NewFeatureSetExtractor()
    if feature_set == "new_atomic":
        return NewFeatureSetExtractor(feature_names=list(NEW_ATOMIC_FEATURE_NAMES))
    if feature_set == "new_interactions_only":
        return NewFeatureSetExtractor(
            feature_names=list(NEW_INTERACTION_FEATURE_NAMES),
        )
    raise ValueError(f"Feature set non supportato: {feature_set}")
