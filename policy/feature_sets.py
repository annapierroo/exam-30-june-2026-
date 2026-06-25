"""Named feature extractor selection for train/evaluate scripts."""

from __future__ import annotations

from typing import Literal

from .features import BriscolaFeatureExtractor
from .new_feature_set import NewFeatureSetExtractor


FeatureSetName = Literal["base", "new_interactions"]
FEATURE_SET_NAMES = {"base", "new_interactions"}


def build_feature_extractor(
    feature_set: FeatureSetName,
) -> BriscolaFeatureExtractor | NewFeatureSetExtractor:
    """Build the extractor associated with a named feature set."""

    if feature_set == "base":
        return BriscolaFeatureExtractor()
    if feature_set == "new_interactions":
        return NewFeatureSetExtractor()
    raise ValueError(f"Feature set non supportato: {feature_set}")
