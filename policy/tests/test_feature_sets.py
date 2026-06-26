from __future__ import annotations

import unittest

from policy.feature_sets import FEATURE_SET_NAMES, build_feature_extractor


class TestFeatureSets(unittest.TestCase):
    def test_base_legacy_name_keeps_full_feature_vector(self):
        # Il nome storico base resta il vettore completo.
        base = build_feature_extractor("base")

        self.assertIn("base", FEATURE_SET_NAMES)
        self.assertEqual(base.size(), 68)
        self.assertEqual(len(base.atomic_feature_names), 56)
        self.assertEqual(len(base.interaction_feature_names), 12)

    def test_new_legacy_name_keeps_full_feature_vector(self):
        # Il nome storico new_interactions resta il vettore completo.
        new_interactions = build_feature_extractor("new_interactions")

        self.assertIn("new_interactions", FEATURE_SET_NAMES)
        self.assertEqual(new_interactions.size(), 88)
        self.assertEqual(len(new_interactions.atomic_feature_names), 20)
        self.assertEqual(len(new_interactions.interaction_feature_names), 68)

    def test_atomic_feature_sets_exclude_engineered_interactions(self):
        # Le varianti atomic tengono solo le feature non-interazione.
        base_atomic = build_feature_extractor("base_atomic")
        new_atomic = build_feature_extractor("new_atomic")

        self.assertEqual(base_atomic.size(), 56)
        self.assertEqual(new_atomic.size(), 20)
        self.assertEqual(
            tuple(base_atomic.feature_names),
            base_atomic.atomic_feature_names,
        )
        self.assertEqual(
            tuple(new_atomic.feature_names),
            new_atomic.atomic_feature_names,
        )
        self.assertFalse(base_atomic.interaction_feature_names)
        self.assertFalse(new_atomic.interaction_feature_names)

    def test_interaction_only_feature_sets_exclude_atomic_features(self):
        # Le varianti interactions_only tengono solo prodotti engineered.
        base_interactions = build_feature_extractor("base_interactions_only")
        new_interactions = build_feature_extractor("new_interactions_only")

        self.assertEqual(base_interactions.size(), 12)
        self.assertEqual(new_interactions.size(), 68)
        self.assertFalse(base_interactions.atomic_feature_names)
        self.assertFalse(new_interactions.atomic_feature_names)
        self.assertEqual(
            tuple(base_interactions.feature_names),
            base_interactions.interaction_feature_names,
        )
        self.assertEqual(
            tuple(new_interactions.feature_names),
            new_interactions.interaction_feature_names,
        )

    def test_aligned_feature_sets_condividono_la_stessa_base_atomica(self):
        # Le varianti allineate cambiano solo le interazioni aggiunte alla base comune.
        common_atomic = build_feature_extractor("common_atomic")
        base_aligned = build_feature_extractor("base_aligned")
        new_aligned = build_feature_extractor("new_aligned")

        self.assertEqual(common_atomic.size(), 73)
        self.assertEqual(base_aligned.size(), 85)
        self.assertEqual(new_aligned.size(), 141)
        self.assertEqual(
            tuple(common_atomic.feature_names),
            common_atomic.atomic_feature_names,
        )
        self.assertEqual(base_aligned.atomic_feature_names, common_atomic.atomic_feature_names)
        self.assertEqual(new_aligned.atomic_feature_names, common_atomic.atomic_feature_names)
        self.assertEqual(len(base_aligned.interaction_feature_names), 12)
        self.assertEqual(len(new_aligned.interaction_feature_names), 68)

    def test_feature_set_names_are_only_supported_public_names(self):
        # Evita alias duplicati che renderebbero confusa la griglia sperimentale.
        self.assertEqual(
            FEATURE_SET_NAMES,
            {
                "base",
                "base_atomic",
                "base_interactions_only",
                "new_interactions",
                "new_atomic",
                "new_interactions_only",
                "common_atomic",
                "base_aligned",
                "new_aligned",
            },
        )

    def test_unknown_feature_set_fails_fast(self):
        # Un nome non supportato deve fallire prima di iniziare training/evaluation.
        with self.assertRaises(ValueError):
            build_feature_extractor("non_tra_le_opzioni")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
