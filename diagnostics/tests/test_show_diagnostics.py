from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from scripts.show_diagnostics import (
    is_decision_diagnostics_report,
    is_neural_interaction_report,
    print_neural_interaction_report,
)


class TestShowDiagnostics(unittest.TestCase):
    def test_riconosce_report_decision_log(self):
        # Existing decision-log reports keep the original display path.
        report = {"decision_log": {"records": []}}

        self.assertTrue(is_decision_diagnostics_report(report))
        self.assertFalse(is_neural_interaction_report(report))

    def test_stampa_report_neural_interactions(self):
        # Neural interaction reports are shown as aggregate tables.
        report = {
            "checkpoint_path": "models/run/checkpoint.json",
            "checkpoint_update_index": 10,
            "games_per_scenario": 100,
            "greedy": True,
            "chosen_only": False,
            "sample_count": 123,
            "samples_by_scenario": {"random_eval": 123},
            "interaction_config": {
                "max_evaluated_order": 3,
                "max_nonzero_order": 3,
                "candidate_count": 2,
                "evaluated_interaction_count": 2,
                "candidate_order_counts": {"2": 1, "3": 1},
            },
            "interaction_order_summary": {
                "3": {
                    "candidate_count": 1,
                    "nonzero_mean_abs_partial_count": 1,
                    "max_weighted_strength": 0.01,
                    "max_mean_abs_partial": 0.02,
                }
            },
            "metric_notes": {
                "logit_scope": "scores explain local neural logits",
            },
            "top_features": [
                {
                    "feature": "carta_briscola",
                    "weighted_strength": 0.12,
                    "mean_abs_gradient": 0.34,
                    "feature_std": 0.56,
                }
            ],
            "top_interactions": [
                {
                    "features": [
                        "carta_briscola",
                        "punti_presa",
                        "carta_prende",
                    ],
                    "weighted_strength": 0.01,
                    "mean_abs_partial": 0.02,
                }
            ],
            "top_interactions_by_order": {
                "3": [
                    {
                        "features": [
                            "carta_briscola",
                            "punti_presa",
                            "carta_prende",
                        ],
                        "weighted_strength": 0.01,
                        "mean_abs_partial": 0.02,
                    }
                ]
            },
        }
        output = io.StringIO()

        with redirect_stdout(output):
            print_neural_interaction_report(report, limit=None)

        printed = output.getvalue()
        self.assertIn("Neural interaction diagnostics", printed)
        self.assertIn("Top feature sensitivities", printed)
        self.assertIn("carta_briscola", printed)
        self.assertIn("Top interactions", printed)
        self.assertIn("Top interactions by order", printed)
        self.assertIn("Order 3", printed)
        self.assertIn("punti_presa", printed)
        self.assertIn("carta_prende", printed)


if __name__ == "__main__":
    unittest.main()
