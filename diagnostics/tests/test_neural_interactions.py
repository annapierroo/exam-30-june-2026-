from __future__ import annotations

import random
import unittest

from game.cards import Carta
from game.observation import Osservazione
from policy import BriscolaFeatureExtractor, NeuralSoftmaxPolicy
from diagnostics.neural_interactions import (
    NeuralActionSample,
    analyze_neural_action_samples,
    build_interaction_candidates,
    collect_neural_action_samples,
    default_interaction_pairs,
    default_interactions,
    neural_interaction_report_to_dict,
)
from evaluation import EvaluationSuite, make_evaluation_cases
from policy import RandomPolicy
from evaluation.suite import EvaluationScenario


def osservazione_con_mano() -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=(Carta("coppe", "asso"), Carta("bastoni", "due")),
        mano_compagno_visibile=False,
        mano_compagno=(),
        seme_briscola="denari",
        briscola_esposta=Carta("denari", "asso"),
        proprietario_briscola_esposta=None,
        carte_sul_campo=(),
        carte_giocate=(),
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=0,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=0,
        posizione_nella_presa=0,
    )


class TestNeuralInteractions(unittest.TestCase):
    def test_analyze_samples_restituisce_feature_e_interazioni(self):
        # The diagnostic ranks local logit sensitivities without running training.
        extractor = BriscolaFeatureExtractor()
        policy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(0),
            hidden_size=4,
        )
        obs = osservazione_con_mano()
        features = tuple(extractor.extract(obs, obs.azioni_legali[0]))
        samples = [
            NeuralActionSample(
                scenario_name="unit",
                game_index=0,
                step_index=0,
                action_id=obs.azioni_legali[0].id,
                chosen=True,
                probability=0.5,
                logit=0.0,
                features=features,
            )
        ]

        analysis = analyze_neural_action_samples(
            policy=policy,
            samples=samples,
            interaction_pairs=(("carta_briscola", "punti_presa"),),
        )

        self.assertEqual(analysis.sample_count, 1)
        self.assertEqual(len(analysis.feature_attributions), extractor.size())
        self.assertEqual(len(analysis.interaction_attributions), 1)
        self.assertEqual(
            analysis.interaction_attributions[0].features,
            ("carta_briscola", "punti_presa"),
        )

    def test_feature_pair_non_valida_solleva_value_error(self):
        # Unknown features fail fast instead of producing a misleading report.
        extractor = BriscolaFeatureExtractor()
        policy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(1),
            hidden_size=4,
        )
        obs = osservazione_con_mano()
        samples = [
            NeuralActionSample(
                scenario_name="unit",
                game_index=0,
                step_index=0,
                action_id=obs.azioni_legali[0].id,
                chosen=True,
                probability=0.5,
                logit=0.0,
                features=tuple(extractor.extract(obs, obs.azioni_legali[0])),
            )
        ]

        with self.assertRaises(ValueError):
            analyze_neural_action_samples(
                policy=policy,
                samples=samples,
                interaction_pairs=(("missing", "punti_presa"),),
            )

    def test_collect_neural_action_samples_da_suite_minima(self):
        # Collection reuses legal decision logs and records learner legal actions.
        extractor = BriscolaFeatureExtractor()
        policy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(2),
            hidden_size=4,
        )
        suite = EvaluationSuite(
            scenarios=(
                EvaluationScenario(
                    name="random_unit",
                    compagno_policy=RandomPolicy(),
                    avversario_successivo_policy=RandomPolicy(),
                    avversario_precedente_policy=RandomPolicy(),
                ),
            )
        )

        samples = collect_neural_action_samples(
            learner_policy=policy,
            suite=suite,
            cases=make_evaluation_cases(
                games=1,
                seed_ambiente_start=10,
                seed_policy_start=20,
            ),
            max_decisions=2,
        )

        self.assertGreaterEqual(len(samples), 2)
        self.assertTrue(all(sample.scenario_name == "random_unit" for sample in samples))

    def test_default_pairs_filtra_feature_disponibili(self):
        pairs = default_interaction_pairs(("carta_briscola", "punti_presa"))

        self.assertEqual(pairs, (("carta_briscola", "punti_presa"),))

    def test_default_interactions_include_coppie_e_triple(self):
        interactions = default_interactions(
            (
                "carta_briscola",
                "punti_presa",
                "carta_prende",
                "posizione_quarto",
            ),
            max_order=3,
        )

        self.assertIn(("carta_briscola", "punti_presa"), interactions)
        self.assertIn(("carta_briscola", "punti_presa", "carta_prende"), interactions)

    def test_build_interaction_candidates_supporta_ordini_superiori_a_quattro(self):
        interactions = build_interaction_candidates(
            features=(
                "carta_briscola",
                "punti_presa",
                "carta_prende",
                "posizione_quarto",
                "fase_finale",
            ),
            max_order=5,
        )

        self.assertIn(
            (
                "carta_briscola",
                "punti_presa",
                "carta_prende",
                "posizione_quarto",
                "fase_finale",
            ),
            interactions,
        )

    def test_build_interaction_candidates_rifiuta_ordini_sopra_le_feature(self):
        with self.assertRaises(ValueError):
            build_interaction_candidates(
                features=("carta_briscola", "punti_presa"),
                max_order=5,
            )

    def test_build_interaction_candidates_rispetta_limite_candidate(self):
        with self.assertRaises(ValueError):
            build_interaction_candidates(
                features=(
                    "carta_briscola",
                    "punti_presa",
                    "carta_prende",
                    "posizione_quarto",
                    "fase_finale",
                ),
                max_order=3,
                max_candidates=10,
            )

    def test_report_include_top_interactions_per_ordine(self):
        extractor = BriscolaFeatureExtractor()
        policy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(3),
            hidden_size=4,
        )
        obs = osservazione_con_mano()
        samples = [
            NeuralActionSample(
                scenario_name="unit",
                game_index=0,
                step_index=0,
                action_id=obs.azioni_legali[0].id,
                chosen=True,
                probability=0.5,
                logit=0.0,
                features=tuple(extractor.extract(obs, obs.azioni_legali[0])),
            )
        ]
        analysis = analyze_neural_action_samples(
            policy=policy,
            samples=samples,
            interaction_candidates=(
                ("carta_briscola", "punti_presa"),
                ("carta_briscola", "punti_presa", "carta_prende"),
            ),
        )

        report = neural_interaction_report_to_dict(
            checkpoint_path="checkpoint.json",
            checkpoint_update_index=1,
            games_per_scenario=1,
            greedy=True,
            chosen_only=False,
            samples=samples,
            analysis=analysis,
            top_n=5,
        )

        self.assertEqual(report["interaction_config"]["candidate_count"], 2)
        self.assertEqual(
            report["interaction_config"]["candidate_order_counts"],
            {"2": 1, "3": 1},
        )
        self.assertIn("2", report["top_interactions_by_order"])
        self.assertIn("3", report["top_interactions_by_order"])


if __name__ == "__main__":
    unittest.main()
