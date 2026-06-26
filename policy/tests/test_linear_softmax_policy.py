from __future__ import annotations

import math
import random
import unittest

import numpy as np

from game.cards import Carta
from game.observation import Osservazione
from policy import BriscolaFeatureExtractor, LinearSoftmaxPolicy
from policy.linear_softmax_policy import add_scaled_in_place


def osservazione(
    mano: tuple[Carta, ...] = (
        Carta("coppe", "asso"),
        Carta("bastoni", "due"),
        Carta("denari", "tre"),
    ),
) -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
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


class TestLinearSoftmaxPolicy(unittest.TestCase):
    def test_initialize_crea_theta_della_dimensione_feature(self):
        # The learner must have one parameter for each feature.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso", "carta_tre"])

        policy = LinearSoftmaxPolicy.initialize(extractor, rng=random.Random(0))

        self.assertEqual(len(policy.theta), extractor.size())
        self.assertEqual(policy.theta.dtype, np.float32)
        self.assertIs(policy.feature_extractor, extractor)

    def test_theta_con_dimensione_errata_solleva_value_error(self):
        # This prevents silent bugs where the dot product and features have different lengths.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])

        with self.assertRaises(ValueError):
            LinearSoftmaxPolicy(theta=[0.0, 1.0], feature_extractor=extractor)

    def test_probabilita_sommano_a_uno_sulle_carte_legali(self):
        # Softmax must return a distribution only over legal actions.
        obs = osservazione()
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(1),
        )

        probabilities = policy.action_probabilities(obs)

        self.assertEqual(set(probabilities), set(obs.azioni_legali))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-6)

    def test_softmax_resta_stabile_con_preferenze_grandi(self):
        # Subtracting the maximum preference prevents numeric overflow.
        obs = osservazione(mano=(Carta("coppe", "asso"), Carta("bastoni", "due")))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[1000.0], feature_extractor=extractor)

        probabilities = policy.action_probabilities(obs)

        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-6)
        self.assertTrue(all(math.isfinite(value) for value in probabilities.values()))

    def test_greedy_sceglie_la_carta_con_probabilita_massima(self):
        # In greedy mode, the policy uses argmax instead of sampling.
        asso = Carta("coppe", "asso")
        due = Carta("bastoni", "due")
        obs = osservazione(mano=(asso, due))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[2.0], feature_extractor=extractor)

        action = policy.select_action(obs, rng=random.Random(0), greedy=True)

        self.assertEqual(action, asso)

    def test_select_action_stocastico_restituisce_carta_legale(self):
        # In stochastic mode, it samples from the distribution over legal actions.
        obs = osservazione()
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(2),
        )

        action = policy.select_action(obs, rng=random.Random(3), greedy=False)

        self.assertIn(action, obs.azioni_legali)

    def test_log_probability_corrisponde_al_log_della_probabilita(self):
        # The log probability is later used by the policy gradient.
        asso = Carta("coppe", "asso")
        due = Carta("bastoni", "due")
        obs = osservazione(mano=(asso, due))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        log_probability = policy.log_probability(obs, asso)

        self.assertAlmostEqual(log_probability, math.log(0.5))

    def test_grad_log_probability_ha_dimensione_theta_e_valore_atteso(self):
        # For linear softmax, the gradient is action features minus expected features.
        asso = Carta("coppe", "asso")
        due = Carta("bastoni", "due")
        obs = osservazione(mano=(asso, due))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        gradient = policy.grad_log_probability(obs, asso)

        self.assertEqual(len(gradient), len(policy.theta))
        self.assertAlmostEqual(gradient[0], 0.5)

    def test_entropy_e_gradiente_entropy_sono_coerenti(self):
        # Entropy regularization uses the closed-form gradient of the softmax entropy.
        asso = Carta("coppe", "asso")
        due = Carta("bastoni", "due")
        obs = osservazione(mano=(asso, due))
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[2.0], feature_extractor=extractor)
        epsilon = 1e-3

        entropy = policy.entropy(obs)
        gradient = policy.grad_entropy(obs)
        entropy_plus = LinearSoftmaxPolicy(
            theta=[2.0 + epsilon],
            feature_extractor=extractor,
        ).entropy(obs)
        entropy_minus = LinearSoftmaxPolicy(
            theta=[2.0 - epsilon],
            feature_extractor=extractor,
        ).entropy(obs)
        finite_difference = (entropy_plus - entropy_minus) / (2 * epsilon)

        self.assertGreater(entropy, 0.0)
        self.assertEqual(gradient.shape, policy.theta.shape)
        self.assertAlmostEqual(gradient[0], finite_difference, places=3)

    def test_action_non_legale_solleva_value_error_per_log_e_gradiente(self):
        # Log probability and gradient are defined only for legal actions.
        obs = osservazione()
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(4),
        )
        illegal_action = Carta("spade", "asso")

        with self.assertRaises(ValueError):
            policy.log_probability(obs, illegal_action)

        with self.assertRaises(ValueError):
            policy.grad_log_probability(obs, illegal_action)

    def test_apply_gradient_aggiorna_theta(self):
        # The update changes the parameters in the gradient direction.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        policy.apply_gradient([2.0], learning_rate=0.1)

        self.assertAlmostEqual(policy.theta[0], 0.2)

    def test_apply_gradient_con_dimensione_errata_solleva_value_error(self):
        # The gradient must have one component for each parameter.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[0.0], feature_extractor=extractor)

        with self.assertRaises(ValueError):
            policy.apply_gradient([1.0, 2.0], learning_rate=0.1)

    def test_copy_duplica_theta_ma_mantiene_feature_extractor(self):
        # Snapshots must have independent parameters but the same feature encoding.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = LinearSoftmaxPolicy(theta=[1.0], feature_extractor=extractor)

        copied = policy.copy(name="snapshot")
        copied.theta[0] = 2.0

        self.assertEqual(copied.name, "snapshot")
        self.assertTrue(np.allclose(policy.theta, np.asarray([1.0], dtype=np.float32)))
        self.assertTrue(np.allclose(copied.theta, np.asarray([2.0], dtype=np.float32)))
        self.assertEqual(copied.theta.dtype, np.float32)
        self.assertIs(copied.feature_extractor, extractor)

    def test_add_scaled_in_place_aggiorna_vettore_target(self):
        # This helper accumulates gradients without creating new vectors.
        target = np.asarray([1.0, 2.0], dtype=np.float32)

        add_scaled_in_place(target, [3.0, 4.0], scale=0.5)

        self.assertTrue(np.allclose(target, np.asarray([2.5, 4.0], dtype=np.float32)))

    def test_mano_vuota_solleva_value_error(self):
        # A softmax distribution requires at least one legal action.
        obs = osservazione(mano=())
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(5),
        )

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
