from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from game.cards import Carta
from game.observation import Osservazione
from training.episode import EpisodeResult, TrajectoryStep
from training.reinforce import ReinforceConfig, reinforce_update


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


@dataclass
class FakeLinearPolicy:
    theta: np.ndarray
    gradient_value: float = 1.0
    entropy_value: float = 0.7
    entropy_gradient_value: float = 2.0
    applied_gradient: np.ndarray | None = None
    applied_learning_rate: float | None = None

    def grad_log_probability(
        self,
        osservazione: Osservazione,
        action: Carta,
    ) -> np.ndarray:
        return np.asarray([self.gradient_value], dtype=np.float32)

    def entropy(self, osservazione: Osservazione) -> float:
        return self.entropy_value

    def grad_entropy(self, osservazione: Osservazione) -> np.ndarray:
        return np.asarray([self.entropy_gradient_value], dtype=np.float32)

    def apply_gradient(
        self,
        gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        self.applied_gradient = np.asarray(gradient, dtype=np.float32).copy()
        self.applied_learning_rate = learning_rate
        self.theta += np.float32(learning_rate) * self.applied_gradient


def episodio(
    reward_to_go_values: list[float],
    *,
    punteggio_squadra: int = 70,
    punteggio_avversari: int = 50,
    episode_return: float | None = None,
) -> EpisodeResult:
    obs = osservazione_con_mano()
    action = obs.azioni_legali[0]
    steps = [
        TrajectoryStep(
            osservazione=obs,
            azione=action,
            global_step_index=index,
            reward_to_go=value,
        )
        for index, value in enumerate(reward_to_go_values)
    ]
    if episode_return is None:
        episode_return = sum(reward_to_go_values)
    return EpisodeResult(
        steps=steps,
        rewards=[],
        punteggi_finali={
            "pari": punteggio_squadra,
            "dispari": punteggio_avversari,
        },
        learner_giocatore_id=0,
        learner_squadra="pari",
        episode_return=float(episode_return),
    )


class TestReinforceUpdate(unittest.TestCase):
    def test_config_default_e_valori_validi(self):
        # Defaults declare the chosen baseline and disabled optional regularizers.
        config = ReinforceConfig()

        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.baseline, "time_dependent")
        self.assertEqual(config.entropy_coef, 0.0)
        self.assertEqual(ReinforceConfig(baseline="none").baseline, "none")
        self.assertEqual(ReinforceConfig(baseline="batch_mean").baseline, "batch_mean")

    def test_config_rifiuta_valori_illegali(self):
        # Fail fast on update hyperparameters.
        with self.assertRaises(ValueError):
            ReinforceConfig(learning_rate=-0.1)

        with self.assertRaises(ValueError):
            ReinforceConfig(baseline="non_tra_le_opzioni")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            ReinforceConfig(entropy_coef=-0.1)

    def test_baseline_none_usa_zero(self):
        # Without a baseline, the gradient uses reward_to_go directly.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        stats = reinforce_update(
            policy,  # type: ignore[arg-type]
            [episodio([2.0, 3.0])],
            ReinforceConfig(learning_rate=1.0, baseline="none"),
        )

        self.assertEqual(stats.baseline_values, ())
        self.assertAlmostEqual(float(policy.applied_gradient[0]), 5.0)

    def test_batch_mean_produce_una_baseline_globale(self):
        # batch_mean subtracts the same mean from all decisions.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        stats = reinforce_update(
            policy,  # type: ignore[arg-type]
            [episodio([2.0, 4.0]), episodio([6.0, 8.0])],
            ReinforceConfig(learning_rate=1.0, baseline="batch_mean"),
        )

        self.assertEqual(stats.baseline_values, (5.0,))
        self.assertAlmostEqual(float(policy.applied_gradient[0]), 0.0)

    def test_time_dependent_produce_una_baseline_per_indice_decisionale(self):
        # time_dependent compares each decision with the same position in the batch.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        stats = reinforce_update(
            policy,  # type: ignore[arg-type]
            [episodio([2.0, 10.0]), episodio([4.0, 20.0])],
            ReinforceConfig(learning_rate=1.0, baseline="time_dependent"),
        )

        self.assertEqual(stats.baseline_values, (3.0, 15.0))
        self.assertAlmostEqual(float(policy.applied_gradient[0]), 0.0)

    def test_update_modifica_theta_e_propaga_learning_rate(self):
        # The update applies the accumulated gradient with the configured rate.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        reinforce_update(
            policy,  # type: ignore[arg-type]
            [episodio([2.0])],
            ReinforceConfig(learning_rate=0.5, baseline="none"),
        )

        self.assertAlmostEqual(float(policy.applied_gradient[0]), 2.0)
        self.assertEqual(policy.applied_learning_rate, 0.5)
        self.assertAlmostEqual(float(policy.theta[0]), 1.0)

    def test_entropy_coef_aggiunge_gradiente_di_esplorazione(self):
        # Entropy regularization adds an exploration gradient to the policy update.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        stats = reinforce_update(
            policy,  # type: ignore[arg-type]
            [episodio([0.0])],
            ReinforceConfig(
                learning_rate=1.0,
                baseline="none",
                entropy_coef=0.25,
            ),
        )

        self.assertAlmostEqual(float(policy.applied_gradient[0]), 0.5)
        self.assertAlmostEqual(float(policy.theta[0]), 0.5)
        self.assertEqual(stats.mean_entropy, 0.7)

    def test_gradiente_normalizzato_per_episodi_non_per_step(self):
        # An episode with 10 decisions does not divide the accumulation by 10.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        reinforce_update(
            policy,  # type: ignore[arg-type]
            [episodio([1.0 for _ in range(10)])],
            ReinforceConfig(learning_rate=1.0, baseline="none"),
        )

        self.assertAlmostEqual(float(policy.applied_gradient[0]), 10.0)

    def test_train_stats_riassume_batch(self):
        # Statistics remain separate from evaluation and pool retention.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        stats = reinforce_update(
            policy,  # type: ignore[arg-type]
            [
                episodio([1.0], punteggio_squadra=70, punteggio_avversari=50, episode_return=1.0),
                episodio([3.0], punteggio_squadra=40, punteggio_avversari=80, episode_return=3.0),
            ],
            ReinforceConfig(learning_rate=0.0, baseline="none"),
        )

        self.assertEqual(stats.episodes, 2)
        self.assertEqual(stats.learner_decisions, 2)
        self.assertEqual(stats.mean_return, 2.0)
        self.assertEqual(stats.mean_score_margin, -10.0)
        self.assertEqual(stats.baseline, "none")

    def test_episodi_vuoti_solleva_value_error(self):
        # An update without episodes has no statistical meaning.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        with self.assertRaises(ValueError):
            reinforce_update(policy, [])  # type: ignore[arg-type]

    def test_episodi_senza_step_solleva_value_error(self):
        # An update without learner decisions cannot build the gradient.
        policy = FakeLinearPolicy(theta=np.asarray([0.0], dtype=np.float32))

        with self.assertRaises(ValueError):
            reinforce_update(
                policy,  # type: ignore[arg-type]
                [episodio([])],
            )


if __name__ == "__main__":
    unittest.main()
