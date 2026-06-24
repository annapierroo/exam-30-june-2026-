from __future__ import annotations

import random
import unittest
from dataclasses import dataclass, field
import json

from diagnostics.decision_log import decision_log_to_dict, record_decision_log
from game.cards import Carta
from game.observation import Osservazione
from policy import RandomPolicy


@dataclass
class TrackingPolicy:
    name: str = "tracking"
    greedy_values: list[bool] = field(default_factory=list)
    selected_cards: list[Carta] = field(default_factory=list)

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        probability = 1.0 / len(osservazione.azioni_legali)
        return {carta: probability for carta in osservazione.azioni_legali}

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        self.greedy_values.append(greedy)
        carta = osservazione.azioni_legali[0]
        self.selected_cards.append(carta)
        return carta


@dataclass
class BadProbabilitiesPolicy(TrackingPolicy):
    name: str = "bad_probabilities"

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        return {}


@dataclass
class IllegalActionPolicy(TrackingPolicy):
    name: str = "illegal_action"

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        return Carta("coppe", "asso")


def random_policies() -> dict[int, RandomPolicy]:
    return {giocatore_id: RandomPolicy() for giocatore_id in range(4)}


class TestDecisionLog(unittest.TestCase):
    def test_record_decision_log_produce_partita_completa(self):
        # A complete game must produce 40 decisions and valid final scores.
        log = record_decision_log(
            policies_by_player=random_policies(),
            seed_ambiente=10,
            seed_policy=20,
            primo_giocatore_id=0,
            greedy=True,
        )

        self.assertEqual(len(log.records), 40)
        self.assertEqual(sum(log.punteggi_finali.values()), 120)
        self.assertEqual(log.seed_ambiente, 10)
        self.assertEqual(log.seed_policy, 20)
        self.assertEqual(log.primo_giocatore_id, 0)
        if log.punteggi_finali["pari"] > log.punteggi_finali["dispari"]:
            self.assertEqual(log.squadra_vincitrice, "pari")
        elif log.punteggi_finali["pari"] < log.punteggi_finali["dispari"]:
            self.assertEqual(log.squadra_vincitrice, "dispari")
        else:
            self.assertIsNone(log.squadra_vincitrice)

    def test_record_contiene_osservazione_legale_e_azioni(self):
        # The record stores the hand, actions, and probabilities seen before the choice.
        log = record_decision_log(
            policies_by_player=random_policies(),
            seed_ambiente=11,
            seed_policy=21,
            primo_giocatore_id=1,
            greedy=True,
        )

        for index, record in enumerate(log.records):
            self.assertEqual(record.step_index, index)
            self.assertEqual(record.azioni_legali, record.osservazione.azioni_legali)
            self.assertIn(record.azione, record.azioni_legali)
            self.assertEqual(
                set(record.action_probabilities),
                set(record.azioni_legali),
            )
            self.assertEqual(record.giocatore_id, record.osservazione.giocatore_id)

    def test_decision_log_to_dict_espone_mano_e_scelta(self):
        # The JSON view is meant for manual inspection of hand and chosen card.
        log = record_decision_log(
            policies_by_player=random_policies(),
            seed_ambiente=111,
            seed_policy=121,
            primo_giocatore_id=0,
            greedy=True,
            focus_giocatore_id=0,
        )

        report = decision_log_to_dict(log)
        first = report["records"][0]

        self.assertEqual(first["mano"], first["osservazione"]["mano"])
        self.assertEqual(first["focus_giocatore_id"], 0)
        self.assertEqual(first["focus_mano"], first["mano"])
        self.assertEqual(first["mani_by_player"]["0"], first["mano"])
        self.assertEqual(set(first["mani_by_player"]), {"0", "1", "2", "3"})
        self.assertEqual(first["seme_briscola"], first["osservazione"]["seme_briscola"])
        self.assertEqual(
            first["briscola_esposta"],
            first["osservazione"]["briscola_esposta"],
        )
        self.assertEqual(
            first["carte_sul_campo"],
            first["osservazione"]["carte_sul_campo"],
        )
        self.assertEqual(first["azioni_legali"], first["mano"])
        self.assertIn(first["azione"], first["mano"])
        self.assertEqual(
            set(first["action_probabilities"]),
            {carta["id"] for carta in first["azioni_legali"]},
        )
        json.dumps(report)

    def test_greedy_viene_propagato_alle_policy(self):
        # The greedy flag must reach the recorded policies unchanged.
        policies = {giocatore_id: TrackingPolicy() for giocatore_id in range(4)}

        log = record_decision_log(
            policies_by_player=policies,
            seed_ambiente=12,
            seed_policy=22,
            primo_giocatore_id=2,
            greedy=False,
        )

        self.assertEqual(len(log.records), 40)
        for policy in policies.values():
            self.assertEqual(set(policy.greedy_values), {False})

    def test_stessi_seed_producono_log_riproducibile(self):
        # With the same seed, players, actions, and scores must repeat.
        first = record_decision_log(
            policies_by_player=random_policies(),
            seed_ambiente=13,
            seed_policy=23,
            primo_giocatore_id=3,
            greedy=True,
        )
        second = record_decision_log(
            policies_by_player=random_policies(),
            seed_ambiente=13,
            seed_policy=23,
            primo_giocatore_id=3,
            greedy=True,
        )

        first_trace = [
            (
                record.giocatore_id,
                record.azione,
                record.outcome.punteggi,
                record.outcome.presa_completata,
            )
            for record in first.records
        ]
        second_trace = [
            (
                record.giocatore_id,
                record.azione,
                record.outcome.punteggi,
                record.outcome.presa_completata,
            )
            for record in second.records
        ]
        self.assertEqual(first_trace, second_trace)

    def test_policies_by_player_deve_contenere_quattro_giocatori(self):
        # Each player must have exactly one explicit policy.
        with self.assertRaises(ValueError):
            record_decision_log(
                policies_by_player={0: RandomPolicy(), 1: RandomPolicy()},
                seed_ambiente=14,
                seed_policy=24,
                primo_giocatore_id=0,
                greedy=True,
            )

        with self.assertRaises(ValueError):
            record_decision_log(
                policies_by_player={
                    0: RandomPolicy(),
                    1: RandomPolicy(),
                    2: RandomPolicy(),
                    3: RandomPolicy(),
                    4: RandomPolicy(),
                },
                seed_ambiente=14,
                seed_policy=24,
                primo_giocatore_id=0,
                greedy=True,
            )

    def test_primo_giocatore_non_valido_solleva_value_error(self):
        # The first player of the game must be a valid id.
        with self.assertRaises(ValueError):
            record_decision_log(
                policies_by_player=random_policies(),
                seed_ambiente=15,
                seed_policy=25,
                primo_giocatore_id=4,
                greedy=True,
            )

    def test_probabilities_non_allineate_alle_azioni_legali_solleva_value_error(self):
        # Stored probabilities must refer only and entirely to playable cards.
        policies = random_policies()
        policies[0] = BadProbabilitiesPolicy()

        with self.assertRaises(ValueError):
            record_decision_log(
                policies_by_player=policies,
                seed_ambiente=16,
                seed_policy=26,
                primo_giocatore_id=0,
                greedy=True,
            )

    def test_policy_che_sceglie_azione_illegale_solleva_value_error(self):
        # A card not present in the hand is rejected as an illegal choice.
        policies = random_policies()
        policies[0] = IllegalActionPolicy()

        with self.assertRaises(ValueError):
            record_decision_log(
                policies_by_player=policies,
                seed_ambiente=17,
                seed_policy=27,
                primo_giocatore_id=0,
                greedy=True,
            )

    def test_outcome_non_contiene_osservazione_successiva(self):
        # The post-move outcome must not include the next player's observation.
        log = record_decision_log(
            policies_by_player=random_policies(),
            seed_ambiente=18,
            seed_policy=28,
            primo_giocatore_id=0,
            greedy=True,
        )

        self.assertFalse(hasattr(log.records[0].outcome, "osservazione"))


if __name__ == "__main__":
    unittest.main()
