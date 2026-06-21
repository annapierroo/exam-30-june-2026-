from __future__ import annotations

import random
import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from policy import GreedyPolicy, HeuristicPolicy


def osservazione(
    *,
    mano: tuple[Carta, ...],
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    seme_briscola: str = "denari",
) -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
        mano_compagno_visibile=False,
        mano_compagno=(),
        seme_briscola=seme_briscola,
        briscola_esposta=Carta(seme_briscola, "asso"),
        proprietario_briscola_esposta=None,
        carte_sul_campo=carte_sul_campo,
        carte_giocate=carte_sul_campo,
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=1,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=0,
        posizione_nella_presa=len(carte_sul_campo),
    )


class TestHeuristicPolicy(unittest.TestCase):
    def test_delega_a_greedy_quando_compagno_non_sta_prendendo(self):
        obs = osservazione(
            mano=(
                Carta("coppe", "fante"),
                Carta("coppe", "sette"),
                Carta("bastoni", "due"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
            ),
        )
        heuristic = HeuristicPolicy()
        greedy = GreedyPolicy()

        heuristic_action = heuristic.select_action(obs, rng=random.Random(0))
        greedy_action = greedy.select_action(obs, rng=random.Random(0))

        self.assertEqual(heuristic_action, greedy_action)

    def test_scarto_meno_costoso_quando_compagno_sta_prendendo(self):
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                Carta("coppe", "asso"),
                scarto,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "asso")),
            ),
        )
        policy = HeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_non_supera_il_compagno_anche_se_potrebbe_prendere(self):
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("coppe", "tre"),
                Carta("denari", "due"),
                scarto,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "re")),
            ),
        )
        policy = HeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_probabilita_divise_tra_scarti_equivalenti_se_compagno_sta_prendendo(self):
        primo_scarto = Carta("coppe", "due")
        secondo_scarto = Carta("bastoni", "due")
        briscola = Carta("denari", "due")
        obs = osservazione(
            mano=(primo_scarto, secondo_scarto, briscola),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("spade", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("spade", "asso")),
            ),
        )
        policy = HeuristicPolicy()

        probabilities = policy.action_probabilities(obs)

        self.assertAlmostEqual(probabilities[primo_scarto], 0.5)
        self.assertAlmostEqual(probabilities[secondo_scarto], 0.5)
        self.assertAlmostEqual(probabilities[briscola], 0.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_mano_vuota_solleva_value_error(self):
        obs = osservazione(mano=())
        policy = HeuristicPolicy()

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
