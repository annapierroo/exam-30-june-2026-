from __future__ import annotations

import random
import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from policy import AdvancedHeuristicPolicy


def osservazione(
    *,
    mano: tuple[Carta, ...],
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    seme_briscola: str = "denari",
    posizione_nella_presa: int | None = None,
) -> Osservazione:
    if posizione_nella_presa is None:
        posizione_nella_presa = len(carte_sul_campo)

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
        posizione_nella_presa=posizione_nella_presa,
    )


class TestAdvancedHeuristicPolicy(unittest.TestCase):
    def test_apre_con_scarto_meno_costoso(self):
        # A presa vuota conserva punti e briscole.
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                Carta("coppe", "asso"),
                scarto,
            )
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_compagno_prende_non_ultimo_lascia_compagno_in_testa(self):
        # Se un avversario deve ancora giocare, non carica e non supera il compagno.
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("coppe", "asso"),
                Carta("denari", "due"),
                Carta("spade", "tre"),
                scarto,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "re")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_compagno_prende_ultimo_carica_con_carico_non_briscola(self):
        # Da ultimo puo' caricare punti sulla presa della squadra senza rischio.
        carico = Carta("bastoni", "asso")
        obs = osservazione(
            mano=(
                Carta("denari", "tre"),
                carico,
                Carta("spade", "due"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "re")),
                CartaGiocata(giocatore_id=3, carta=Carta("spade", "sette")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, carico)

    def test_avversario_prende_ultimo_gioca_carico_non_briscola_che_prende(self):
        # Da ultimo puo' prendere con un carico non briscola per metterlo al sicuro.
        carico_che_prende = Carta("coppe", "asso")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                carico_che_prende,
                Carta("coppe", "fante"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "re")),
                CartaGiocata(giocatore_id=2, carta=Carta("spade", "due")),
                CartaGiocata(giocatore_id=3, carta=Carta("bastoni", "due")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, carico_che_prende)

    def test_avversario_prende_ultimo_su_presa_povera_non_spende_briscola(self):
        # Su presa povera non spende briscola se non puo' prendere senza briscola.
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                Carta("spade", "sette"),
                scarto,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("spade", "cinque")),
                CartaGiocata(giocatore_id=3, carta=Carta("bastoni", "quattro")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_avversario_prende_non_ultimo_su_presa_ricca_spende_briscola_bassa(self):
        # Su presa ricca e non da ultimo, una briscola bassa forza l'avversario dopo.
        briscola_bassa = Carta("denari", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "tre"),
                briscola_bassa,
                Carta("coppe", "re"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "asso")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, briscola_bassa)

    def test_avversario_prende_non_ultimo_su_presa_povera_prende_senza_carico_o_briscola(self):
        # Su presa povera prende solo se basta una carta non carico e non briscola.
        presa_povera = Carta("coppe", "sette")
        obs = osservazione(
            mano=(
                Carta("coppe", "asso"),
                Carta("denari", "due"),
                presa_povera,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, presa_povera)

    def test_probabilita_divise_tra_carte_migliori_equivalenti(self):
        # Le carte equivalenti restano entrambe disponibili alla scelta stocastica.
        primo_scarto = Carta("coppe", "due")
        secondo_scarto = Carta("bastoni", "due")
        briscola = Carta("denari", "due")
        obs = osservazione(mano=(primo_scarto, secondo_scarto, briscola))
        policy = AdvancedHeuristicPolicy()

        probabilities = policy.action_probabilities(obs)

        self.assertAlmostEqual(probabilities[primo_scarto], 0.5)
        self.assertAlmostEqual(probabilities[secondo_scarto], 0.5)
        self.assertAlmostEqual(probabilities[briscola], 0.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_mano_vuota_solleva_value_error(self):
        # Una policy non puo' scegliere se non ci sono azioni legali.
        obs = osservazione(mano=())
        policy = AdvancedHeuristicPolicy()

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
