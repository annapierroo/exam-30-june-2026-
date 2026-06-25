from __future__ import annotations

import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from policy.new_feature_set import NewFeatureSetExtractor


def osservazione(
    *,
    mano: tuple[Carta, ...] = (
        Carta("coppe", "asso"),
        Carta("denari", "due"),
        Carta("bastoni", "sette"),
    ),
    mano_compagno_visibile: bool = False,
    mano_compagno: tuple[Carta, ...] = (),
    seme_briscola: str = "denari",
    briscola_esposta: Carta = Carta("denari", "asso"),
    proprietario_briscola_esposta: int | None = None,
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    carte_giocate: tuple[CartaGiocata, ...] = (),
    punteggio_squadra: int = 0,
    punteggio_avversari: int = 0,
    carte_nel_mazzo: int = 28,
    indice_presa: int = 0,
) -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
        mano_compagno_visibile=mano_compagno_visibile,
        mano_compagno=mano_compagno,
        seme_briscola=seme_briscola,
        briscola_esposta=briscola_esposta,
        proprietario_briscola_esposta=proprietario_briscola_esposta,
        carte_sul_campo=carte_sul_campo,
        carte_giocate=carte_giocate,
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=punteggio_squadra,
        punteggio_avversari=punteggio_avversari,
        primo_giocatore_presa=1,
        giocatore_corrente=0,
        carte_nel_mazzo=carte_nel_mazzo,
        indice_presa=indice_presa,
        posizione_nella_presa=len(carte_sul_campo),
    )


def feature(extractor: NewFeatureSetExtractor, values: list[float], name: str) -> float:
    return values[extractor.feature_names.index(name)]


class TestNewFeatureSetExtractor(unittest.TestCase):
    def test_vettore_ha_dimensione_dichiarata_e_non_contiene_carta_rischiosa(self):
        # Il nuovo set resta separato e non usa la vecchia feature generica.
        extractor = NewFeatureSetExtractor()
        obs = osservazione()

        values = extractor.extract(obs, obs.mano[0])

        self.assertEqual(len(values), extractor.size())
        self.assertNotIn("carta_rischiosa", extractor.feature_names)
        self.assertNotIn(
            "carta_liscia_x_carta_prende_x_tavolo_ha_briscola_alta",
            extractor.feature_names,
        )
        self.assertTrue(all(isinstance(value, float) for value in values))

    def test_categorie_carta_separano_carico_non_briscola_e_briscole(self):
        # Asso/tre di briscola non devono confondersi con carichi non briscola.
        extractor = NewFeatureSetExtractor()
        tre_briscola = Carta("denari", "tre")
        taglietto = Carta("denari", "due")
        briscola_alta = Carta("denari", "re")
        obs = osservazione(mano=(tre_briscola, taglietto, briscola_alta))

        values_tre = extractor.extract(obs, tre_briscola)
        values_taglietto = extractor.extract(obs, taglietto)
        values_alta = extractor.extract(obs, briscola_alta)

        self.assertEqual(feature(extractor, values_tre, "carta_carico"), 1.0)
        self.assertEqual(
            feature(extractor, values_tre, "carta_carico_non_briscola"),
            0.0,
        )
        self.assertEqual(feature(extractor, values_tre, "carta_tre_briscola"), 1.0)
        self.assertEqual(feature(extractor, values_taglietto, "carta_taglietto"), 1.0)
        self.assertEqual(feature(extractor, values_alta, "carta_briscola_alta"), 1.0)

    def test_categorie_carta_riconoscono_punticino_e_carico_non_briscola(self):
        # Punticini e carichi non briscola sono categorie diverse per lo scarto.
        extractor = NewFeatureSetExtractor()
        punticino = Carta("coppe", "re")
        carico = Carta("coppe", "asso")
        obs = osservazione(mano=(punticino, carico))

        values_punticino = extractor.extract(obs, punticino)
        values_carico = extractor.extract(obs, carico)

        self.assertEqual(feature(extractor, values_punticino, "carta_punticino"), 1.0)
        self.assertEqual(
            feature(extractor, values_punticino, "carta_carico_non_briscola"),
            0.0,
        )
        self.assertEqual(
            feature(extractor, values_carico, "carta_carico_non_briscola"),
            1.0,
        )

    def test_tavolo_guarda_la_carta_attualmente_vincente(self):
        # Le feature tavolo descrivono la carta vincente, non qualunque carta vista.
        extractor = NewFeatureSetExtractor()
        carta = Carta("denari", "tre")
        obs = osservazione(
            mano=(carta,),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "asso")),
                CartaGiocata(giocatore_id=2, carta=Carta("denari", "due")),
            ),
        )

        values = extractor.extract(obs, carta)

        self.assertEqual(
            feature(extractor, values, "carta_prende_x_tavolo_ha_taglietto"),
            1.0,
        )
        self.assertEqual(
            feature(extractor, values, "carta_prende_x_tavolo_ha_carico"),
            0.0,
        )

    def test_ultimo_con_compagno_che_prende_attiva_carico_sicuro(self):
        # Da quarto, se il compagno prende, i punti caricati sono al sicuro.
        extractor = NewFeatureSetExtractor()
        carta = Carta("coppe", "tre")
        obs = osservazione(
            mano=(carta,),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "due")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "asso")),
                CartaGiocata(giocatore_id=3, carta=Carta("bastoni", "re")),
            ),
        )

        values = extractor.extract(obs, carta)

        self.assertAlmostEqual(
            feature(
                extractor,
                values,
                "posizione_quarto_x_compagno_sta_prendendo_x_punti_carta",
            ),
            10 / 11,
        )
        self.assertEqual(
            feature(
                extractor,
                values,
                "posizione_quarto_x_compagno_sta_prendendo_x_carico_non_briscola",
            ),
            1.0,
        )

    def test_ultimo_con_avversario_che_prende_attiva_presa_sicura(self):
        # Da quarto, una carta che supera l'avversario chiude la presa.
        extractor = NewFeatureSetExtractor()
        carta = Carta("coppe", "asso")
        obs = osservazione(
            mano=(carta,),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "re")),
                CartaGiocata(giocatore_id=2, carta=Carta("bastoni", "due")),
                CartaGiocata(giocatore_id=3, carta=Carta("spade", "due")),
            ),
        )

        values = extractor.extract(obs, carta)

        self.assertEqual(
            feature(
                extractor,
                values,
                "posizione_quarto_x_avversario_sta_prendendo_x_carta_prende",
            ),
            1.0,
        )
        self.assertGreater(
            feature(
                extractor,
                values,
                "posizione_quarto_x_carta_prende_x_punti_presa_x_avversario_sta_prendendo",
            ),
            0.0,
        )

    def test_briscola_esposta_futura_si_attiva_solo_se_pescata_determinabile(self):
        # Con quattro carte nel mazzo e presa chiusa, il futuro proprietario e' noto.
        extractor = NewFeatureSetExtractor()
        carta = Carta("spade", "due")
        obs = osservazione(
            mano=(carta,),
            briscola_esposta=Carta("denari", "asso"),
            carte_nel_mazzo=4,
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("denari", "tre")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "asso")),
                CartaGiocata(giocatore_id=3, carta=Carta("bastoni", "re")),
            ),
        )

        values = extractor.extract(obs, carta)

        self.assertEqual(
            feature(
                extractor,
                values,
                "carta_fa_pescare_briscola_esposta_alla_squadra_nostra",
            ),
            1.0,
        )
        self.assertAlmostEqual(
            feature(
                extractor,
                values,
                "carta_fa_pescare_briscola_esposta_alla_squadra_nostra_x_punti_briscola_esposta",
            ),
            1.0,
        )

    def test_briscola_esposta_futura_resta_spenta_se_presa_non_chiusa(self):
        # Prima dell'ultima carta della presa, la conseguenza non e' ancora certa.
        extractor = NewFeatureSetExtractor()
        carta = Carta("spade", "due")
        obs = osservazione(
            mano=(carta,),
            briscola_esposta=Carta("denari", "asso"),
            carte_nel_mazzo=4,
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("denari", "tre")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "asso")),
            ),
        )

        values = extractor.extract(obs, carta)

        self.assertEqual(
            feature(
                extractor,
                values,
                "carta_fa_pescare_briscola_esposta_alla_squadra_nostra",
            ),
            0.0,
        )
        self.assertEqual(
            feature(
                extractor,
                values,
                "carta_fa_pescare_briscola_esposta_a_squadra_avversaria",
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
