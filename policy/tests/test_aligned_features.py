from __future__ import annotations

import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from policy.aligned_features import (
    BASE_ALIGNED_FEATURE_NAMES,
    COMMON_ATOMIC_FEATURE_NAMES,
    NEW_ALIGNED_FEATURE_NAMES,
    NEW_CONTEXT_FACTOR_FEATURE_NAMES,
    AlignedFeatureExtractor,
)
from policy.features import (
    DEFAULT_ATOMIC_FEATURE_NAMES as BASE_ATOMIC_FEATURE_NAMES,
    BriscolaFeatureExtractor,
)
from policy.new_feature_set import (
    DEFAULT_ATOMIC_FEATURE_NAMES as NEW_ATOMIC_FEATURE_NAMES,
    NewFeatureSetExtractor,
)


def osservazione(
    *,
    mano: tuple[Carta, ...],
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    indice_presa: int = 0,
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
        carte_sul_campo=carte_sul_campo,
        carte_giocate=(),
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=1,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=indice_presa,
        posizione_nella_presa=len(carte_sul_campo),
    )


class TestAlignedFeatureSchema(unittest.TestCase):
    def test_schema_comune_non_contiene_duplicati(self):
        # The shared schema must give one coordinate to each atomic factor.
        self.assertEqual(
            len(COMMON_ATOMIC_FEATURE_NAMES),
            len(set(COMMON_ATOMIC_FEATURE_NAMES)),
        )
        self.assertEqual(len(COMMON_ATOMIC_FEATURE_NAMES), 73)

    def test_schema_comune_contiene_base_new_e_fattori_contesto(self):
        # Existing atoms and dissolved new interaction factors stay available.
        common_names = set(COMMON_ATOMIC_FEATURE_NAMES)

        self.assertTrue(set(BASE_ATOMIC_FEATURE_NAMES) <= common_names)
        self.assertTrue(set(NEW_ATOMIC_FEATURE_NAMES) <= common_names)
        self.assertTrue(set(NEW_CONTEXT_FACTOR_FEATURE_NAMES) <= common_names)

    def test_schema_allineati_conservano_atomi_comuni_e_solo_propri_prodotti(self):
        # Le due varianti differiscono solo nella coda di interazioni engineered.
        self.assertEqual(len(BASE_ALIGNED_FEATURE_NAMES), 85)
        self.assertEqual(len(NEW_ALIGNED_FEATURE_NAMES), 141)
        self.assertEqual(
            BASE_ALIGNED_FEATURE_NAMES[: len(COMMON_ATOMIC_FEATURE_NAMES)],
            COMMON_ATOMIC_FEATURE_NAMES,
        )
        self.assertEqual(
            NEW_ALIGNED_FEATURE_NAMES[: len(COMMON_ATOMIC_FEATURE_NAMES)],
            COMMON_ATOMIC_FEATURE_NAMES,
        )

    def test_nomi_atomici_condivisi_hanno_stesso_valore_nelle_due_implementazioni(self):
        # I nomi sovrapposti devono mantenere formula e scala identiche.
        shared_names = tuple(
            name for name in BASE_ATOMIC_FEATURE_NAMES if name in NEW_ATOMIC_FEATURE_NAMES
        )
        base = BriscolaFeatureExtractor(feature_names=list(shared_names))
        new = NewFeatureSetExtractor(feature_names=list(shared_names))
        carta = Carta("denari", "tre")
        observations = (
            osservazione(mano=(carta,)),
            osservazione(
                mano=(carta,),
                carte_sul_campo=(
                    CartaGiocata(giocatore_id=1, carta=Carta("coppe", "asso")),
                    CartaGiocata(giocatore_id=2, carta=Carta("denari", "due")),
                ),
                indice_presa=6,
            ),
        )

        for obs in observations:
            self.assertEqual(base.extract(obs, carta), new.extract(obs, carta))

    def test_vettori_allineati_condividono_lo_stesso_prefisso_atomico(self):
        # A parita' di osservazione, la base comune e' identica per entrambe.
        carta = Carta("denari", "tre")
        obs = osservazione(
            mano=(carta,),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "asso")),
                CartaGiocata(giocatore_id=2, carta=Carta("bastoni", "re")),
                CartaGiocata(giocatore_id=3, carta=Carta("denari", "due")),
            ),
            indice_presa=6,
        )
        base = AlignedFeatureExtractor(feature_names=list(BASE_ALIGNED_FEATURE_NAMES))
        new = AlignedFeatureExtractor(feature_names=list(NEW_ALIGNED_FEATURE_NAMES))

        base_values = base.extract(obs, carta)
        new_values = new.extract(obs, carta)

        self.assertEqual(base_values[:73], new_values[:73])
        self.assertEqual(base.size(), 85)
        self.assertEqual(new.size(), 141)


if __name__ == "__main__":
    unittest.main()
