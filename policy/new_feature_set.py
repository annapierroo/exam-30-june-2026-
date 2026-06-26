"""Alternative feature set for Briscola learnable policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from game.cards import Carta, CartaGiocata, crea_mazzo
from game.observation import Osservazione
from game.rules import NUMERO_GIOCATORI, ordine_giocatori_da, punti_presa, vincitore_presa


MAX_PUNTI_CARTA = 11
MAX_PUNTI_PRESA = 44
PUNTI_TOTALI = 120
TOTALE_BRISCOLE = 10
MAX_FORZA_CARTA = 10
MAX_CARTE_SUPERIORI = 9

FIGURE = {"re", "cavallo", "fante"}
CARICHI = {"asso", "tre"}


DEFAULT_ATOMIC_FEATURE_NAMES: tuple[str, ...] = (
    "punti_carta",
    "forza_carta",
    "carta_briscola",
    "carta_asso",
    "carta_tre",
    "carta_figura",
    "carta_carico",
    "carta_liscia",
    "carta_prende",
    "carta_supera_compagno",
    "carta_supera_avversario",
    "superiori_stesso_seme_non_osservate",
    "briscole_che_battono_non_osservate",
    "carta_punticino",
    "carta_carico_non_briscola",
    "carta_taglietto",
    "carta_briscola_alta",
    "carta_tre_briscola",
    "carta_fa_pescare_briscola_esposta_a_squadra_avversaria",
    "carta_fa_pescare_briscola_esposta_alla_squadra_nostra",
)

# These values already feed engineered products below. Keeping them selectable
# on their own lets another policy representation learn non-linear products.
NEW_CONTEXT_FACTOR_FEATURE_NAMES: tuple[str, ...] = (
    "presa_povera",
    "presa_media",
    "presa_ricca",
    "ultime_quattro_prese",
    "quartultima_presa",
    "tavolo_ha_carico",
    "tavolo_ha_taglietto",
    "tavolo_ha_briscola_alta",
    "punti_briscola_esposta",
    "forza_briscola_esposta",
)

DEFAULT_INTERACTION_FEATURE_NAMES: tuple[str, ...] = (
    "posizione_primo_x_carta_liscia",
    "posizione_primo_x_carta_punticino",
    "posizione_primo_x_carta_carico_non_briscola",
    "posizione_primo_x_carta_taglietto",
    "posizione_primo_x_carta_briscola_alta",
    "posizione_quarto_x_compagno_sta_prendendo_x_punti_carta",
    "posizione_quarto_x_compagno_sta_prendendo_x_carico_non_briscola",
    "posizione_quarto_x_compagno_sta_prendendo_x_carta_punticino",
    "posizione_quarto_x_compagno_sta_prendendo_x_carta_liscia",
    "posizione_quarto_x_compagno_sta_prendendo_x_carta_briscola",
    "avversari_dopo_x_compagno_sta_prendendo_x_punti_carta",
    "avversari_dopo_x_compagno_sta_prendendo_x_carico_non_briscola",
    "avversari_dopo_x_compagno_sta_prendendo_x_carta_briscola",
    "compagno_puo_prendere_x_avversario_sta_prendendo_x_carta_liscia",
    "compagno_puo_prendere_x_avversario_sta_prendendo_x_carta_punticino",
    "compagno_puo_prendere_x_avversario_sta_prendendo_x_carta_briscola",
    "compagno_puo_prendere_x_avversario_sta_prendendo_x_carico_non_briscola",
    "posizione_quarto_x_avversario_sta_prendendo_x_carta_prende",
    "posizione_quarto_x_carta_prende_x_punti_presa_x_avversario_sta_prendendo",
    "posizione_quarto_x_carta_prende_x_carta_carico_non_briscola",
    "posizione_quarto_x_carta_prende_x_carta_liscia",
    "posizione_quarto_x_carta_prende_x_carta_punticino",
    "posizione_quarto_x_carta_prende_x_carta_taglietto",
    "posizione_quarto_x_carta_prende_x_carta_briscola_alta",
    "avversari_dopo_x_carta_prende_x_punti_presa",
    "avversari_dopo_x_carta_prende_x_carta_briscola",
    "avversari_dopo_x_carta_prende_x_carico_non_briscola",
    "avversari_dopo_x_carta_prende_x_briscole_che_battono_non_osservate",
    "avversari_dopo_x_carta_prende_x_superiori_stesso_seme_non_osservate",
    "carta_briscola_x_carta_prende",
    "carta_briscola_x_carta_prende_x_presa_povera",
    "carta_briscola_x_carta_prende_x_presa_media",
    "carta_briscola_x_carta_prende_x_presa_ricca",
    "carta_liscia_x_carta_prende",
    "carta_liscia_x_carta_prende_x_presa_ricca",
    "carta_punticino_x_carta_prende",
    "carta_punticino_x_carta_prende_x_presa_ricca",
    "carta_carico_non_briscola_x_carta_prende",
    "fase_iniziale_x_carta_briscola",
    "fase_iniziale_x_carta_carico_non_briscola",
    "fase_iniziale_x_carta_taglietto",
    "fase_iniziale_x_carta_briscola_alta",
    "ultime_quattro_prese_x_carta_briscola",
    "ultime_quattro_prese_x_carta_prende",
    "ultime_quattro_prese_x_carta_carico_non_briscola",
    "quartultima_presa_x_forza_briscola_esposta_x_carta_prende_x_punti_presa",
    "fase_finale_x_carta_briscola",
    "fase_finale_x_carta_carico_non_briscola",
    "fase_finale_x_carta_prende_x_punti_presa",
    "mazzo_vuoto_x_carta_prende_x_punti_presa",
    "carta_prende_x_superiori_stesso_seme_non_osservate",
    "carta_prende_x_briscole_che_battono_non_osservate",
    "carta_briscola_x_briscole_che_battono_non_osservate",
    "fase_finale_x_carta_prende_x_briscole_che_battono_non_osservate",
    "fase_finale_x_carta_prende_x_superiori_stesso_seme_non_osservate",
    "carta_prende_x_tavolo_ha_carico",
    "carta_prende_x_tavolo_ha_briscola_alta",
    "carta_prende_x_tavolo_ha_taglietto",
    "carta_briscola_x_carta_prende_x_tavolo_ha_carico",
    "ultime_quattro_prese_x_carta_prende_x_tavolo_ha_briscola_alta",
    "differenza_punteggio_x_carta_prende",
    "differenza_punteggio_x_carta_carico_non_briscola",
    "punti_squadra_nostra_x_punti_presa_corrente_x_carta_prende_x_briscola_x_avversario_sta_prendendo_x_punti_carta",
    "punti_squadra_avversaria_x_punti_presa_corrente_x_carta_prende_x_briscola_x_avversario_sta_prendendo",
    "carta_fa_pescare_briscola_esposta_alla_squadra_nostra_x_punti_briscola_esposta",
    "carta_fa_pescare_briscola_esposta_a_squadra_avversaria_x_punti_briscola_esposta",
    "carta_fa_pescare_briscola_esposta_alla_squadra_nostra_x_forza_briscola_esposta",
    "carta_fa_pescare_briscola_esposta_a_squadra_avversaria_x_forza_briscola_esposta",
)

# Preserve the historical full-vector order for checkpoint compatibility.
DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    DEFAULT_ATOMIC_FEATURE_NAMES[:-2]
    + DEFAULT_INTERACTION_FEATURE_NAMES[:-4]
    + DEFAULT_ATOMIC_FEATURE_NAMES[-2:]
    + DEFAULT_INTERACTION_FEATURE_NAMES[-4:]
)


@dataclass
class NewFeatureSetExtractor:
    """Build ``phi(osservazione, carta)`` for the new feature set."""

    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(DEFAULT_FEATURE_NAMES)

    def size(self) -> int:
        """Return the feature vector size."""

        return len(self.feature_names)

    @property
    def atomic_feature_names(self) -> tuple[str, ...]:
        """Return the active non-interaction feature names."""

        atomic_names = set(
            DEFAULT_ATOMIC_FEATURE_NAMES + NEW_CONTEXT_FACTOR_FEATURE_NAMES,
        )
        return tuple(name for name in self.feature_names if name in atomic_names)

    @property
    def interaction_feature_names(self) -> tuple[str, ...]:
        """Return the active engineered interaction feature names."""

        interaction_names = set(DEFAULT_INTERACTION_FEATURE_NAMES)
        return tuple(name for name in self.feature_names if name in interaction_names)

    def extract(self, osservazione: Osservazione, carta: Carta) -> list[float]:
        """Extract features from legal information and one legal candidate card."""

        if carta not in osservazione.azioni_legali:
            raise ValueError("Features can only be extracted for legal hand cards")

        vincitore_corrente = self._vincitore_corrente(osservazione)
        vincitore_candidato = self._vincitore_dopo_carta(
            osservazione=osservazione,
            carta=carta,
            giocatore_id=osservazione.giocatore_id,
        )
        carta_vincente_tavolo = self._carta_vincente_tavolo(osservazione)
        carte_osservate = self._carte_osservate(osservazione)

        compagno_sta_prendendo = vincitore_corrente == osservazione.compagno_id
        avversario_sta_prendendo = vincitore_corrente in osservazione.avversari
        carta_prende = vincitore_candidato == osservazione.giocatore_id
        carta_supera_compagno = compagno_sta_prendendo and carta_prende
        carta_supera_avversario = avversario_sta_prendendo and carta_prende

        giocatori_dopo = max(0, 3 - osservazione.posizione_nella_presa)
        ordine_dopo = [
            (osservazione.giocatore_id + offset) % NUMERO_GIOCATORI
            for offset in range(1, giocatori_dopo + 1)
        ]
        avversari_dopo = sum(
            1 for giocatore in ordine_dopo if giocatore in osservazione.avversari
        )
        avversari_dopo_norm = avversari_dopo / 2

        carta_briscola = self._briscola(osservazione, carta)
        carta_liscia = carta.punti == 0 and not carta_briscola
        carta_punticino = self._punticino(osservazione, carta)
        carta_carico = self._carico(carta)
        carta_carico_non_briscola = self._carico_non_briscola(osservazione, carta)
        carta_taglietto = self._taglietto(osservazione, carta)
        carta_briscola_alta = self._briscola_alta(osservazione, carta)
        carta_tre_briscola = carta.rango == "tre" and carta_briscola

        punti_carta = carta.punti / MAX_PUNTI_CARTA
        forza_carta = carta.forza / MAX_FORZA_CARTA
        punti_presa_corrente = punti_presa(osservazione.carte_sul_campo)
        punti_presa_norm = punti_presa_corrente / MAX_PUNTI_PRESA
        differenza_punteggio = (
            osservazione.punteggio_squadra - osservazione.punteggio_avversari
        ) / PUNTI_TOTALI

        presa_povera = punti_presa_corrente <= 4
        presa_media = 5 <= punti_presa_corrente <= 10
        presa_ricca = punti_presa_corrente > 10
        fase_iniziale = osservazione.indice_presa <= 2
        fase_finale = osservazione.indice_presa >= 7
        ultime_quattro_prese = osservazione.indice_presa >= 6
        quartultima_presa = osservazione.indice_presa == 6
        mazzo_vuoto = osservazione.carte_nel_mazzo == 0

        posizione_primo = osservazione.posizione_nella_presa == 0
        posizione_quarto = osservazione.posizione_nella_presa == 3

        compagno_puo_prendere = self._compagno_puo_prendere(osservazione)
        superiori_stesso_seme = self._superiori_stesso_seme_non_osservate(
            carta=carta,
            carte_osservate=carte_osservate,
        )
        briscole_che_battono = self._briscole_che_battono_non_osservate(
            osservazione=osservazione,
            carta=carta,
            carte_osservate=carte_osservate,
        )
        superiori_stesso_seme_norm = superiori_stesso_seme / MAX_CARTE_SUPERIORI
        briscole_che_battono_norm = briscole_che_battono / TOTALE_BRISCOLE

        tavolo_ha_carico = self._carico_non_briscola(
            osservazione,
            carta_vincente_tavolo,
        )
        tavolo_ha_taglietto = self._taglietto(osservazione, carta_vincente_tavolo)
        tavolo_ha_briscola_alta = self._briscola_alta(
            osservazione,
            carta_vincente_tavolo,
        )

        briscola_esposta_futura = self._briscola_esposta_futura(osservazione, carta)
        fa_pescare_nostra = briscola_esposta_futura == "nostra"
        fa_pescare_avversaria = briscola_esposta_futura == "avversaria"
        punti_briscola_esposta = osservazione.briscola_esposta.punti / MAX_PUNTI_CARTA
        forza_briscola_esposta = osservazione.briscola_esposta.forza / MAX_FORZA_CARTA

        values = {
            "punti_carta": punti_carta,
            "forza_carta": forza_carta,
            "carta_briscola": float(carta_briscola),
            "carta_asso": float(carta.rango == "asso"),
            "carta_tre": float(carta.rango == "tre"),
            "carta_figura": float(carta.rango in FIGURE),
            "carta_carico": float(carta_carico),
            "carta_liscia": float(carta_liscia),
            "carta_prende": float(carta_prende),
            "carta_supera_compagno": float(carta_supera_compagno),
            "carta_supera_avversario": float(carta_supera_avversario),
            "superiori_stesso_seme_non_osservate": superiori_stesso_seme_norm,
            "briscole_che_battono_non_osservate": briscole_che_battono_norm,
            "carta_punticino": float(carta_punticino),
            "carta_carico_non_briscola": float(carta_carico_non_briscola),
            "carta_taglietto": float(carta_taglietto),
            "carta_briscola_alta": float(carta_briscola_alta),
            "carta_tre_briscola": float(carta_tre_briscola),
            "presa_povera": float(presa_povera),
            "presa_media": float(presa_media),
            "presa_ricca": float(presa_ricca),
            "ultime_quattro_prese": float(ultime_quattro_prese),
            "quartultima_presa": float(quartultima_presa),
            "tavolo_ha_carico": float(tavolo_ha_carico),
            "tavolo_ha_taglietto": float(tavolo_ha_taglietto),
            "tavolo_ha_briscola_alta": float(tavolo_ha_briscola_alta),
            "punti_briscola_esposta": punti_briscola_esposta,
            "forza_briscola_esposta": forza_briscola_esposta,
            "posizione_primo_x_carta_liscia": (
                float(posizione_primo) * float(carta_liscia)
            ),
            "posizione_primo_x_carta_punticino": (
                float(posizione_primo) * float(carta_punticino)
            ),
            "posizione_primo_x_carta_carico_non_briscola": (
                float(posizione_primo) * float(carta_carico_non_briscola)
            ),
            "posizione_primo_x_carta_taglietto": (
                float(posizione_primo) * float(carta_taglietto)
            ),
            "posizione_primo_x_carta_briscola_alta": (
                float(posizione_primo) * float(carta_briscola_alta)
            ),
            "posizione_quarto_x_compagno_sta_prendendo_x_punti_carta": (
                float(posizione_quarto) * float(compagno_sta_prendendo) * punti_carta
            ),
            "posizione_quarto_x_compagno_sta_prendendo_x_carico_non_briscola": (
                float(posizione_quarto)
                * float(compagno_sta_prendendo)
                * float(carta_carico_non_briscola)
            ),
            "posizione_quarto_x_compagno_sta_prendendo_x_carta_punticino": (
                float(posizione_quarto)
                * float(compagno_sta_prendendo)
                * float(carta_punticino)
            ),
            "posizione_quarto_x_compagno_sta_prendendo_x_carta_liscia": (
                float(posizione_quarto)
                * float(compagno_sta_prendendo)
                * float(carta_liscia)
            ),
            "posizione_quarto_x_compagno_sta_prendendo_x_carta_briscola": (
                float(posizione_quarto)
                * float(compagno_sta_prendendo)
                * float(carta_briscola)
            ),
            "avversari_dopo_x_compagno_sta_prendendo_x_punti_carta": (
                avversari_dopo_norm * float(compagno_sta_prendendo) * punti_carta
            ),
            "avversari_dopo_x_compagno_sta_prendendo_x_carico_non_briscola": (
                avversari_dopo_norm
                * float(compagno_sta_prendendo)
                * float(carta_carico_non_briscola)
            ),
            "avversari_dopo_x_compagno_sta_prendendo_x_carta_briscola": (
                avversari_dopo_norm
                * float(compagno_sta_prendendo)
                * float(carta_briscola)
            ),
            "compagno_puo_prendere_x_avversario_sta_prendendo_x_carta_liscia": (
                float(compagno_puo_prendere)
                * float(avversario_sta_prendendo)
                * float(carta_liscia)
            ),
            "compagno_puo_prendere_x_avversario_sta_prendendo_x_carta_punticino": (
                float(compagno_puo_prendere)
                * float(avversario_sta_prendendo)
                * float(carta_punticino)
            ),
            "compagno_puo_prendere_x_avversario_sta_prendendo_x_carta_briscola": (
                float(compagno_puo_prendere)
                * float(avversario_sta_prendendo)
                * float(carta_briscola)
            ),
            "compagno_puo_prendere_x_avversario_sta_prendendo_x_carico_non_briscola": (
                float(compagno_puo_prendere)
                * float(avversario_sta_prendendo)
                * float(carta_carico_non_briscola)
            ),
            "posizione_quarto_x_avversario_sta_prendendo_x_carta_prende": (
                float(posizione_quarto) * float(avversario_sta_prendendo) * float(carta_prende)
            ),
            "posizione_quarto_x_carta_prende_x_punti_presa_x_avversario_sta_prendendo": (
                float(posizione_quarto)
                * float(carta_prende)
                * punti_presa_norm
                * float(avversario_sta_prendendo)
            ),
            "posizione_quarto_x_carta_prende_x_carta_carico_non_briscola": (
                float(posizione_quarto)
                * float(carta_prende)
                * float(carta_carico_non_briscola)
            ),
            "posizione_quarto_x_carta_prende_x_carta_liscia": (
                float(posizione_quarto) * float(carta_prende) * float(carta_liscia)
            ),
            "posizione_quarto_x_carta_prende_x_carta_punticino": (
                float(posizione_quarto) * float(carta_prende) * float(carta_punticino)
            ),
            "posizione_quarto_x_carta_prende_x_carta_taglietto": (
                float(posizione_quarto) * float(carta_prende) * float(carta_taglietto)
            ),
            "posizione_quarto_x_carta_prende_x_carta_briscola_alta": (
                float(posizione_quarto)
                * float(carta_prende)
                * float(carta_briscola_alta)
            ),
            "avversari_dopo_x_carta_prende_x_punti_presa": (
                avversari_dopo_norm * float(carta_prende) * punti_presa_norm
            ),
            "avversari_dopo_x_carta_prende_x_carta_briscola": (
                avversari_dopo_norm * float(carta_prende) * float(carta_briscola)
            ),
            "avversari_dopo_x_carta_prende_x_carico_non_briscola": (
                avversari_dopo_norm
                * float(carta_prende)
                * float(carta_carico_non_briscola)
            ),
            "avversari_dopo_x_carta_prende_x_briscole_che_battono_non_osservate": (
                avversari_dopo_norm * float(carta_prende) * briscole_che_battono_norm
            ),
            "avversari_dopo_x_carta_prende_x_superiori_stesso_seme_non_osservate": (
                avversari_dopo_norm * float(carta_prende) * superiori_stesso_seme_norm
            ),
            "carta_briscola_x_carta_prende": float(carta_briscola)
            * float(carta_prende),
            "carta_briscola_x_carta_prende_x_presa_povera": (
                float(carta_briscola) * float(carta_prende) * float(presa_povera)
            ),
            "carta_briscola_x_carta_prende_x_presa_media": (
                float(carta_briscola) * float(carta_prende) * float(presa_media)
            ),
            "carta_briscola_x_carta_prende_x_presa_ricca": (
                float(carta_briscola) * float(carta_prende) * float(presa_ricca)
            ),
            "carta_liscia_x_carta_prende": float(carta_liscia)
            * float(carta_prende),
            "carta_liscia_x_carta_prende_x_presa_ricca": (
                float(carta_liscia) * float(carta_prende) * float(presa_ricca)
            ),
            "carta_punticino_x_carta_prende": float(carta_punticino)
            * float(carta_prende),
            "carta_punticino_x_carta_prende_x_presa_ricca": (
                float(carta_punticino) * float(carta_prende) * float(presa_ricca)
            ),
            "carta_carico_non_briscola_x_carta_prende": (
                float(carta_carico_non_briscola) * float(carta_prende)
            ),
            "fase_iniziale_x_carta_briscola": float(fase_iniziale)
            * float(carta_briscola),
            "fase_iniziale_x_carta_carico_non_briscola": (
                float(fase_iniziale) * float(carta_carico_non_briscola)
            ),
            "fase_iniziale_x_carta_taglietto": float(fase_iniziale)
            * float(carta_taglietto),
            "fase_iniziale_x_carta_briscola_alta": float(fase_iniziale)
            * float(carta_briscola_alta),
            "ultime_quattro_prese_x_carta_briscola": (
                float(ultime_quattro_prese) * float(carta_briscola)
            ),
            "ultime_quattro_prese_x_carta_prende": (
                float(ultime_quattro_prese) * float(carta_prende)
            ),
            "ultime_quattro_prese_x_carta_carico_non_briscola": (
                float(ultime_quattro_prese) * float(carta_carico_non_briscola)
            ),
            "quartultima_presa_x_forza_briscola_esposta_x_carta_prende_x_punti_presa": (
                float(quartultima_presa)
                * forza_briscola_esposta
                * float(carta_prende)
                * punti_presa_norm
            ),
            "fase_finale_x_carta_briscola": float(fase_finale)
            * float(carta_briscola),
            "fase_finale_x_carta_carico_non_briscola": (
                float(fase_finale) * float(carta_carico_non_briscola)
            ),
            "fase_finale_x_carta_prende_x_punti_presa": (
                float(fase_finale) * float(carta_prende) * punti_presa_norm
            ),
            "mazzo_vuoto_x_carta_prende_x_punti_presa": (
                float(mazzo_vuoto) * float(carta_prende) * punti_presa_norm
            ),
            "carta_prende_x_superiori_stesso_seme_non_osservate": (
                float(carta_prende) * superiori_stesso_seme_norm
            ),
            "carta_prende_x_briscole_che_battono_non_osservate": (
                float(carta_prende) * briscole_che_battono_norm
            ),
            "carta_briscola_x_briscole_che_battono_non_osservate": (
                float(carta_briscola) * briscole_che_battono_norm
            ),
            "fase_finale_x_carta_prende_x_briscole_che_battono_non_osservate": (
                float(fase_finale) * float(carta_prende) * briscole_che_battono_norm
            ),
            "fase_finale_x_carta_prende_x_superiori_stesso_seme_non_osservate": (
                float(fase_finale) * float(carta_prende) * superiori_stesso_seme_norm
            ),
            "carta_prende_x_tavolo_ha_carico": float(carta_prende)
            * float(tavolo_ha_carico),
            "carta_prende_x_tavolo_ha_briscola_alta": float(carta_prende)
            * float(tavolo_ha_briscola_alta),
            "carta_prende_x_tavolo_ha_taglietto": float(carta_prende)
            * float(tavolo_ha_taglietto),
            "carta_briscola_x_carta_prende_x_tavolo_ha_carico": (
                float(carta_briscola) * float(carta_prende) * float(tavolo_ha_carico)
            ),
            "ultime_quattro_prese_x_carta_prende_x_tavolo_ha_briscola_alta": (
                float(ultime_quattro_prese)
                * float(carta_prende)
                * float(tavolo_ha_briscola_alta)
            ),
            "differenza_punteggio_x_carta_prende": differenza_punteggio
            * float(carta_prende),
            "differenza_punteggio_x_carta_carico_non_briscola": (
                differenza_punteggio * float(carta_carico_non_briscola)
            ),
            "punti_squadra_nostra_x_punti_presa_corrente_x_carta_prende_x_briscola_x_avversario_sta_prendendo_x_punti_carta": (
                (osservazione.punteggio_squadra / PUNTI_TOTALI)
                * punti_presa_norm
                * float(carta_prende)
                * float(carta_briscola)
                * float(avversario_sta_prendendo)
                * punti_carta
            ),
            "punti_squadra_avversaria_x_punti_presa_corrente_x_carta_prende_x_briscola_x_avversario_sta_prendendo": (
                (osservazione.punteggio_avversari / PUNTI_TOTALI)
                * punti_presa_norm
                * float(carta_prende)
                * float(carta_briscola)
                * float(avversario_sta_prendendo)
            ),
            "carta_fa_pescare_briscola_esposta_a_squadra_avversaria": float(
                fa_pescare_avversaria
            ),
            "carta_fa_pescare_briscola_esposta_alla_squadra_nostra": float(
                fa_pescare_nostra
            ),
            "carta_fa_pescare_briscola_esposta_alla_squadra_nostra_x_punti_briscola_esposta": (
                float(fa_pescare_nostra) * punti_briscola_esposta
            ),
            "carta_fa_pescare_briscola_esposta_a_squadra_avversaria_x_punti_briscola_esposta": (
                float(fa_pescare_avversaria) * punti_briscola_esposta
            ),
            "carta_fa_pescare_briscola_esposta_alla_squadra_nostra_x_forza_briscola_esposta": (
                float(fa_pescare_nostra) * forza_briscola_esposta
            ),
            "carta_fa_pescare_briscola_esposta_a_squadra_avversaria_x_forza_briscola_esposta": (
                float(fa_pescare_avversaria) * forza_briscola_esposta
            ),
        }

        return [float(values[name]) for name in self.feature_names]

    def _vincitore_corrente(self, osservazione: Osservazione) -> int | None:
        if not osservazione.carte_sul_campo:
            return None
        return vincitore_presa(
            osservazione.carte_sul_campo,
            seme_briscola=osservazione.seme_briscola,
        ).giocatore_id

    def _carta_vincente_tavolo(self, osservazione: Osservazione) -> Carta | None:
        if not osservazione.carte_sul_campo:
            return None
        return vincitore_presa(
            osservazione.carte_sul_campo,
            seme_briscola=osservazione.seme_briscola,
        ).carta

    def _vincitore_dopo_carta(
        self,
        osservazione: Osservazione,
        carta: Carta,
        giocatore_id: int,
    ) -> int:
        presa_candidata = tuple(osservazione.carte_sul_campo) + (
            CartaGiocata(giocatore_id=giocatore_id, carta=carta),
        )
        vincitore = vincitore_presa(
            presa_candidata,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id

    def _carte_osservate(self, osservazione: Osservazione) -> set[Carta]:
        osservate = set(osservazione.mano)
        osservate.update(giocata.carta for giocata in osservazione.carte_sul_campo)
        osservate.update(giocata.carta for giocata in osservazione.carte_giocate)
        osservate.add(osservazione.briscola_esposta)
        if osservazione.mano_compagno_visibile:
            osservate.update(osservazione.mano_compagno)
        return osservate

    def _compagno_puo_prendere(self, osservazione: Osservazione) -> bool:
        if not osservazione.mano_compagno_visibile:
            return False
        return any(
            self._vincitore_dopo_carta(
                osservazione=osservazione,
                carta=carta,
                giocatore_id=osservazione.compagno_id,
            )
            == osservazione.compagno_id
            for carta in osservazione.mano_compagno
        )

    def _superiori_stesso_seme_non_osservate(
        self,
        carta: Carta,
        carte_osservate: set[Carta],
    ) -> int:
        return sum(
            1
            for altra in crea_mazzo()
            if altra.seme == carta.seme
            and altra.forza > carta.forza
            and altra not in carte_osservate
        )

    def _briscole_che_battono_non_osservate(
        self,
        osservazione: Osservazione,
        carta: Carta,
        carte_osservate: set[Carta],
    ) -> int:
        if not self._briscola(osservazione, carta):
            return sum(
                1
                for altra in crea_mazzo()
                if altra.seme == osservazione.seme_briscola
                and altra not in carte_osservate
            )

        return sum(
            1
            for altra in crea_mazzo()
            if altra.seme == osservazione.seme_briscola
            and altra.forza > carta.forza
            and altra not in carte_osservate
        )

    def _briscola_esposta_futura(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> str | None:
        if osservazione.proprietario_briscola_esposta is not None:
            return None
        if osservazione.carte_nel_mazzo != NUMERO_GIOCATORI:
            return None
        if len(osservazione.carte_sul_campo) + 1 != NUMERO_GIOCATORI:
            return None

        vincitore = self._vincitore_dopo_carta(
            osservazione=osservazione,
            carta=carta,
            giocatore_id=osservazione.giocatore_id,
        )
        proprietario_futuro = ordine_giocatori_da(vincitore)[-1]
        if proprietario_futuro in (osservazione.giocatore_id, osservazione.compagno_id):
            return "nostra"
        if proprietario_futuro in osservazione.avversari:
            return "avversaria"
        return None

    def _briscola(self, osservazione: Osservazione, carta: Carta | None) -> bool:
        return carta is not None and carta.seme == osservazione.seme_briscola

    def _carico(self, carta: Carta | None) -> bool:
        return carta is not None and carta.rango in CARICHI

    def _carico_non_briscola(
        self,
        osservazione: Osservazione,
        carta: Carta | None,
    ) -> bool:
        return self._carico(carta) and not self._briscola(osservazione, carta)

    def _liscio(self, osservazione: Osservazione, carta: Carta | None) -> bool:
        return carta is not None and carta.punti == 0 and not self._briscola(
            osservazione,
            carta,
        )

    def _punticino(self, osservazione: Osservazione, carta: Carta | None) -> bool:
        return (
            carta is not None
            and carta.punti in {2, 3, 4}
            and not self._briscola(osservazione, carta)
        )

    def _taglietto(self, osservazione: Osservazione, carta: Carta | None) -> bool:
        return carta is not None and carta.punti == 0 and self._briscola(
            osservazione,
            carta,
        )

    def _briscola_alta(self, osservazione: Osservazione, carta: Carta | None) -> bool:
        return carta is not None and carta.punti > 0 and self._briscola(
            osservazione,
            carta,
        )
