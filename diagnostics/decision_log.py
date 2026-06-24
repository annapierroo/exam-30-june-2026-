"""Decision logs for frozen Briscola policy games."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta, CartaGiocata
from game.environment import Ambiente, CartaPescata, EsitoMossa, EventoPubblico
from game.observation import Osservazione
from game.rules import NUMERO_GIOCATORI, valida_giocatore_id
from policy import Policy


@dataclass(frozen=True)
class DecisionOutcome:
    """Public outcome immediately after a recorded decision."""

    partita_finita: bool
    presa_completata: bool
    carte_presa_completata: tuple[CartaGiocata, ...]
    vincitore_presa: int | None
    punti_presa: int
    carte_pescate: tuple[CartaPescata, ...]
    prossimo_giocatore: int | None
    punteggi: dict[str, int]
    eventi_pubblici: tuple[EventoPubblico, ...]


@dataclass(frozen=True)
class DecisionRecord:
    """One legal observation, chosen action, and public outcome."""

    step_index: int
    giocatore_id: int
    focus_giocatore_id: int | None
    focus_mano: tuple[Carta, ...]
    mani_by_player: dict[int, tuple[Carta, ...]]
    policy_name: str
    osservazione: Osservazione
    azioni_legali: tuple[Carta, ...]
    action_probabilities: dict[Carta, float]
    azione: Carta
    greedy: bool
    outcome: DecisionOutcome


@dataclass(frozen=True)
class DecisionLog:
    """Complete diagnostic log for one played game."""

    seed_ambiente: int
    seed_policy: int
    primo_giocatore_id: int
    greedy: bool
    records: tuple[DecisionRecord, ...]
    punteggi_finali: dict[str, int]
    squadra_vincitrice: str | None


def decision_log_to_dict(log: DecisionLog) -> dict:
    """Serialize a decision log into plain JSON-compatible values."""

    return {
        "seed_ambiente": log.seed_ambiente,
        "seed_policy": log.seed_policy,
        "primo_giocatore_id": log.primo_giocatore_id,
        "greedy": log.greedy,
        "punteggi_finali": dict(log.punteggi_finali),
        "squadra_vincitrice": log.squadra_vincitrice,
        "records": [_record_to_dict(record) for record in log.records],
    }


def record_decision_log(
    *,
    policies_by_player: dict[int, Policy],
    seed_ambiente: int,
    seed_policy: int,
    primo_giocatore_id: int,
    greedy: bool,
    focus_giocatore_id: int | None = None,
) -> DecisionLog:
    """Play one game and record only legal observations plus public outcomes."""

    _valida_policies_by_player(policies_by_player)
    valida_giocatore_id(primo_giocatore_id)
    if focus_giocatore_id is not None:
        valida_giocatore_id(focus_giocatore_id)

    ambiente = Ambiente(
        seed=seed_ambiente,
        primo_giocatore_id=primo_giocatore_id,
    )
    rng_policy = random.Random(seed_policy)
    records: list[DecisionRecord] = []

    while not ambiente.finita:
        step_index = len(records)
        giocatore_id = ambiente.giocatore_corrente
        osservazione = ambiente.osserva(giocatore_id)
        azioni_legali = osservazione.azioni_legali
        policy = policies_by_player[giocatore_id]
        action_probabilities = policy.action_probabilities(osservazione)
        _valida_probabilities(action_probabilities, azioni_legali)

        azione = policy.select_action(osservazione, rng_policy, greedy=greedy)
        if azione not in azioni_legali:
            raise ValueError("La policy ha scelto una carta non legale")

        focus_mano = (
            tuple(ambiente.mani[focus_giocatore_id])
            if focus_giocatore_id is not None
            else ()
        )
        mani_by_player = {
            player_id: tuple(mano)
            for player_id, mano in enumerate(ambiente.mani)
        }
        esito = ambiente.gioca(azione)
        records.append(
            DecisionRecord(
                step_index=step_index,
                giocatore_id=giocatore_id,
                focus_giocatore_id=focus_giocatore_id,
                focus_mano=focus_mano,
                mani_by_player=mani_by_player,
                policy_name=policy.name,
                osservazione=osservazione,
                azioni_legali=azioni_legali,
                action_probabilities=dict(action_probabilities),
                azione=azione,
                greedy=greedy,
                outcome=_decision_outcome(esito),
            )
        )

    ambiente.verifica_integrita_stato()

    return DecisionLog(
        seed_ambiente=seed_ambiente,
        seed_policy=seed_policy,
        primo_giocatore_id=primo_giocatore_id,
        greedy=greedy,
        records=tuple(records),
        punteggi_finali=dict(ambiente.punteggi),
        squadra_vincitrice=ambiente.squadra_vincitrice(),
    )


def _decision_outcome(esito: EsitoMossa) -> DecisionOutcome:
    return DecisionOutcome(
        partita_finita=esito.partita_finita,
        presa_completata=esito.presa_completata,
        carte_presa_completata=esito.carte_presa_completata,
        vincitore_presa=esito.vincitore_presa,
        punti_presa=esito.punti_presa,
        carte_pescate=esito.carte_pescate,
        prossimo_giocatore=esito.prossimo_giocatore,
        punteggi=dict(esito.punteggi),
        eventi_pubblici=esito.eventi_pubblici,
    )


def _record_to_dict(record: DecisionRecord) -> dict:
    osservazione = record.osservazione
    return {
        "step_index": record.step_index,
        "giocatore_id": record.giocatore_id,
        "focus_giocatore_id": record.focus_giocatore_id,
        "focus_mano": [_card_to_dict(carta) for carta in record.focus_mano],
        "mani_by_player": {
            str(player_id): [_card_to_dict(carta) for carta in mano]
            for player_id, mano in record.mani_by_player.items()
        },
        "policy_name": record.policy_name,
        "greedy": record.greedy,
        "mano": [_card_to_dict(carta) for carta in osservazione.mano],
        "seme_briscola": osservazione.seme_briscola,
        "briscola_esposta": _card_to_dict(osservazione.briscola_esposta),
        "carte_sul_campo": [
            _played_card_to_dict(giocata)
            for giocata in osservazione.carte_sul_campo
        ],
        "azioni_legali": [_card_to_dict(carta) for carta in record.azioni_legali],
        "azione": _card_to_dict(record.azione),
        "action_probabilities": {
            carta.id: probability
            for carta, probability in record.action_probabilities.items()
        },
        "osservazione": _observation_to_dict(osservazione),
        "outcome": _outcome_to_dict(record.outcome),
    }


def _observation_to_dict(osservazione: Osservazione) -> dict:
    return {
        "giocatore_id": osservazione.giocatore_id,
        "compagno_id": osservazione.compagno_id,
        "avversari": list(osservazione.avversari),
        "squadra": osservazione.squadra,
        "squadra_avversaria": osservazione.squadra_avversaria,
        "punteggio_squadra": osservazione.punteggio_squadra,
        "punteggio_avversari": osservazione.punteggio_avversari,
        "mano": [_card_to_dict(carta) for carta in osservazione.mano],
        "mano_compagno_visibile": osservazione.mano_compagno_visibile,
        "mano_compagno": [
            _card_to_dict(carta) for carta in osservazione.mano_compagno
        ],
        "seme_briscola": osservazione.seme_briscola,
        "briscola_esposta": _card_to_dict(osservazione.briscola_esposta),
        "proprietario_briscola_esposta": (
            osservazione.proprietario_briscola_esposta
        ),
        "carte_sul_campo": [
            _played_card_to_dict(giocata)
            for giocata in osservazione.carte_sul_campo
        ],
        "carte_giocate": [
            _played_card_to_dict(giocata)
            for giocata in osservazione.carte_giocate
        ],
        "vincitori_prese": list(osservazione.vincitori_prese),
        "primo_giocatore_presa": osservazione.primo_giocatore_presa,
        "giocatore_corrente": osservazione.giocatore_corrente,
        "carte_nel_mazzo": osservazione.carte_nel_mazzo,
        "indice_presa": osservazione.indice_presa,
        "posizione_nella_presa": osservazione.posizione_nella_presa,
    }


def _outcome_to_dict(outcome: DecisionOutcome) -> dict:
    return {
        "partita_finita": outcome.partita_finita,
        "presa_completata": outcome.presa_completata,
        "carte_presa_completata": [
            _played_card_to_dict(giocata)
            for giocata in outcome.carte_presa_completata
        ],
        "vincitore_presa": outcome.vincitore_presa,
        "punti_presa": outcome.punti_presa,
        "carte_pescate": [
            _drawn_card_to_dict(carta_pescata)
            for carta_pescata in outcome.carte_pescate
        ],
        "prossimo_giocatore": outcome.prossimo_giocatore,
        "punteggi": dict(outcome.punteggi),
        "eventi_pubblici": [
            _public_event_to_dict(evento) for evento in outcome.eventi_pubblici
        ],
    }


def _card_to_dict(carta: Carta | None) -> dict | None:
    if carta is None:
        return None
    return {
        "id": carta.id,
        "seme": carta.seme,
        "rango": carta.rango,
        "punti": carta.punti,
        "forza": carta.forza,
    }


def _played_card_to_dict(giocata: CartaGiocata) -> dict:
    return {
        "giocatore_id": giocata.giocatore_id,
        "carta": _card_to_dict(giocata.carta),
    }


def _drawn_card_to_dict(carta_pescata: CartaPescata) -> dict:
    return {
        "giocatore_id": carta_pescata.giocatore_id,
        "carta_visibile": _card_to_dict(carta_pescata.carta_visibile),
    }


def _public_event_to_dict(evento: EventoPubblico) -> dict:
    return {
        "tipo": evento.tipo,
        "giocatore_id": evento.giocatore_id,
        "carta": _card_to_dict(evento.carta),
        "punti": evento.punti,
    }


def _valida_policies_by_player(policies_by_player: dict[int, Policy]) -> None:
    expected_players = set(range(NUMERO_GIOCATORI))
    actual_players = set(policies_by_player)
    if actual_players != expected_players:
        raise ValueError(
            "policies_by_player deve contenere esattamente i giocatori "
            f"{sorted(expected_players)}, ottenuto {sorted(actual_players)}"
        )
    for giocatore_id in policies_by_player:
        valida_giocatore_id(giocatore_id)


def _valida_probabilities(
    action_probabilities: dict[Carta, float],
    azioni_legali: tuple[Carta, ...],
) -> None:
    if set(action_probabilities) != set(azioni_legali):
        raise ValueError(
            "action_probabilities deve avere esattamente le azioni legali"
        )
