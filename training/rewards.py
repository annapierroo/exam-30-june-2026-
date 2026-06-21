"""Reward helpers for Briscola training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PUNTI_TOTALI_PARTITA = 120
REWARD_MODES = {"combined_terminal", "dense_presa"}

RewardMode = Literal["combined_terminal", "dense_presa"]


@dataclass(frozen=True)
class RewardConfig:
    """Configurazione esplicita della reward di training."""

    mode: RewardMode = "combined_terminal"
    alpha: float = 1.0
    lambda_margin: float = 0.2

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(f"Reward mode non supportata: {self.mode}")
        if self.alpha < 0.0:
            raise ValueError("alpha deve essere non negativo")
        if self.lambda_margin < 0.0:
            raise ValueError("lambda_margin deve essere non negativo")


def calcola_margine(punti_squadra: int, punti_avversari: int) -> int:
    """Calcola il margine dal punto di vista della squadra del learner."""

    return punti_squadra - punti_avversari


def calcola_segno(margine: int) -> float:
    """Codifica vittoria, sconfitta e pareggio in un segno numerico."""

    if margine > 0:
        return 1.0
    if margine < 0:
        return -1.0
    return 0.0


def normalizza_margine(margine: int) -> float:
    """Porta un margine punti sulla scala della partita completa."""

    return float(margine) / PUNTI_TOTALI_PARTITA


def reward_finale(
    punti_squadra: int,
    punti_avversari: int,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Calcola la reward assegnata a fine partita."""

    margine = calcola_margine(punti_squadra, punti_avversari)
    segno = calcola_segno(margine)

    if config.mode == "combined_terminal":
        return float(
            config.alpha * segno
            + config.lambda_margin * normalizza_margine(margine)
        )
    if config.mode == "dense_presa":
        return float(config.alpha * segno)

    raise ValueError(f"Reward mode non supportata: {config.mode}")


def reward_presa(
    punti_presa: int,
    presa_vinta_da_squadra: bool,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Calcola la reward immediata quando una presa viene completata."""

    if config.mode == "combined_terminal":
        return 0.0

    if config.mode == "dense_presa":
        segno = 1.0 if presa_vinta_da_squadra else -1.0
        return float(
            config.lambda_margin * segno * (punti_presa / PUNTI_TOTALI_PARTITA)
        )

    raise ValueError(f"Reward mode non supportata: {config.mode}")
