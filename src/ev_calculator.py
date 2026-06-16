"""Calculador de Valor Esperado (EV).

EV = (Probabilidad * Cuota) - 1
  > 0  : apuesta con valor (a la larga, ganadora)
  <= 0 : sin valor (la casa cubre la probabilidad)

"Donde si y donde no" es esta formula, no una IA. Determinista y transparente.
"""

from __future__ import annotations


def calcular_ev(probabilidad: float, cuota: float) -> tuple[float, bool]:
    """Devuelve (ev_porcentual, es_apuesta_de_valor).

    probabilidad: prob del modelo (0..1). cuota: cuota decimal ofrecida por la casa.
    """
    ev = probabilidad * cuota - 1.0
    return ev * 100.0, ev > 0.0
