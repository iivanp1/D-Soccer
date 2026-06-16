"""Compuerta de validacion del modelo de Player Props.

Metodo: por cada titular en StatsBomb (2024 + AFCON23), usa el XI real mas el
equipo-λ OBSERVADO (no proyectado) para aislar el error del modelo de distribucion
del error del engine. Si el Brier del modelo < Brier baseline → OK para deployar.

Calibracion: distribucion de probabilidades. Si el modelo dice P=70% para algo, ¿ocurre
el 70% de las veces? Se mide en buckets de 10pp.

Uso:
    python -m src.validar_props              # valida 2024 + AFCON23
    python -m src.validar_props --solo-2024
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd

from src import config
from src.fixtures import armar_xi
from src.jugadores_model import _norm
from src.props_data import cargar as cargar_tiros
from src.props_model import calcular_props_equipo
from src.validar_statsbomb import DB, SEASONS_SB, _cod, _modelo, _titulares


def _muestra_props(con: sqlite3.Connection, dfj: pd.DataFrame, solo_2024: bool) -> list[dict]:
    """Construye la muestra de validacion: (jugador, lam_model, tiros_reales)."""
    seasons = ("2024",) if solo_2024 else SEASONS_SB
    ph = ",".join("?" * len(seasons))
    datos_tiros = cargar_tiros()
    if not datos_tiros:
        print("Falta tiros_intl.json. Corre: python -m src.props_data")
        return []

    filas = []
    rows = con.execute(
        f"SELECT match_id, equipo_local, equipo_visitante, "
        f"goles_local, goles_visitante FROM partidos WHERE season IN ({ph})",
        seasons).fetchall()

    for mid, local, visit, gl, gv in rows:
        for equipo, goles in ((local, gl), (visit, gv)):
            cod = _cod(equipo)
            if not cod or goles is None:
                continue
            xi_names = armar_xi(_titulares(con, mid, equipo), cod, dfj)["xi_real"]
            if not xi_names:
                continue

            # lambda equipo OBSERVADO (no proyectado): goles reales / conv_rate
            # Esto aísla el modelo de distribución del error del engine.
            conv = datos_tiros["meta"]["conversion_rate_intl"]
            lam_goles_obs = max(0.5, float(goles))  # usar goles reales como proxy
            props = calcular_props_equipo(xi_names, cod, lam_goles_obs, dfj, datos_tiros)

            # Stats reales por jugador (de la DB)
            stats_reales = {
                row[0]: row[1]  # player_norm -> remates
                for row in con.execute(
                    "SELECT j.player_norm, j.remates FROM jugador_partido_stats j "
                    "JOIN alineaciones a ON a.match_id=j.match_id AND a.player_id=j.player_id "
                    "AND a.es_titular=1 WHERE j.match_id=? AND j.equipo=?",
                    (mid, equipo)).fetchall()
            }

            for nombre, v in props.items():
                nrm = _norm(nombre)
                real_remates = stats_reales.get(nrm)
                if real_remates is None:
                    # intenta sin acento (normalizado)
                    real_remates = next((vv for kk, vv in stats_reales.items()
                                         if kk == nrm), None)
                if real_remates is None:
                    continue
                real_remates = int(real_remates)
                filas.append({
                    "nombre": nombre,
                    "fuente": v["fuente"],
                    "lam": v["lam"],
                    "p_over_0_5": v["p_over_0_5"],
                    "p_over_1_5": v["p_over_1_5"],
                    "real_tiros": real_remates,
                    "real_over_0_5": int(real_remates >= 1),
                    "real_over_1_5": int(real_remates >= 2),
                })
    return filas


def validar(solo_2024: bool = False) -> None:
    jm, dfj = _modelo()
    con = sqlite3.connect(DB)
    muestra = _muestra_props(con, dfj, solo_2024)
    con.close()

    if not muestra:
        print("Muestra vacia. Revisa la DB y tiros_intl.json.")
        return

    n = len(muestra)
    tag = "solo 2024" if solo_2024 else "2024 + AFCON23"

    # --- Brier score para Over 1.5 tiros ---------------------------------- #
    p15 = np.array([m["p_over_1_5"] for m in muestra])
    y15 = np.array([m["real_over_1_5"] for m in muestra])
    brier_mod = float(np.mean((p15 - y15) ** 2))
    brier_base = float(np.mean((y15.mean() - y15) ** 2))  # baseline: predecir la media
    mejora = (brier_base - brier_mod) / brier_base * 100

    # Separar por fuente
    idx_intl = [i for i, m in enumerate(muestra) if m["fuente"] == "intl"]
    idx_club = [i for i, m in enumerate(muestra) if m["fuente"] == "club"]

    brier_intl = float(np.mean((p15[idx_intl] - y15[idx_intl]) ** 2)) if idx_intl else float("nan")
    brier_club = float(np.mean((p15[idx_club] - y15[idx_club]) ** 2)) if idx_club else float("nan")

    print("=" * 66)
    print(f"  VALIDACION PLAYER PROPS (StatsBomb) | {n} jugador-partido ({tag})")
    print("=" * 66)
    print(f"  Base rate Over 1.5 tiros: {y15.mean()*100:.1f}%")
    print(f"  {'fuente':<12} {'n':>6}  {'Brier':>8}  {'vs baseline':>12}")
    print(f"  {'GLOBAL':<12} {n:>6}  {brier_mod:>8.4f}  {mejora:>+10.1f}%")
    if idx_intl:
        b_i_base = float(np.mean((y15[idx_intl] - y15[idx_intl].mean()) ** 2))
        mejora_i = (b_i_base - brier_intl) / b_i_base * 100 if b_i_base > 0 else 0.0
        print(f"  {'intl':<12} {len(idx_intl):>6}  {brier_intl:>8.4f}  {mejora_i:>+10.1f}%")
    if idx_club:
        b_c_base = float(np.mean((y15[idx_club] - y15[idx_club].mean()) ** 2))
        mejora_c = (b_c_base - brier_club) / b_c_base * 100 if b_c_base > 0 else 0.0
        print(f"  {'club':<12} {len(idx_club):>6}  {brier_club:>8.4f}  {mejora_c:>+10.1f}%")

    # --- Calibracion (precision por bucket de probabilidad) --------------- #
    print(f"\n  Calibracion (Over 1.5 tiros) — ¿cuando dice X%, ocurre X%?")
    print(f"  {'bucket':>12}  {'n':>5}  {'pred_med':>9}  {'real':>7}  {'error':>7}")
    edges = np.arange(0, 1.1, 0.1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p15 >= lo) & (p15 < hi)
        n_b = mask.sum()
        if n_b < 5:
            continue
        pred_med = float(p15[mask].mean())
        real_med = float(y15[mask].mean())
        err = real_med - pred_med
        print(f"  {lo*100:.0f}-{hi*100:.0f}%      {n_b:>5}  {pred_med*100:>8.1f}%  "
              f"{real_med*100:>6.1f}%  {err*100:>+6.1f}pp")

    # --- Compuerta --------------------------------------------------------- #
    print()
    if mejora > 0:
        print(f"  COMPUERTA: PASADA (Brier modelo {brier_mod:.4f} < baseline {brier_base:.4f})")
        print(f"  DECISION: OK para deployar en produccion.")
    else:
        print(f"  COMPUERTA: NO PASADA (Brier modelo {brier_mod:.4f} >= baseline {brier_base:.4f})")
        print(f"  DECISION: NO deployar. Revisar fuentes de datos o escala.")


def main() -> None:
    validar(solo_2024="--solo-2024" in sys.argv)


if __name__ == "__main__":
    main()
