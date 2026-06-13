"""Valida los MERCADOS SECUNDARIOS del modelo de jugadores (faltas/tarjetas) sobre StatsBomb.

Es el pago del pivote estrategico: en 1X2 el Elo le gana al modelo de jugadores (validar_statsbomb),
pero el edge del player model deberia estar en los mercados que el Elo NO toca: faltas y tarjetas
(via disciplina_seleccion, que suma las tasas por-90 del XI). Aca se mide contra los datos REALES
cosechados de StatsBomb (tabla equipo_partido_stats).

Metodo: por cada equipo-partido, se cruza el XI titular con jugadores.csv (armar_xi), se predice
disciplina_seleccion(xi, nacion) -> faltas/tarjetas, y se compara con lo real. Se reporta:
  - MAE del modelo vs MAE del baseline (predecir la media) -> ¿aporta en valor absoluto?
  - Correlacion pred-real -> ¿RANKEA bien (capta que tal equipo comete mas faltas) aunque la
    escala club!=internacional este corrida? (la escala se puede recalibrar; la senal no).

CAVEAT: el factor arbitro NO se aplica (los arbitros internacionales casi no estan en datos de
clubes -> neutro). Se valida el mecanismo de jugadores puro. Necesita la DB con la tabla
equipo_partido_stats (python -m src.ingesta_historica --desde 2023 --rehacer).

Uso:
    python -m src.validar_estilos_statsbomb              # 2024 + AFCON23
    python -m src.validar_estilos_statsbomb --solo-2024
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np

from src.fixtures import armar_xi
from src.validar_statsbomb import DB, SEASONS_SB, _cod, _modelo, _titulares


def _muestra(con, dfj, solo_2024: bool) -> list[dict]:
    seasons = ("2024",) if solo_2024 else SEASONS_SB
    ph = ",".join("?" * len(seasons))
    q = f"SELECT match_id, equipo_local, equipo_visitante FROM partidos WHERE season IN ({ph})"
    filas = []
    for mid, local, visit in con.execute(q, seasons):
        for equipo in (local, visit):
            cod = _cod(equipo)
            if not cod:
                continue
            real = con.execute(
                "SELECT faltas, tarjetas, corners FROM equipo_partido_stats WHERE match_id=? AND equipo=?",
                (mid, equipo)).fetchone()
            if not real or real[0] is None:
                continue
            xi = armar_xi(_titulares(con, mid, equipo), cod, dfj)["xi_real"]  # nombres del modelo
            filas.append({"cod": cod, "xi": xi, "faltas": real[0],
                          "tarjetas": real[1], "corners": real[2]})
    return filas


def validar(solo_2024: bool = False) -> None:
    jm, dfj = _modelo()
    con = sqlite3.connect(DB)
    muestra = _muestra(con, dfj, solo_2024)
    con.close()
    if not muestra:
        print("Muestra vacia. Corre antes: python -m src.ingesta_historica --desde 2023 --rehacer")
        return

    pf, pt, rf, rt, rc = [], [], [], [], []
    for m in muestra:
        d = jm.disciplina_seleccion(m["xi"], m["cod"])
        pf.append(d["faltas"]); pt.append(d["tarjetas"])
        rf.append(m["faltas"]); rt.append(m["tarjetas"]); rc.append(m["corners"])
    pf, pt, rf, rt, rc = map(np.array, (pf, pt, rf, rt, rc))

    print("=" * 66)
    print(f"  VALIDACION MERCADOS SECUNDARIOS (StatsBomb) | {len(muestra)} equipos-partido "
          f"({'solo 2024' if solo_2024 else '2024 + AFCON23'})")
    print("=" * 66)
    print(f"  {'mercado':<10}{'real_med':>9}{'mod_med':>9}{'MAE_mod':>9}{'MAE_base':>10}{'mejora':>9}{'corr':>7}")
    for nombre, pred, real in [("Faltas", pf, rf), ("Tarjetas", pt, rt)]:
        mae_m = np.mean(np.abs(pred - real))
        mae_b = np.mean(np.abs(real - real.mean()))
        mejora = (mae_b - mae_m) / mae_b * 100
        corr = np.corrcoef(pred, real)[0, 1]
        print(f"  {nombre:<10}{real.mean():>9.1f}{pred.mean():>9.1f}{mae_m:>9.2f}{mae_b:>10.2f}"
              f"{mejora:>8.1f}%{corr:>7.2f}")
    print(f"\n  Corners (real medio {rc.mean():.1f}): el modelo de jugadores NO las predice -> aparte.")
    print("  mejora>0 = el modelo le gana al baseline (predecir la media).")
    print("  corr alto = RANKEA bien aunque la escala club!=intl este corrida (recalibrable).")
    print("  (Arbitro neutro: los intl casi no estan en datos de clubes. Tarjetas: techo bajo.)")


def main() -> None:
    validar(solo_2024="--solo-2024" in sys.argv)


if __name__ == "__main__":
    main()
