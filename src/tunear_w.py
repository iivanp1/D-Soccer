"""Tuneo empirico de w (peso Elo vs jugadores en el hibrido) sobre la muestra del harvest.

Por que existe: w=0.65 es el ULTIMO prior sin convertir en dato. harvest.py validaba el
modelo bottom-up PURO (nunca cargaba el Elo), asi que nunca se midio en muestra grande el
hibrido que de verdad corre en produccion. Este script carga el Elo y barre w en [0,1]
buscando el que minimiza el Brier sobre los partidos reales ya cosechados.

Como funciona:
  - Reusa la maquinaria del harvest: ratings de clubes PERIODO-CORRECTOS (season 2324 para
    validar 2024, sin fuga) + fixtures/alineaciones REALES cacheados.
  - SOLO procesa partidos cuyas alineaciones YA estan en data/raw/api_cache/ -> CERO
    llamadas nuevas a la API (no gasta cuota). Para ampliar la muestra, primero correr
    harvest (python -m src.harvest <liga> 2024 <limite>) y volver a correr esto.
  - Para cada w del grid, recalcula la prediccion del hibrido y promedia el Brier.
  - Reporta la curva Brier(w), el w optimo in-sample y una validacion leave-one-out (LOO)
    que es mas honesta con muestras chicas (mide el w elegido en partidos que no vio).

CAVEAT (leerlo): el Elo que se carga es el ACTUAL (2026), aplicado a partidos de 2024 ->
leve fuga de futuro por ese lado (el ranking Elo es estable, pero no es period-correcto).
Con la muestra cacheada actual (~8 partidos) esto es SENAL PRELIMINAR, no veredicto.
Proximo salto: ampliar muestra (Copa America liga 9, amistosos liga 10) + Elo period-correcto.

Uso:
    python -m src.tunear_w                 # usa lo cacheado (default: Euro 2024)
    python -m src.tunear_w 4 9 10          # ligas a incluir (solo partidos ya cacheados)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src import config
from src.backtest import brier_score, resultado_real
from src.harvest import CACHE, _fixtures, _lineups, _modelo_periodo_correcto

GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
W_ACTUAL = 0.65


def _cargar_elo() -> dict:
    """Mismo criterio que mundial_engine: prefiere el Elo propio calibrado."""
    propio = config.DATA_PROC / "elo_propio.json"
    if propio.exists():
        return json.loads(propio.read_text(encoding="utf-8"))
    return json.loads((config.DATA_PROC / "elo.json").read_text(encoding="utf-8"))


def _muestra(ligas: list[int], season: int = 2024) -> list[dict]:
    """Arma la muestra de validacion SOLO con partidos cuyas alineaciones ya estan en cache.

    Devuelve, por partido: los XI reales emparejados, los codigos de nacion y el resultado.
    Cero llamadas nuevas a la API (skipea cualquier fixture sin lineup cacheado).
    """
    from src.fixtures import _codigo_nacion, armar_xi

    _, dfj = _modelo_periodo_correcto()
    filas = []
    for liga in ligas:
        for f in _fixtures(liga, season):  # fixtures_{liga}_{season}.json ya cacheado
            fid = f["fixture"]["id"]
            if not (CACHE / f"lineups_{fid}.json").exists():
                continue  # sin alineacion cacheada -> NO pulleamos (no gastar cuota)
            gl, gv = f["goals"]["home"], f["goals"]["away"]
            if gl is None or gv is None:
                continue
            nl, nv = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
            cl, cv = _codigo_nacion(nl), _codigo_nacion(nv)
            if not cl or not cv:
                continue
            xi_l = xi_v = None
            for eq in _lineups(fid):
                cod = _codigo_nacion(eq["team"]["name"])
                nombres = [p["player"]["name"] for p in eq.get("startXI", [])]
                r = armar_xi(nombres, cod, dfj)
                if cod == cl:
                    xi_l = r["xi_real"]
                elif cod == cv:
                    xi_v = r["xi_real"]
            filas.append({"local": nl, "visitante": nv, "cl": cl, "cv": cv,
                          "xi_l": xi_l, "xi_v": xi_v, "real": resultado_real(gl, gv)})
    return filas


def _brier_en(jm, muestra: list[dict], w: float) -> float:
    """Brier medio del hibrido con un w dado, sobre la muestra."""
    jm.w_elo = w
    bs = []
    for m in muestra:
        p = jm.predecir_partido_mundial(m["xi_l"], m["xi_v"], m["cl"], m["cv"])
        bs.append(brier_score(p["prob_local"], p["prob_empate"], p["prob_visitante"], m["real"]))
    return float(np.mean(bs))


def tunear(ligas: list[int]) -> None:
    jm, _ = _modelo_periodo_correcto()
    jm.cargar_elo(_cargar_elo())
    muestra = _muestra(ligas)

    if not muestra:
        print("No hay partidos cacheados para esas ligas. Corre antes:")
        print("   python -m src.harvest 4 2024 8   (Eurocopa)   /   9 (Copa America)   /   10 (amistosos)")
        return

    # Cuantas selecciones tienen Elo (las que no, caen a jugadores puro y w no las afecta)
    con_elo = sum(1 for m in muestra if m["cl"] in jm.elo and m["cv"] in jm.elo)

    print("=" * 58)
    print(f"  TUNEO DE w (Elo vs jugadores)  |  {len(muestra)} partidos cacheados")
    print(f"  ligas {ligas} 2024  |  {con_elo} con Elo en ambas selecciones")
    print("  CAVEAT: Elo actual sobre 2024 (leve fuga) + muestra chica = senal preliminar")
    print("=" * 58)

    curva = [(w, _brier_en(jm, muestra, w)) for w in GRID]
    w_opt, b_opt = min(curva, key=lambda t: t[1])
    b_actual = dict(curva).get(W_ACTUAL) or _brier_en(jm, muestra, W_ACTUAL)
    b_player = dict(curva)[0.0]
    b_elo = dict(curva)[1.0]

    print(f"\n  {'w':>5}   {'Brier':>7}")
    for w, b in curva:
        marca = ""
        if w == w_opt:
            marca = "  <- optimo"
        elif abs(w - W_ACTUAL) < 1e-9:
            marca = "  <- actual"
        elif w == 0.0:
            marca = "  (jugadores puro)"
        elif w == 1.0:
            marca = "  (Elo puro)"
        print(f"  {w:>5.2f}   {b:>7.4f}{marca}")

    print(f"\n  w optimo (in-sample) : {w_opt:.2f}  (Brier {b_opt:.4f})")
    print(f"  w actual (0.65)      : {b_actual:.4f}")
    print(f"  jugadores puro (w=0) : {b_player:.4f}")
    print(f"  Elo puro (w=1)       : {b_elo:.4f}")

    # Leave-one-out: para cada partido, elige el mejor w SIN ese partido y lo evalua en el.
    # Mas honesto con n chico (mide generalizacion, no ajuste in-sample).
    loo = []
    for i in range(len(muestra)):
        resto = muestra[:i] + muestra[i + 1:]
        w_i = min(GRID, key=lambda w: _brier_en(jm, resto, w))
        loo.append(_brier_en(jm, [muestra[i]], w_i))
    print(f"\n  Brier LOO (w elegido fuera de muestra): {np.mean(loo):.4f}")
    print(f"  -> {'el tuneo generaliza' if np.mean(loo) < b_actual else 'NO supera al w=0.65 actual (muestra insuficiente)'}")
    print(f"\n  (Con <15-20 partidos esto es orientativo. Ampliar muestra antes de fijar w.)")


def main() -> None:
    ligas = [int(x) for x in sys.argv[1:]] or [4]
    tunear(ligas)


if __name__ == "__main__":
    main()
