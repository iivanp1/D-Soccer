"""Backtest COMPARATIVO: Dixon-Coles clasico (goles) vs Dixon-Coles-xG, vs Pinnacle.

La pregunta de la fase 4: ¿entrenar con xG (menos ruido de finiquito) alcanza para
batir al mercado donde el modelo de goles ya demostro que NO (backtest_roi: ROI -11 a
-14%, CLV -4%)? La literatura (Wilkens 2026, Bundesliga) dice que puede; aca se MIDE.

Metodologia (identica a backtest_roi, PAREADA para que la comparacion sea justa):
  - Walk-forward semanal sobre la temporada test (2425), 4 ligas (las que tienen xG).
  - Cada ventana entrena AMBOS modelos solo con partidos anteriores:
      clasico: goles reales de football-data;  xG: partidos_xg (Understat) mapeado a
      nombres football-data, con rho re-estimado sobre goles enteros (dixon_coles_xg).
  - Solo se evaluan partidos donde AMBOS modelos pueden predecir (mismo universo).
  - Senal: p_modelo - p_sharp_demarg(Pinnacle pre) > umbral; ejecucion best(B365, PS);
    CLV contra el CIERRE de Pinnacle (PSC*). Stake plano 1u.
  - Desglose por MERCADO (Local/Empate/Visitante) para ver donde vive el rendimiento.

Uso:
    python -m src.backtest_xg              # test 2425, reentreno semanal, umbral 4%
    python -m src.backtest_xg 2425 7
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import timedelta

import pandas as pd

from src import config
from src.backtest import resultado_real
from src.backtest_roi import cargar_con_cuotas
from src.dixon_coles import DixonColes
from src.dixon_coles_xg import DixonColesXG
from src.estadisticas_detalladas import LIGA_A_FD, _mapa_fd
from src.valor import _demargin

DB = config.DATA_PROC / "dsoccer_clubes.db"
TEMPORADA_TEST = "2425"
DIAS = 7
UMBRALES = [0.02, 0.04, 0.06]
LIGAS_FD = list(LIGA_A_FD.values())  # SP1, E0, D1, I1 (las 4 con xG)


def cargar_xg_mapeado() -> tuple[pd.DataFrame, int]:
    """partidos_xg con nombres TRADUCIDOS a football-data (para que ambos modelos hablen
    el mismo idioma de equipos). Devuelve (df, n_sin_mapeo)."""
    con = sqlite3.connect(DB)
    xg = pd.read_sql_query(
        "SELECT liga, temporada, fecha, nombre_local, nombre_visitante, "
        "xg_local, xg_visitante FROM partidos_xg", con)
    con.close()
    mapa = _mapa_fd()
    xg["local"] = [mapa.get((l, n)) for l, n in zip(xg["liga"], xg["nombre_local"])]
    xg["visitante"] = [mapa.get((l, n)) for l, n in zip(xg["liga"], xg["nombre_visitante"])]
    sin_mapeo = int(xg["local"].isna().sum() + xg["visitante"].isna().sum())
    xg = xg.dropna(subset=["local", "visitante"]).copy()
    xg["fecha"] = pd.to_datetime(xg["fecha"])
    # La interfaz de DixonColes espera goles_*: aca van los xG (quasi-Poisson continuo)
    xg["goles_local"] = xg["xg_local"]
    xg["goles_visitante"] = xg["xg_visitante"]
    return xg.sort_values("fecha").reset_index(drop=True), sin_mapeo


def correr(temporada_test: str = TEMPORADA_TEST, dias: int = DIAS) -> pd.DataFrame:
    fd = cargar_con_cuotas()
    fd = fd[fd["liga"].isin(LIGAS_FD)].reset_index(drop=True)
    xg, sin_mapeo = cargar_xg_mapeado()
    test = fd[fd["temporada"] == temporada_test]
    print(f"football-data 4 ligas: {len(fd)} | xG mapeado: {len(xg)} partidos "
          f"({sin_mapeo} lados sin mapeo) | test {temporada_test}: {len(test)}")
    print(f"Walk-forward semanal con DOS modelos por ventana. Tarda varios minutos...\n")

    filas = []
    ventana, fin = test["fecha"].min(), test["fecha"].max()
    while ventana <= fin:
        tope = ventana + timedelta(days=dias)
        lote = test[(test["fecha"] >= ventana) & (test["fecha"] < tope)]
        if lote.empty:
            ventana = tope
            continue
        fd_train = fd[fd["fecha"] < ventana]
        xg_train = xg[xg["fecha"] < ventana]
        m_gol = DixonColes().entrenar(fd_train)
        m_xg = DixonColesXG().entrenar_xg(xg_train, df_goles=fd_train)
        for _, p in lote.iterrows():
            try:
                pg = m_gol.predecir(p["local"], p["visitante"])
                px = m_xg.predecir(p["local"], p["visitante"])
            except KeyError:
                continue  # equipo que algun modelo no conoce -> fuera de AMBOS (pareado)
            filas.append({
                "fecha": p["fecha"], "liga": p["liga"],
                "real": resultado_real(p["goles_local"], p["goles_visitante"]),
                "p_gol": (pg["prob_local"], pg["prob_empate"], pg["prob_visitante"]),
                "p_xg": (px["prob_local"], px["prob_empate"], px["prob_visitante"]),
                **{c: p.get(c) for c in ("PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
                                          "B365H", "B365D", "B365A")},
            })
        ventana = tope
    print(f"Partidos evaluados (ambos modelos): {len(filas)}")
    return pd.DataFrame(filas)


IDX = {0: "H", 1: "D", 2: "A"}
NOMBRE_MERCADO = {0: "Victoria local", 1: "Empate", 2: "Victoria visitante"}


def _evaluar(res: pd.DataFrame, col_probs: str, umbral: float) -> dict:
    """ROI/CLV global y por mercado del modelo `col_probs` con senal > umbral."""
    tot = {"n": 0, "gan": 0, "roi": 0.0, "clv": 0.0, "beat": 0}
    por_mercado = {i: {"n": 0, "roi": 0.0, "clv": 0.0} for i in range(3)}
    for _, f in res.iterrows():
        pre = _demargin([f["PSH"], f["PSD"], f["PSA"]])
        cie = _demargin([f["PSCH"], f["PSCD"], f["PSCA"]])
        if not pre or not cie:
            continue
        pm = f[col_probs]
        edges = [pm[i] - pre[i] for i in range(3)]
        i = max(range(3), key=lambda k: edges[k])
        if edges[i] <= umbral:
            continue
        cuota = max([c for c in (f[["B365H", "B365D", "B365A"][i]], f[["PSH", "PSD", "PSA"][i]])
                     if pd.notna(c)], default=None)
        if not cuota or cuota <= 1:
            continue
        win = f["real"] == IDX[i]
        pnl = (cuota - 1.0) if win else -1.0
        clv = cuota * cie[i] - 1.0
        tot["n"] += 1
        tot["gan"] += win
        tot["roi"] += pnl
        tot["clv"] += clv
        tot["beat"] += clv > 0
        por_mercado[i]["n"] += 1
        por_mercado[i]["roi"] += pnl
        por_mercado[i]["clv"] += clv
    n = tot["n"] or 1
    return {
        "n": tot["n"], "hit": tot["gan"] / n, "roi": tot["roi"] / n,
        "clv": tot["clv"] / n, "beat": tot["beat"] / n,
        "mercados": {NOMBRE_MERCADO[i]: {
            "n": m["n"],
            "roi": m["roi"] / m["n"] if m["n"] else None,
            "clv": m["clv"] / m["n"] if m["n"] else None,
        } for i, m in por_mercado.items()},
    }


def reportar(res: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print("  DC CLASICO (goles) vs DC-xG (Understat)  |  mismas cuotas, mismos partidos")
    print("=" * 76)
    for u in UMBRALES:
        g = _evaluar(res, "p_gol", u)
        x = _evaluar(res, "p_xg", u)
        print(f"\n  --- umbral de edge {u*100:.0f}% ---")
        print(f"  {'modelo':<10}{'n':>6}{'hit':>8}{'ROI':>9}{'CLV':>9}{'beat-close':>12}")
        print(f"  {'goles':<10}{g['n']:>6}{g['hit']*100:>7.1f}%{g['roi']*100:>+8.1f}%"
              f"{g['clv']*100:>+8.2f}%{g['beat']*100:>11.1f}%")
        print(f"  {'xG':<10}{x['n']:>6}{x['hit']*100:>7.1f}%{x['roi']*100:>+8.1f}%"
              f"{x['clv']*100:>+8.2f}%{x['beat']*100:>11.1f}%")

    # Desglose por mercado con el umbral central (4%)
    u = 0.04
    print(f"\n  --- desglose por MERCADO (umbral {u*100:.0f}%) ---")
    print(f"  {'mercado':<20}{'modelo':<8}{'n':>6}{'ROI':>9}{'CLV':>9}")
    for nombre in NOMBRE_MERCADO.values():
        for etiqueta, col in (("goles", "p_gol"), ("xG", "p_xg")):
            m = _evaluar(res, col, u)["mercados"][nombre]
            roi = f"{m['roi']*100:+8.1f}%" if m["roi"] is not None else "     s/d"
            clv = f"{m['clv']*100:+8.2f}%" if m["clv"] is not None else "     s/d"
            print(f"  {nombre:<20}{etiqueta:<8}{m['n']:>6}{roi}{clv}")

    print("\n  COMO LEER: el CLV es el veredicto (ROI con n de cientos = ruido +-3-4pp).")
    print("  CLV > 0 sostenido = el modelo anticipa a Pinnacle = edge real -> cronjob.")
    print("  CLV < 0 = el mercado sigue ganando; el xG mejora el modelo pero no alcanza.")


def main() -> None:
    temporada = sys.argv[1] if len(sys.argv) >= 2 else TEMPORADA_TEST
    dias = int(sys.argv[2]) if len(sys.argv) >= 3 else DIAS
    res = correr(temporada, dias)
    if res.empty:
        print("Sin partidos evaluados. ¿Falta la ingesta xG de temporadas previas? "
              "Corre: python -m src.ingest_xg --temporadas 2223 2324 2425")
        return
    reportar(res)


if __name__ == "__main__":
    main()
