"""Backtest de ROI REAL del detector de valor sobre clubes (con cuotas historicas).

LA prueba de fuego del proyecto: los CSVs de football-data.co.uk traen las cuotas de
Pinnacle PRE-cierre (PSH/PSD/PSA) y de CIERRE (PSCH/PSCD/PSCA) + Bet365 + promedio del
mercado, para las 7 ligas x 3 temporadas. Eso permite simular la estrategia completa
con dinero virtual y SIN fuga de futuro:

  1. Walk-forward: Dixon-Coles entrenado SOLO con partidos anteriores (igual que backtest.py).
  2. SENAL: apostar el resultado donde p_modelo supera a la prob de-marginada de Pinnacle
     pre-cierre por mas de un UMBRAL de edge.
  3. EJECUCION: a la mejor cuota entre Bet365 y Pinnacle (retail realista, sin outliers).
  4. VEREDICTO doble:
       - ROI flat (stake 1 por apuesta): el P&L que habrias tenido.
       - CLV vs el CIERRE de Pinnacle: cuota_tomada * prob_justa_cierre - 1. El CLV es
         el proxy de edge con MENOS varianza: si es negativo, el ROI positivo es suerte.

Mercados: 1X2 (Dixon-Coles validado, Brier 0.587) y Over/Under 2.5 (senal vs promedio
del mercado; el cierre para O/U es el promedio de cierre, no hay Pinnacle O/U en el CSV).

Guardarrail anti-engano: se reportan TODOS los umbrales (no solo el que mejor dio) y la
linea base "apostar todo sin senal" (= pagar el margen). Con ~2000 partidos el ROI tiene
ruido de +-3-4pp; el CLV converge mucho mas rapido.

Uso:
    python -m src.backtest_roi                # temporada 2425, reentreno semanal
    python -m src.backtest_roi 2425 7
"""

from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd

from src import config
from src.backtest import resultado_real
from src.dixon_coles import DixonColes
from src.valor import _demargin

TEMPORADA_TEST = "2425"
DIAS_REENTRENO = 7
UMBRALES = [0.02, 0.04, 0.06, 0.08]

# Columnas de cuotas que levantamos del CSV crudo (ademas de las de juego).
ODDS = ["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
        "B365H", "B365D", "B365A",
        "B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5", "AvgC>2.5", "AvgC<2.5"]


def cargar_con_cuotas() -> pd.DataFrame:
    """Carga los CSVs CRUDOS (data/raw) con partido + cuotas, esquema compatible con el DC."""
    frames = []
    for liga in config.LIGAS:
        for temporada in config.TEMPORADAS:
            p = config.DATA_RAW / f"{liga}_{temporada}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            keep = {"Date": "fecha", "HomeTeam": "local", "AwayTeam": "visitante",
                    "FTHG": "goles_local", "FTAG": "goles_visitante"}
            cols = [c for c in list(keep) + ODDS if c in df.columns]
            df = df[cols].rename(columns=keep)
            df["liga"] = liga
            df["temporada"] = temporada
            frames.append(df)
    todo = pd.concat(frames, ignore_index=True)
    todo["fecha"] = pd.to_datetime(todo["fecha"], dayfirst=True, format="mixed")
    todo = todo.dropna(subset=["fecha", "goles_local", "goles_visitante"])
    return todo.sort_values("fecha").reset_index(drop=True)


def correr(temporada_test: str = TEMPORADA_TEST, dias: int = DIAS_REENTRENO) -> pd.DataFrame:
    df = cargar_con_cuotas()
    test = df[df["temporada"] == temporada_test]
    print(f"Datos: {len(df)} partidos con cuotas | test {temporada_test}: {len(test)}")
    print(f"Walk-forward, reentreno cada {dias} dias. Esto tarda unos minutos...\n")

    filas = []
    ventana = test["fecha"].min()
    fin = test["fecha"].max()
    while ventana <= fin:
        tope = ventana + timedelta(days=dias)
        lote = test[(test["fecha"] >= ventana) & (test["fecha"] < tope)]
        if lote.empty:
            ventana = tope
            continue
        modelo = DixonColes().entrenar(df[df["fecha"] < ventana])
        for _, p in lote.iterrows():
            try:
                pred = modelo.predecir(p["local"], p["visitante"])
            except KeyError:
                continue  # equipo nuevo para el modelo
            filas.append({
                "fecha": p["fecha"], "liga": p["liga"],
                "real": resultado_real(p["goles_local"], p["goles_visitante"]),
                "over_real": int(p["goles_local"] + p["goles_visitante"] > 2.5),
                "p_mod": (pred["prob_local"], pred["prob_empate"], pred["prob_visitante"]),
                "p_over": pred["prob_over_2_5"],
                **{c: p.get(c) for c in ODDS},
            })
        ventana = tope
    print(f"Partidos evaluados: {len(filas)}")
    return pd.DataFrame(filas)


def _evaluar_1x2(res: pd.DataFrame, umbral: float) -> dict:
    """Apuesta el outcome con p_mod - p_sharp_pre > umbral (el de mayor discrepancia)."""
    n = gan = 0
    roi = clv = beat = 0.0
    idx = {0: "H", 1: "D", 2: "A"}
    for _, f in res.iterrows():
        pre = _demargin([f["PSH"], f["PSD"], f["PSA"]])
        cie = _demargin([f["PSCH"], f["PSCD"], f["PSCA"]])
        if not pre or not cie:
            continue
        pm = f["p_mod"]
        edges = [pm[i] - pre[i] for i in range(3)]
        i = max(range(3), key=lambda k: edges[k])
        if edges[i] <= umbral:
            continue
        # ejecucion: mejor entre Bet365 y Pinnacle pre-cierre
        cuota = max([c for c in (f[["B365H", "B365D", "B365A"][i]], f[["PSH", "PSD", "PSA"][i]])
                     if pd.notna(c)], default=None)
        if not cuota or cuota <= 1:
            continue
        n += 1
        win = f["real"] == idx[i]
        gan += win
        roi += (cuota - 1.0) if win else -1.0
        c = cuota * cie[i] - 1.0
        clv += c
        beat += c > 0
    return {"n": n, "roi": roi / n if n else 0.0, "hit": gan / n if n else 0.0,
            "clv": clv / n if n else 0.0, "beat": beat / n if n else 0.0}


def _evaluar_ou(res: pd.DataFrame, umbral: float) -> dict:
    """Over/Under 2.5: senal vs promedio del mercado; CLV vs promedio de CIERRE."""
    n = gan = 0
    roi = clv = beat = 0.0
    for _, f in res.iterrows():
        pre = _demargin([f["Avg>2.5"], f["Avg<2.5"]])
        cie = _demargin([f["AvgC>2.5"], f["AvgC<2.5"]])
        if not pre or not cie:
            continue
        pm = (f["p_over"], 1.0 - f["p_over"])
        edges = [pm[i] - pre[i] for i in range(2)]
        i = 0 if edges[0] >= edges[1] else 1
        if edges[i] <= umbral:
            continue
        cuota = f["B365>2.5"] if i == 0 else f["B365<2.5"]
        if pd.isna(cuota) or cuota <= 1:
            continue
        n += 1
        win = (f["over_real"] == 1) if i == 0 else (f["over_real"] == 0)
        gan += win
        roi += (cuota - 1.0) if win else -1.0
        c = cuota * cie[i] - 1.0
        clv += c
        beat += c > 0
    return {"n": n, "roi": roi / n if n else 0.0, "hit": gan / n if n else 0.0,
            "clv": clv / n if n else 0.0, "beat": beat / n if n else 0.0}


def _base_ciega(res: pd.DataFrame) -> float:
    """Linea base: apostar SIEMPRE al favorito de Pinnacle a cuota Bet365 (= pagar margen)."""
    roi = n = 0
    idx = {0: "H", 1: "D", 2: "A"}
    for _, f in res.iterrows():
        pre = _demargin([f["PSH"], f["PSD"], f["PSA"]])
        if not pre:
            continue
        i = max(range(3), key=lambda k: pre[k])
        cuota = f[["B365H", "B365D", "B365A"][i]]
        if pd.isna(cuota):
            continue
        n += 1
        roi += (cuota - 1.0) if f["real"] == idx[i] else -1.0
    return roi / n if n else 0.0


def reportar(res: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("  BACKTEST DE ROI (stake plano 1u) | ejecucion best(B365, Pinnacle pre)")
    print("=" * 74)
    print(f"\n  Linea base (favorito sharp a ciegas, sin modelo): ROI {_base_ciega(res)*100:+.1f}%")
    print("  (eso es el margen que se paga sin senal; el modelo debe superarlo)\n")
    for nombre, fn in (("1X2 (Dixon-Coles vs Pinnacle)", _evaluar_1x2),
                       ("Over/Under 2.5 (DC vs mercado)", _evaluar_ou)):
        print(f"  --- {nombre} ---")
        print(f"  {'umbral':>7}{'n':>6}{'hit':>7}{'ROI':>8}{'CLV':>8}{'beat-close':>11}")
        for u in UMBRALES:
            r = fn(res, u)
            print(f"  {u*100:>6.0f}%{r['n']:>6}{r['hit']*100:>6.1f}%{r['roi']*100:>+7.1f}%"
                  f"{r['clv']*100:>+7.2f}%{r['beat']*100:>10.1f}%")
        print()
    print("  COMO LEER: ROI = plata simulada (ruido +-3-4pp con n chico). CLV = el veredicto")
    print("  robusto: >0 sostenido = edge real; <0 = el ROI positivo (si lo hay) es varianza.")
    print("  beat-close = % de apuestas que le ganaron a la linea de cierre de Pinnacle.")


def main() -> None:
    temporada = sys.argv[1] if len(sys.argv) >= 2 else TEMPORADA_TEST
    dias = int(sys.argv[2]) if len(sys.argv) >= 3 else DIAS_REENTRENO
    res = correr(temporada, dias)
    if res.empty:
        print("Sin partidos evaluados (faltan CSVs crudos? correr python -m src.ingest).")
        return
    reportar(res)


if __name__ == "__main__":
    main()
