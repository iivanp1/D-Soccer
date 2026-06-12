"""Computa nuestro PROPIO Elo desde la historia de resultados + ajusta el mapeo Elo->goles.

Fuente: el CSV de Kaggle "International football results from 1872 to 2026"
(martj42), un archivo con fecha, equipos, marcador, torneo y si fue en cancha neutral.
Descargarlo y dejarlo en data/raw/results.csv.

Por que esto importa:
  - 45.000+ partidos de selecciones (vs 2024-only de la API gratis): tuneo a ESCALA.
  - Computamos nuestro propio Elo (no dependemos de scrapear eloratings.net) y obtenemos
    el Elo PRE-PARTIDO historico, que nos deja AJUSTAR el mapeo Elo->goles con datos reales
    en vez del multiplicador 2E inventado en jugadores_model._calcular_lambda_elo.
  - OJO: este CSV NO trae alineaciones -> sirve para el lado ELO, no para el peso 'w' del
    blend ni el modelo de jugadores (eso necesita lineups: usar el harvest 2024).

Uso:
    python -m src.elo_history          # computa Elo + analiza el mapeo Elo->goles
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.fixtures import PAIS_API_A_CODIGO
from src.jugadores_model import _norm

RESULTS = config.RAIZ / "data" / "raw" / "results.csv"
SALIDA_ELO = config.DATA_PROC / "elo_propio.json"
# Fuente publica (se genera el dataset de Kaggle desde aca): sin login ni token.
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# --- Parametros del Elo (estilo World Football Elo) ---
ELO_INICIAL = 1500.0
VENTAJA_LOCAL = 70.0   # solo si no es cancha neutral
def _k_torneo(torneo: str) -> float:
    t = str(torneo).lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "qualif" in t or "nations" in t or "championship" in t or "copa" in t:
        return 45.0
    if "friendly" in t:
        return 20.0
    return 35.0


def _mult_goles(gd: int) -> float:
    """Multiplicador por diferencia de gol (World Football Elo)."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def _descargar_resultados() -> None:
    """Baja el CSV de resultados desde GitHub (martj42), publico y sin login."""
    import requests
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    print("Bajando resultados historicos (martj42/international_results)...")
    r = requests.get(RESULTS_URL, timeout=60)
    r.raise_for_status()
    RESULTS.write_text(r.text, encoding="utf-8")


def cargar_resultados(refrescar: bool = False) -> pd.DataFrame:
    if refrescar or not RESULTS.exists():
        _descargar_resultados()
    df = pd.read_csv(RESULTS)
    # Deteccion robusta de columnas (el dataset estandar usa estos nombres)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "date": ren[c] = "fecha"
        elif "home_team" in cl: ren[c] = "local"
        elif "away_team" in cl: ren[c] = "visitante"
        elif "home_score" in cl: ren[c] = "gl"
        elif "away_score" in cl: ren[c] = "gv"
        elif cl == "tournament": ren[c] = "torneo"
        elif cl == "neutral": ren[c] = "neutral"
    df = df.rename(columns=ren)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha", "local", "visitante", "gl", "gv"]).sort_values("fecha")
    df["gl"] = df["gl"].astype(int); df["gv"] = df["gv"].astype(int)
    if "neutral" not in df.columns:
        df["neutral"] = False
    if "torneo" not in df.columns:
        df["torneo"] = "Friendly"
    return df.reset_index(drop=True)


def computar_elo(df: pd.DataFrame):
    """Recorre los partidos cronologicamente y computa el Elo. Devuelve (df_con_elo_pre, ratings)."""
    ratings: dict[str, float] = {}
    pre_l, pre_v = [], []
    for _, m in df.iterrows():
        rl = ratings.get(m["local"], ELO_INICIAL)
        rv = ratings.get(m["visitante"], ELO_INICIAL)
        pre_l.append(rl); pre_v.append(rv)

        ventaja = 0.0 if bool(m["neutral"]) else VENTAJA_LOCAL
        e_local = 1.0 / (1.0 + 10 ** (-(rl - rv + ventaja) / 400.0))
        w = 1.0 if m["gl"] > m["gv"] else (0.5 if m["gl"] == m["gv"] else 0.0)
        ajuste = _k_torneo(m["torneo"]) * _mult_goles(m["gl"] - m["gv"]) * (w - e_local)
        ratings[m["local"]] = rl + ajuste
        ratings[m["visitante"]] = rv - ajuste

    df = df.copy()
    df["elo_l"] = pre_l; df["elo_v"] = pre_v
    return df, ratings


def analizar(df: pd.DataFrame) -> None:
    """Ajusta el mapeo Elo->goles y mide el poder predictivo del Elo (en partidos recientes)."""
    reciente = df[df["fecha"] >= "2015-01-01"].copy()
    reciente["dr"] = reciente["elo_l"] - reciente["elo_v"]
    reciente["gd"] = reciente["gl"] - reciente["gv"]
    reciente["total"] = reciente["gl"] + reciente["gv"]

    # 1) Mapeo: supremacia de gol ~ pendiente * diferencia de Elo
    pend, corte = np.polyfit(reciente["dr"], reciente["gd"], 1)
    print(f"Partidos analizados (>=2015): {len(reciente)}")
    print(f"Media de goles por equipo: {reciente['total'].mean()/2:.2f}  "
          f"(cross-check del base_real 1.39)")
    print(f"\nMAPEO Elo->goles ajustado a datos:")
    print(f"  supremacia_gol ~ {pend:.5f} * dif_Elo   ->  +400 Elo = +{pend*400:.2f} goles de ventaja")
    print(f"  (nuestro _calcular_lambda_elo usa multiplicador 2E; esto lo calibra de verdad)")

    # 2) Poder predictivo: el equipo de mas Elo, ¿gana los decididos?
    dec = reciente[reciente["gd"] != 0]
    acierto = ((dec["dr"] > 0) == (dec["gd"] > 0)).mean()
    print(f"\nPODER PREDICTIVO del Elo (partidos decididos, >=2015):")
    print(f"  el equipo con mas Elo gano el {acierto*100:.1f}% de las veces  ({len(dec)} partidos)")


def main() -> None:
    import sys
    df = cargar_resultados(refrescar="--refrescar" in sys.argv)
    print(f"Resultados cargados: {len(df)} partidos, {df['fecha'].min().date()} a {df['fecha'].max().date()}\n")
    df_elo, ratings = computar_elo(df)

    # Guardar nuestro Elo actual mapeado a codigos FBref (para usar en el motor)
    elo_fbref = {}
    for equipo, r in ratings.items():
        cod = PAIS_API_A_CODIGO.get(_norm(equipo))
        if cod:
            elo_fbref[cod] = {"overall": round(r, 1)}
    SALIDA_ELO.write_text(json.dumps(elo_fbref, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Elo propio guardado: {len(elo_fbref)} selecciones -> {SALIDA_ELO.name}")
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:8]
    print("Top 8 (Elo computado por nosotros):")
    for n, r in top:
        print(f"  {n:<16} {r:.0f}")
    print()
    analizar(df_elo)


if __name__ == "__main__":
    main()
