"""Calibracion offline para el modulo de Player Props.

Computa desde dsoccer_historico.db (StatsBomb) + jugadores.csv (FBref):
  - CONVERSION_RATE_INTL: goles / tiros totales en partidos internacionales
  - ESCALA_TIROS_SELECCION: ratio tiros_intl/partido vs tiros_club/90 por jugador
  - Tasas de tiros/90 por jugador (internacionales, para jugadores con >= 3 partidos)
  - SOT_RATE por jugador y global

Guarda data/processed/tiros_intl.json (como calibracion.json o xg_ajuste.csv).
Artefacto comprometido en git: el server solo necesita 'git pull'.
En el server (314 partidos) el JSON tendra mas jugadores y mejores estimados.

Uso:
    python -m src.props_data          # construye tiros_intl.json
    python -m src.props_data --test   # ademas muestra top remateadores y calibracion
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.fixtures import _candidatos_por_nacion, emparejar_nombre
from src.jugadores_model import _norm

DB = config.DATA_PROC / "dsoccer_historico.db"
JUGADORES_CSV = config.DATA_PROC / "jugadores.csv"
SALIDA = config.DATA_PROC / "tiros_intl.json"

MIN_PARTIDOS_INTL = 3   # minimo de partidos internacionales para usar tasa intl del jugador
UMBRAL_MATCH = 82       # score minimo rapidfuzz para aceptar un emparejamiento de nombre


def _tasa_club(dfj: pd.DataFrame, nombre_norm: str) -> tuple[float, float]:
    """(tiros_90, sot_rate) desde FBref. Usa la temporada con mas minutos."""
    sub = dfj[dfj["player"].apply(_norm) == nombre_norm]
    if sub.empty or sub["noventas"].max() < 3:
        return 0.0, 0.0
    row = sub.loc[sub["minutos"].idxmax()]
    n90 = float(row["noventas"])
    tiros = float(row["tiros"])
    arco = float(row["tiros_arco"])
    if n90 <= 0 or tiros <= 0:
        return 0.0, 0.0
    return tiros / n90, arco / tiros


def construir(db: Path = DB, jugadores_csv: Path = JUGADORES_CSV,
              salida: Path = SALIDA) -> dict:
    if not db.exists():
        print(f"DB no encontrada: {db}. Corre: python -m src.ingesta_historica")
        return {}

    con = sqlite3.connect(db)

    # ---- 1. CONVERSION_RATE_INTL (goles reales + xG) -------------------- #
    filas_match = con.execute("""
        SELECT p.match_id, p.goles_local + p.goles_visitante,
               COALESCE(SUM(j.remates), 0), COALESCE(SUM(j.xg), 0.0)
        FROM partidos p
        LEFT JOIN jugador_partido_stats j ON j.match_id = p.match_id
        GROUP BY p.match_id
    """).fetchall()
    n_partidos = len(filas_match)
    goles_tot = sum(r[1] for r in filas_match if r[1] is not None)
    tiros_tot = sum(r[2] for r in filas_match)
    xg_tot_global = sum(r[3] for r in filas_match)
    conversion_rate = goles_tot / tiros_tot if tiros_tot > 0 else 0.091
    # xG-based: mas estable que goles reales (menos varianza match-a-match)
    conv_rate_xg = xg_tot_global / tiros_tot if tiros_tot > 0 else conversion_rate

    # ---- 2. Tiros internacionales por jugador (titulares, >= MIN_PARTIDOS_INTL) #
    rows_jug = con.execute("""
        SELECT j.player_norm, j.player, j.equipo,
               COUNT(DISTINCT j.match_id) AS n_partidos,
               SUM(j.remates) AS total_remates,
               COALESCE(SUM(j.xg), 0.0) AS total_xg
        FROM jugador_partido_stats j
        JOIN alineaciones a
          ON a.match_id = j.match_id AND a.player_id = j.player_id AND a.es_titular = 1
        GROUP BY j.player_norm
        HAVING n_partidos >= ?
        ORDER BY total_remates DESC
    """, (MIN_PARTIDOS_INTL,)).fetchall()
    con.close()

    # ---- 3. FBref data para matching y calibracion ----------------------- #
    dfj = pd.read_csv(jugadores_csv)

    # SOT_RATE global (FBref, jugadores con tiros > 5)
    dft = dfj[dfj["tiros"] > 5]
    sot_rate_default = float((dft["tiros_arco"] / dft["tiros"]).median())

    # Shadow rates por posicion desde FBref
    shadow = {}
    for pos in ["FW", "MF", "DF"]:
        sub = dfj[dfj["posicion"].str.startswith(pos) & (dfj["noventas"] > 5)]
        shadow[pos] = round(float((sub["tiros"] / sub["noventas"]).median()), 2) if not sub.empty else {"FW": 2.5, "MF": 1.4, "DF": 0.6}[pos]
    shadow["GK"] = 0.10

    # ---- 4. Match jugadores intl -> FBref (fuzzy por nacion) ------------- #
    # Para cada jugador en los datos intl, intentamos encontrar su nombre en FBref
    # via fuzzy match por nacion (mismo mecanismo que enriquecer_xg.py)

    # Mapeo equipo StatsBomb -> codigo FBref (mismo _codigo_nacion de fixtures)
    from src.fixtures import _codigo_nacion
    jugadores_out = {}
    ratios_escala = []

    for player_norm, player_name, equipo, n_part, total_rem, total_xg in rows_jug:
        cod = _codigo_nacion(equipo)
        tiros_pp = total_rem / n_part
        xg_pp = total_xg / n_part
        xg_per_shot = total_xg / total_rem if total_rem > 0 else conv_rate_xg

        candidatos = _candidatos_por_nacion(dfj, cod) if cod else {}
        fbref_name, score = emparejar_nombre(player_name, candidatos, UMBRAL_MATCH)

        tiros_90_club, sot_rate_jug = (0.0, sot_rate_default)
        if fbref_name:
            t90c, sotr = _tasa_club(dfj, _norm(fbref_name))
            if t90c > 0:
                tiros_90_club = t90c
                sot_rate_jug = sotr if sotr > 0 else sot_rate_default
                if 0.2 < (tiros_pp / t90c) < 4.0:  # filtro de outliers
                    ratios_escala.append(tiros_pp / t90c)

        jugadores_out[_norm(fbref_name) if fbref_name else player_norm] = {
            "n_intl": n_part,
            "tiros_pp_intl": round(tiros_pp, 3),
            "xg_pp_intl": round(xg_pp, 3),
            "xg_per_shot_intl": round(xg_per_shot, 4),
            "tiros_90_club": round(tiros_90_club, 3),
            "sot_rate": round(sot_rate_jug, 3),
            "fuente_fbref": fbref_name or "",
            "fuente": "intl" if n_part >= MIN_PARTIDOS_INTL else "club",
        }

    # ESCALA_TIROS_SELECCION: mediana de ratio intl/club (>1 = selecciones tiran mas)
    escala_tiros = float(np.median(ratios_escala)) if ratios_escala else 1.0

    datos = {
        "meta": {
            "n_partidos": n_partidos,
            "goles_totales": int(goles_tot),
            "tiros_totales": int(tiros_tot),
            "xg_totales": round(xg_tot_global, 2),
            "tiros_por_equipo_partido": round(tiros_tot / n_partidos / 2, 2),
            "conversion_rate_intl": round(conversion_rate, 4),
            "conversion_rate_xg": round(conv_rate_xg, 4),
            "escala_tiros_seleccion": round(escala_tiros, 3),
            "sot_rate_default": round(sot_rate_default, 3),
            "shadow_tiros_90": shadow,
            "min_partidos_intl": MIN_PARTIDOS_INTL,
        },
        "jugadores": jugadores_out,
    }
    salida.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[props_data] {len(jugadores_out)} jugadores | "
          f"conv_rate={conversion_rate:.3f} | escala_tiros={escala_tiros:.3f} | "
          f"sot_default={sot_rate_default:.3f} | guardado en {salida.name}")
    return datos


def cargar(path: Path = SALIDA) -> dict:
    """Carga tiros_intl.json. Devuelve {} si no existe."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    datos = construir()
    if not datos or "--test" not in sys.argv:
        return
    meta = datos["meta"]
    print(f"\n  Partidos: {meta['n_partidos']} | tiros/equipo/partido: {meta['tiros_por_equipo_partido']}")
    print(f"  conv_rate goles: {meta['conversion_rate_intl']:.4f}  "
          f"conv_rate xG: {meta['conversion_rate_xg']:.4f}  "
          f"(delta: {(meta['conversion_rate_xg']-meta['conversion_rate_intl'])*100:+.1f}pp)")
    print(f"  Escala tiros intl/club: {meta['escala_tiros_seleccion']:.3f}")
    print(f"  SOT rate default: {meta['sot_rate_default']:.3f}")
    # Top: solo jugadores con volumen real (>= 1.5 tiros/partido) para evitar
    # el sesgo de defensores que patearon 1 penal (xG/tiro ~ 0.76 por el PK)
    print(f"\n  Top remateadores por xG/tiro (lam >= 1.5 tiros/p):")
    top_xg = sorted(
        [(n, v) for n, v in datos["jugadores"].items()
         if v.get("xg_per_shot_intl", 0) > 0 and v["tiros_pp_intl"] >= 1.5],
        key=lambda x: x[1]["xg_per_shot_intl"], reverse=True)[:10]
    for nombre, v in top_xg:
        display = (v["fuente_fbref"] or nombre).encode("ascii", "replace").decode("ascii")
        print(f"    {display:<30}  xG/tiro={v['xg_per_shot_intl']:.3f}  "
              f"lam={v['tiros_pp_intl']:.2f}/p  xG/p={v['xg_pp_intl']:.2f}  n={v['n_intl']}")


if __name__ == "__main__":
    main()
