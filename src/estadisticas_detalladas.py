"""Capa de consulta para el frontend web: rachas, consistencia y valor del portero.

Las tres funciones que consumira la interfaz (todas devuelven dict/list serializables):

  obtener_racha_equipo(equipo_id, limite=5)
      Ultimos N partidos del equipo CRUZANDO Understat (goles, xG, por id de equipo) con
      football-data (tiros a puerta, corners, por nombre+fecha). Cada fila trae el NOMBRE
      DEL RIVAL explicito y la condicion Local/Visitante.

  calcular_consistencia(equipo_id, n=10)
      Media y desviacion estandar de corners y tiros a puerta de los ultimos N partidos,
      SEPARANDO local y visitante. std baja = mercado mas predecible para ese equipo.

  obtener_valor_portero(equipo_id)
      goles_evitados del portero TITULAR (mas minutos) del equipo: >0 ataja por encima
      de lo esperado por volumen de tiros (candidato a regresion), <0 por debajo.
      (Proxy de PSxG-GA; la limitacion esta documentada en ingest_porteros.)

CRUCE DE NOMBRES (el detalle importante): Understat, football-data y FBref usan nombres
distintos para el mismo club ("Athletic Club" / "Ath Bilbao" / "Athletic Club"). Se
resuelve con OVERRIDES manuales para los casos patologicos + fuzzy matching (rapidfuzz)
para el resto, y el cruce por partido usa (local, visitante, fecha +-1 dia). TODA funcion
maneja el caso "sin datos cruzados": devuelve None en los campos que falten, nunca revienta.

Uso (demo CLI):
    python -m src.estadisticas_detalladas 148          # por id Understat (Barcelona)
    python -m src.estadisticas_detalladas Barcelona    # por nombre
"""

from __future__ import annotations

import json
import sqlite3
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src import config

DB = config.DATA_PROC / "dsoccer_clubes.db"

# Liga Understat -> codigo de liga en football-data (partidos.csv)
LIGA_A_FD = {"ESP-La Liga": "SP1", "ENG-Premier League": "E0",
             "GER-Bundesliga": "D1", "ITA-Serie A": "I1"}

# Overrides Understat -> football-data (los que el fuzzy no resuelve con seguridad)
OVERRIDES_FD = {
    "athletic club": "Ath Bilbao", "atletico madrid": "Ath Madrid", "real betis": "Betis",
    "real sociedad": "Sociedad", "espanyol": "Espanol", "rayo vallecano": "Vallecano",
    "celta vigo": "Celta",
    "manchester united": "Man United", "manchester city": "Man City",
    "wolverhampton wanderers": "Wolves", "nottingham forest": "Nott'm Forest",
    "newcastle united": "Newcastle",
    "borussia dortmund": "Dortmund", "borussia m.gladbach": "M'gladbach",
    "rasenballsport leipzig": "RB Leipzig", "eintracht frankfurt": "Ein Frankfurt",
    "bayer leverkusen": "Leverkusen", "vfb stuttgart": "Stuttgart", "sc freiburg": "Freiburg",
    "fc cologne": "FC Koln", "st. pauli": "St Pauli", "mainz 05": "Mainz",
    "fc heidenheim": "Heidenheim", "hamburger sv": "Hamburg",
    "ac milan": "Milan", "hellas verona": "Verona", "parma calcio 1913": "Parma",
}
UMBRAL_FUZZY = 82


def _con() -> sqlite3.Connection:
    return sqlite3.connect(DB)


# --------------------------------------------------------------------------- #
#  Cruce de nombres Understat <-> football-data
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _fd() -> pd.DataFrame:
    """partidos.csv (football-data) con fecha normalizada a str YYYY-MM-DD."""
    df = pd.read_csv(config.DATA_PROC / "partidos.csv", parse_dates=["fecha"])
    df["fecha_str"] = df["fecha"].dt.strftime("%Y-%m-%d")
    return df


@lru_cache(maxsize=1)
def _mapa_fd() -> dict[tuple[str, str], str]:
    """{(liga_understat, nombre_understat) -> nombre_football-data} via overrides + fuzzy."""
    con = _con()
    us = pd.read_sql_query(
        "SELECT DISTINCT liga, nombre_local AS nombre FROM partidos_xg "
        "UNION SELECT DISTINCT liga, nombre_visitante FROM partidos_xg", con)
    con.close()
    fd = _fd()
    mapa = {}
    for liga_us, cod_fd in LIGA_A_FD.items():
        sub = fd[fd["liga"] == cod_fd]
        candidatos = sorted(set(sub["local"]) | set(sub["visitante"]))
        for nombre in us[us["liga"] == liga_us]["nombre"]:
            ov = OVERRIDES_FD.get(nombre.lower())
            if ov and ov in candidatos:
                mapa[(liga_us, nombre)] = ov
                continue
            m = process.extractOne(nombre, candidatos, scorer=fuzz.WRatio)
            if m and m[1] >= UMBRAL_FUZZY:
                mapa[(liga_us, nombre)] = m[0]
            # sin match -> no se agrega: el cruce devolvera None (manejado)
    return mapa


@lru_cache(maxsize=1)
def _indice_fd() -> dict[tuple[str, str, str], dict]:
    """{(local_fd, visitante_fd, fecha) -> fila de partidos.csv} para cruce O(1)."""
    idx = {}
    for _, r in _fd().iterrows():
        idx[(r["local"], r["visitante"], r["fecha_str"])] = r
    return idx


def _cruzar_fd(liga: str, nombre_local: str, nombre_visitante: str, fecha: str) -> dict | None:
    """Busca el partido en football-data por nombres mapeados y fecha (+-1 dia)."""
    mapa = _mapa_fd()
    loc = mapa.get((liga, nombre_local))
    vis = mapa.get((liga, nombre_visitante))
    if not loc or not vis:
        return None
    idx = _indice_fd()
    base = pd.Timestamp(fecha)
    for delta in (0, 1, -1):
        f = (base + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
        r = idx.get((loc, vis, f))
        if r is not None:
            return r
    return None


def _equipo(con: sqlite3.Connection, equipo_id: int) -> tuple[str, str] | None:
    """(nombre_understat, liga) de un id, o None si no existe."""
    r = con.execute(
        "SELECT nombre_local, liga FROM partidos_xg WHERE id_equipo_local=? "
        "UNION SELECT nombre_visitante, liga FROM partidos_xg WHERE id_equipo_visitante=? "
        "LIMIT 1", (equipo_id, equipo_id)).fetchone()
    return (r[0], r[1]) if r else None


def _filas_equipo(con: sqlite3.Connection, equipo_id: int, limite: int) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return con.execute(
        "SELECT * FROM partidos_xg WHERE id_equipo_local=? OR id_equipo_visitante=? "
        "ORDER BY fecha DESC LIMIT ?", (equipo_id, equipo_id, limite)).fetchall()


# --------------------------------------------------------------------------- #
#  1. Racha del equipo (para el frontend)
# --------------------------------------------------------------------------- #
def obtener_racha_equipo(equipo_id: int, limite: int = 5) -> list[dict]:
    """Ultimos `limite` partidos: goles/xG (Understat) + tiros a puerta/corners (football-data).

    Cada dict trae el nombre del RIVAL explicito. Si el cruce con football-data falla para
    un partido (nombre no mapeado / fecha distinta), tiros_puerta y corners van en None.
    """
    con = _con()
    filas = _filas_equipo(con, equipo_id, limite)
    con.close()
    out = []
    for r in filas:
        es_local = r["id_equipo_local"] == equipo_id
        lado, rlado = ("local", "visitante") if es_local else ("visitante", "local")
        fd = _cruzar_fd(r["liga"], r["nombre_local"], r["nombre_visitante"], r["fecha"])
        out.append({
            "fecha": r["fecha"],
            "rival": r[f"nombre_{rlado}"],
            "condicion": "Local" if es_local else "Visitante",
            "goles": r[f"goles_{lado}"],
            "goles_rival": r[f"goles_{rlado}"],
            "xG": round(r[f"xg_{lado}"], 2),
            "xG_rival": round(r[f"xg_{rlado}"], 2),
            "tiros_puerta": int(fd[f"tiros_arco_{lado}"]) if fd is not None and pd.notna(fd[f"tiros_arco_{lado}"]) else None,
            "corners": int(fd[f"corners_{lado}"]) if fd is not None and pd.notna(fd[f"corners_{lado}"]) else None,
            "corners_rival": int(fd[f"corners_{rlado}"]) if fd is not None and pd.notna(fd[f"corners_{rlado}"]) else None,
        })
    return out


# --------------------------------------------------------------------------- #
#  2. Consistencia (media + desviacion estandar, local/visitante)
# --------------------------------------------------------------------------- #
def calcular_consistencia(equipo_id: int, n: int = 10) -> dict:
    """Media y std de corners y tiros a puerta de los ultimos n partidos, por condicion.

    std BAJA = equipo consistente = mercado mas predecible (mejor para apostar lineas).
    Solo cuenta partidos con cruce football-data exitoso (n_local/n_visitante lo reportan).
    """
    racha = obtener_racha_equipo(equipo_id, limite=n)
    out = {}
    for cond in ("Local", "Visitante"):
        sub = [p for p in racha if p["condicion"] == cond and p["tiros_puerta"] is not None]
        if not sub:
            out[cond.lower()] = {"n": 0, "corners": None, "tiros_puerta": None}
            continue
        corners = np.array([p["corners"] for p in sub], dtype=float)
        tiros = np.array([p["tiros_puerta"] for p in sub], dtype=float)
        std = lambda a: round(float(a.std(ddof=1)), 2) if len(a) > 1 else None
        out[cond.lower()] = {
            "n": len(sub),
            "corners": {"media": round(float(corners.mean()), 2), "std": std(corners)},
            "tiros_puerta": {"media": round(float(tiros.mean()), 2), "std": std(tiros)},
        }
    return out


# --------------------------------------------------------------------------- #
#  3. Valor del portero titular
# --------------------------------------------------------------------------- #
def obtener_valor_portero(equipo_id: int) -> dict | None:
    """goles_evitados del portero TITULAR (mas minutos, temporada mas reciente) del equipo.

    El nombre de equipo Understat se cruza contra los squads de FBref (tabla porteros)
    por fuzzy. None si el equipo no existe o no hay porteros cruzables (caso manejado).
    """
    con = _con()
    info = _equipo(con, equipo_id)
    if info is None:
        con.close()
        return None
    nombre_us, liga = info
    squads = [r[0] for r in con.execute(
        "SELECT DISTINCT equipo FROM porteros WHERE liga=?", (liga,))]
    if not squads:
        con.close()
        return None
    m = process.extractOne(nombre_us, squads, scorer=fuzz.WRatio)
    if not m or m[1] < 75:
        con.close()
        return None
    squad = m[0]
    con.row_factory = sqlite3.Row
    r = con.execute(
        "SELECT * FROM porteros WHERE equipo=? ORDER BY temporada DESC, minutos DESC LIMIT 1",
        (squad,)).fetchone()
    con.close()
    if r is None:
        return None
    ge = r["goles_evitados"]
    return {
        "portero": r["player"], "equipo": squad, "liga": r["liga"], "temporada": r["temporada"],
        "minutos": r["minutos"], "tiros_recibidos": r["tiros_recibidos"],
        "goles_encajados": r["goles_encajados"], "save_pct": r["save_pct"],
        "save_pct_liga": r["save_pct_liga"], "goles_evitados": ge,
        "lectura": ("ataja POR ENCIMA de lo esperado (ojo regresion a la media)" if ge > 2
                    else "ataja POR DEBAJO de lo esperado" if ge < -2
                    else "en linea con lo esperado"),
        "nota": "proxy por volumen de tiros (no PSxG real); ver ingest_porteros",
    }


# --------------------------------------------------------------------------- #
def _id_por_nombre(nombre: str) -> int | None:
    con = _con()
    r = con.execute(
        "SELECT id_equipo_local FROM partidos_xg WHERE LOWER(nombre_local) LIKE ? "
        "UNION SELECT id_equipo_visitante FROM partidos_xg WHERE LOWER(nombre_visitante) LIKE ? "
        "LIMIT 1", (f"%{nombre.lower()}%", f"%{nombre.lower()}%")).fetchone()
    con.close()
    return r[0] if r else None


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m src.estadisticas_detalladas <equipo_id | nombre>")
        return
    arg = sys.argv[1]
    equipo_id = int(arg) if arg.isdigit() else _id_por_nombre(arg)
    if equipo_id is None:
        print(f"No encontre el equipo '{arg}' en partidos_xg.")
        return
    print(f"== obtener_racha_equipo({equipo_id}, limite=5) ==")
    print(json.dumps(obtener_racha_equipo(equipo_id), ensure_ascii=False, indent=1))
    print(f"\n== calcular_consistencia({equipo_id}) ==")
    print(json.dumps(calcular_consistencia(equipo_id), ensure_ascii=False, indent=1))
    print(f"\n== obtener_valor_portero({equipo_id}) ==")
    print(json.dumps(obtener_valor_portero(equipo_id), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
