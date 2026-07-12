"""Ingesta de xG/xPts por partido desde Understat (via soccerdata) -> SQLite de clubes.

Fase 1 del producto de clubes 2026-27: baja las 4 grandes ligas (Espana, Inglaterra,
Alemania, Italia) y guarda POR PARTIDO: goles, xG, npxG, puntos, xPts (Understat lo trae
YA calculado), PPDA y deep completions -- con id_equipo_local/id_equipo_visitante y
nombre_local/nombre_visitante EXPLICITOS para poder consultar contra quien jugo un equipo.

DB: data/processed/dsoccer_clubes.db
  - tabla partidos_xg  (PK game_id de Understat -> re-correr NO duplica, actualiza)
  - vista equipos_temporada (puntos vs xPts acumulados + goles vs xG = sobre/sub-rendimiento)

Robustez:
  - try/except + reintentos con backoff exponencial POR liga-temporada (si Understat
    rate-limitea o se cae la red, se reintenta y si no, se saltea SIN tumbar el resto).
  - Una temporada sin datos (ej. 2627 antes del 15 de agosto) se reporta y se saltea.
  - Pausa aleatoria entre ligas (scraping respetuoso).
  - soccerdata cachea en ~/soccerdata/data/Understat; --refrescar fuerza re-descarga
    (para el cron semanal del server sobre la temporada en curso).

Uso:
    python -m src.ingest_xg                          # temporadas de config (2526, 2627)
    python -m src.ingest_xg --temporadas 2425 2526   # override
    python -m src.ingest_xg --refrescar              # ignora el cache (cron semanal)
    python -m src.ingest_xg --export-csv             # ademas exporta partidos_xg.csv
"""

from __future__ import annotations

import random
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from src import config

DB = config.DATA_PROC / "dsoccer_clubes.db"
REINTENTOS = 3

SQL_TABLA = """
CREATE TABLE IF NOT EXISTS partidos_xg (
    game_id             INTEGER PRIMARY KEY,   -- id de partido de Understat (idempotencia)
    liga                TEXT NOT NULL,
    temporada           TEXT NOT NULL,
    fecha               TEXT NOT NULL,
    id_equipo_local     INTEGER NOT NULL,
    id_equipo_visitante INTEGER NOT NULL,
    nombre_local        TEXT NOT NULL,
    nombre_visitante    TEXT NOT NULL,
    goles_local         INTEGER,
    goles_visitante     INTEGER,
    xg_local            REAL,
    xg_visitante        REAL,
    npxg_local          REAL,
    npxg_visitante      REAL,
    pts_local           INTEGER,
    pts_visitante       INTEGER,
    xpts_local          REAL,
    xpts_visitante      REAL,
    ppda_local          REAL,
    ppda_visitante      REAL,
    deep_local          INTEGER,
    deep_visitante      INTEGER,
    ingestado_en        TEXT
);
"""

SQL_INDICES = [
    "CREATE INDEX IF NOT EXISTS ix_xg_local  ON partidos_xg(id_equipo_local, fecha);",
    "CREATE INDEX IF NOT EXISTS ix_xg_visit  ON partidos_xg(id_equipo_visitante, fecha);",
    "CREATE INDEX IF NOT EXISTS ix_xg_liga   ON partidos_xg(liga, temporada, fecha);",
]

# Vista derivada (siempre consistente con partidos_xg, sin doble contabilidad):
# una fila por equipo-temporada con puntos vs xPts y goles vs xG acumulados.
SQL_VISTA = """
CREATE VIEW IF NOT EXISTS equipos_temporada AS
WITH lados AS (
    SELECT liga, temporada, id_equipo_local AS id_equipo, nombre_local AS equipo,
           pts_local AS pts, xpts_local AS xpts, goles_local AS gf, goles_visitante AS gc,
           xg_local AS xg_favor, xg_visitante AS xg_contra
    FROM partidos_xg
    UNION ALL
    SELECT liga, temporada, id_equipo_visitante, nombre_visitante,
           pts_visitante, xpts_visitante, goles_visitante, goles_local,
           xg_visitante, xg_local
    FROM partidos_xg
)
SELECT liga, temporada, id_equipo, equipo,
       COUNT(*)                        AS pj,
       SUM(pts)                        AS puntos,
       ROUND(SUM(xpts), 2)             AS xpts,
       ROUND(SUM(pts) - SUM(xpts), 2)  AS dif_pts,    -- >0 sobre-rendimiento ("suerte"), <0 mala suerte
       SUM(gf)                         AS gf,
       SUM(gc)                         AS gc,
       ROUND(SUM(xg_favor), 2)         AS xg_favor,
       ROUND(SUM(xg_contra), 2)        AS xg_contra,
       ROUND(SUM(gf) - SUM(xg_favor), 2) AS dif_goles -- >0 define sobre xG (candidato a regresion)
FROM lados
GROUP BY liga, temporada, id_equipo, equipo;
"""

# Mapeo columna Understat (read_team_match_stats) -> columna nuestra
MAPEO = {
    "game_id": "game_id", "date": "fecha",
    "home_team_id": "id_equipo_local", "away_team_id": "id_equipo_visitante",
    "home_team": "nombre_local", "away_team": "nombre_visitante",
    "home_goals": "goles_local", "away_goals": "goles_visitante",
    "home_xg": "xg_local", "away_xg": "xg_visitante",
    "home_np_xg": "npxg_local", "away_np_xg": "npxg_visitante",
    "home_points": "pts_local", "away_points": "pts_visitante",
    "home_expected_points": "xpts_local", "away_expected_points": "xpts_visitante",
    "home_ppda": "ppda_local", "away_ppda": "ppda_visitante",
    "home_deep_completions": "deep_local", "away_deep_completions": "deep_visitante",
}


def crear_esquema(con: sqlite3.Connection) -> None:
    con.execute(SQL_TABLA)
    for s in SQL_INDICES:
        con.execute(s)
    con.execute(SQL_VISTA)
    con.commit()


def _leer_understat(liga: str, temporada: str, refrescar: bool) -> pd.DataFrame | None:
    """Baja los partidos de una liga-temporada con reintentos + backoff. None si no se pudo.

    Cubre: caida de red, rate-limit de Understat (HTTP 429/5xx dentro de la excepcion de
    soccerdata) y temporadas aun sin datos (2627 antes de agosto): todos terminan en None
    y el caller SALTEA esa liga-temporada sin tumbar la corrida.
    """
    import soccerdata as sd
    for intento in range(1, REINTENTOS + 1):
        try:
            us = sd.Understat(leagues=liga, seasons=temporada, no_cache=refrescar)
            df = us.read_team_match_stats().reset_index()
            if df.empty:
                print(f"    {liga} {temporada}: sin partidos (temporada aun no arranco?) -> se saltea")
                return None
            return df
        except Exception as e:
            espera = 10 * (2 ** (intento - 1)) + random.uniform(0, 5)  # 10-15s, 20-25s, 40-45s
            print(f"    {liga} {temporada}: intento {intento}/{REINTENTOS} fallo "
                  f"({type(e).__name__}: {str(e)[:90]})")
            if intento < REINTENTOS:
                print(f"      reintentando en {espera:.0f}s (backoff por si es rate-limit)...")
                time.sleep(espera)
    print(f"    {liga} {temporada}: DROP tras {REINTENTOS} intentos -> se saltea (re-correr luego)")
    return None


def _upsert(con: sqlite3.Connection, df: pd.DataFrame, liga: str, temporada: str) -> int:
    """INSERT OR REPLACE por game_id: re-correr actualiza sin duplicar."""
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols_nuestras = list(MAPEO.values()) + ["liga", "temporada", "ingestado_en"]
    placeholders = ",".join("?" * len(cols_nuestras))
    sql = (f"INSERT OR REPLACE INTO partidos_xg ({','.join(cols_nuestras)}) "
           f"VALUES ({placeholders})")
    n = 0
    for _, r in df.iterrows():
        # Solo partidos JUGADOS (los futuros vienen sin goles); NaN -> None para SQLite.
        if pd.isna(r.get("home_goals")) or pd.isna(r.get("away_goals")):
            continue
        fila = []
        for col_us in MAPEO:
            v = r.get(col_us)
            if pd.isna(v):
                fila.append(None)
            elif col_us == "date":
                fila.append(str(v)[:10])
            else:
                fila.append(v.item() if hasattr(v, "item") else v)
        fila += [liga, temporada, ahora]
        con.execute(sql, fila)
        n += 1
    con.commit()
    return n


def ingerir(temporadas: list[str], refrescar: bool = False, export_csv: bool = False) -> None:
    con = sqlite3.connect(DB)
    crear_esquema(con)
    total = 0
    for liga in config.LIGAS_UNDERSTAT:
        for temporada in temporadas:
            print(f"  {liga} {temporada}...")
            df = _leer_understat(liga, temporada, refrescar)
            if df is None:
                continue
            n = _upsert(con, df, liga, temporada)
            total += n
            print(f"    -> {n} partidos jugados guardados")
            time.sleep(random.uniform(4, 8))  # pausa entre ligas (scraping respetuoso)

    n_db = con.execute("SELECT COUNT(*) FROM partidos_xg").fetchone()[0]
    print(f"\n[ingest_xg] corrida: {total} filas upsert | DB total: {n_db} partidos -> {DB.name}")

    if export_csv:
        out = config.DATA_PROC / "partidos_xg.csv"
        pd.read_sql_query("SELECT * FROM partidos_xg ORDER BY fecha", con).to_csv(
            out, index=False, encoding="utf-8")
        print(f"[ingest_xg] exportado {out.name}")
    con.close()


def main() -> None:
    args = sys.argv[1:]
    temporadas = list(config.TEMPORADAS_XG)
    if "--temporadas" in args:
        i = args.index("--temporadas")
        temporadas = [a for a in args[i + 1:] if not a.startswith("-")]
    ingerir(temporadas,
            refrescar="--refrescar" in args,
            export_csv="--export-csv" in args)


if __name__ == "__main__":
    main()
