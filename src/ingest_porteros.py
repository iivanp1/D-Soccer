"""Ingesta de PORTEROS desde FBref (keeper) -> tabla porteros en dsoccer_clubes.db.

Que guarda por portero-temporada: tiros a puerta recibidos (SoTA), goles encajados (GA),
paradas, % de paradas, porterias a cero, minutos -- y el "valor" del portero:

  goles_evitados = SoTA * (1 - save_pct_liga) - GA

  = cuantos goles evito vs un arquero PROMEDIO de su liga recibiendo el mismo volumen
  de tiros a puerta. Positivo -> ataja por ENCIMA de lo esperado (candidato a regresion
  a la media); negativo -> por debajo.

LIMITACION HONESTA: esto es un PROXY de PSxG-GA. El PSxG real (que ajusta por la CALIDAD
de cada tiro, no solo la cantidad) NO esta disponible: soccerdata 1.9.0 solo expone
stat_types [standard, keeper, shooting, playing_time, misc] para FBref (verificado en el
codigo fuente; mismo limite ya documentado con el xG de FBref). Si un equipo concede
tiros inusualmente faciles/dificiles, el proxy se sesga. Para rankear y detectar extremos
sirve; para afinar, la mejora futura es scrapear la tabla keepersadv de FBref directo.

Robustez: reintentos con backoff por si FBref rate-limitea; corre sobre el cache de
soccerdata si ya se descargo (segunda corrida = instantanea).

Uso:
    python -m src.ingest_porteros                     # temporada 2526 (default)
    python -m src.ingest_porteros --temporadas 2425 2526
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
TEMPORADAS_DEFAULT = ["2526"]
REINTENTOS = 3
BIG5 = "Big 5 European Leagues Combined"  # una sola request para las 5; filtramos las 4 nuestras

SQL_TABLA = """
CREATE TABLE IF NOT EXISTS porteros (
    player          TEXT NOT NULL,
    equipo          TEXT NOT NULL,
    liga            TEXT NOT NULL,
    temporada       TEXT NOT NULL,
    partidos        INTEGER,
    minutos         INTEGER,
    tiros_recibidos INTEGER,   -- SoTA: tiros a puerta recibidos
    goles_encajados INTEGER,   -- GA
    paradas         INTEGER,
    save_pct        REAL,      -- % de paradas del portero
    porterias_cero  INTEGER,   -- clean sheets
    save_pct_liga   REAL,      -- referencia: save% de la liga (ponderado por SoTA)
    goles_evitados  REAL,      -- SoTA*(1-save_pct_liga) - GA  (proxy de PSxG-GA, ver docstring)
    ingestado_en    TEXT,
    PRIMARY KEY (player, equipo, liga, temporada)
);
"""
SQL_INDICE = "CREATE INDEX IF NOT EXISTS ix_porteros_eq ON porteros(equipo, temporada);"


def _leer_keeper(temporada: str) -> pd.DataFrame | None:
    """Baja el stat 'keeper' del Big-5 con reintentos + backoff. None si no se pudo."""
    import soccerdata as sd
    from src.ingest_jugadores import _aplanar
    for intento in range(1, REINTENTOS + 1):
        try:
            fb = sd.FBref(leagues=BIG5, seasons=temporada)
            df = _aplanar(fb.read_player_season_stats(stat_type="keeper"))
            if df.empty:
                print(f"    keeper {temporada}: vacio (temporada aun sin datos?) -> se saltea")
                return None
            return df
        except Exception as e:
            espera = 15 * (2 ** (intento - 1)) + random.uniform(0, 5)
            print(f"    keeper {temporada}: intento {intento}/{REINTENTOS} fallo "
                  f"({type(e).__name__}: {str(e)[:90]})")
            if intento < REINTENTOS:
                time.sleep(espera)
    return None


def ingerir(temporadas: list[str]) -> None:
    con = sqlite3.connect(DB)
    con.execute(SQL_TABLA)
    con.execute(SQL_INDICE)
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0

    for temporada in temporadas:
        print(f"  FBref keeper {temporada}...")
        df = _leer_keeper(temporada)
        if df is None:
            continue
        # Bug conocido del Big-5 combinado: "Fussball-Bundesliga" (con eszett) no traduce
        # y los alemanes vienen con league=NaN (mismo fix que ingest_jugadores._bajar_fuente).
        df["league"] = df["league"].fillna("GER-Bundesliga")
        df = df[df["league"].isin(config.LIGAS_UNDERSTAT)].copy()

        # Numericos (FBref trae <NA>/None): a numerico o NaN
        for c in ("Performance_GA", "Performance_SoTA", "Performance_Saves",
                  "Performance_Save%", "Performance_CS", "Playing Time_MP", "Playing Time_Min"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Performance_SoTA", "Performance_GA"])

        # Save% de referencia POR LIGA, ponderado por volumen (sum saves / sum SoTA):
        # mas robusto que promediar los save% individuales (los suplentes con 3 tiros no pesan).
        ref = (df.groupby("league")
                 .apply(lambda g: g["Performance_Saves"].sum() / max(g["Performance_SoTA"].sum(), 1),
                        include_groups=False)
                 .to_dict())

        filas = 0
        for _, r in df.iterrows():
            liga = r["league"]
            sota, ga = float(r["Performance_SoTA"]), float(r["Performance_GA"])
            spct_liga = ref.get(liga, 0.70)
            goles_evitados = sota * (1.0 - spct_liga) - ga
            con.execute(
                "INSERT OR REPLACE INTO porteros VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["player"], r["team"], liga, temporada,
                 _in(r["Playing Time_MP"]), _in(r["Playing Time_Min"]),
                 int(sota), int(ga), _in(r["Performance_Saves"]),
                 _fl(r["Performance_Save%"]), _in(r["Performance_CS"]),
                 round(spct_liga * 100, 1), round(goles_evitados, 2), ahora))
            filas += 1
        con.commit()
        total += filas
        print(f"    -> {filas} porteros de las 4 ligas guardados "
              f"(save% liga: {', '.join(f'{k.split(chr(45))[0]} {v*100:.0f}%' for k, v in ref.items())})")

    n_db = con.execute("SELECT COUNT(*) FROM porteros").fetchone()[0]
    print(f"\n[ingest_porteros] corrida: {total} upsert | tabla porteros: {n_db} filas -> {DB.name}")
    con.close()


def _in(x) -> int | None:
    return int(x) if pd.notna(x) else None


def _fl(x) -> float | None:
    return float(x) if pd.notna(x) else None


def main() -> None:
    args = sys.argv[1:]
    temporadas = list(TEMPORADAS_DEFAULT)
    if "--temporadas" in args:
        i = args.index("--temporadas")
        temporadas = [a for a in args[i + 1:] if not a.startswith("-")]
    ingerir(temporadas)


if __name__ == "__main__":
    main()
