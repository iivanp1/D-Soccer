"""Enriquece el rating ofensivo con xG REAL de StatsBomb (ataca la sub-diferenciacion).

El rating ofensivo del modelo usa npg_90 (goles-sin-penal realizados), que premia la SUERTE:
un goleador de racha (muchos goles, poco xG) queda sobrevalorado y un generador de peligro real
(mucho xG, pocos goles) subvalorado -> el modelo no diferencia bien a las selecciones.

Este modulo calcula, por jugador, un FACTOR de correccion desde su xG y goles INTERNACIONALES
(cosechados en dsoccer_historico.db) y lo escribe en data/processed/xg_ajuste.csv. El modelo
(jugadores_model.entrenar_jugadores) lo aplica como npg_90 *= factor:

    factor = clamp( (xG + K) / (goles + K), LO, HI ) ** GAMMA      (1.0 si pocos remates)

  - Generador (xG>goles, ej. Cristiano 4.5>2) -> factor>1 -> SUBE  (premia peligro real).
  - Suertudo  (goles>xG, ej. Lautaro 5>2.9)  -> factor<1 -> BAJA  (castiga la suerte).
  - K suaviza muestras chicas (factor->1); GAMMA<1 amortigua la transferencia internacional->club.

JOIN: los nombres de StatsBomb son legales completos ("cristiano ronaldo dos santos aveiro"),
asi que se usa matching DIFUSO por nacion (fixtures.emparejar_nombre), no exacto, si no se
perderian justo las estrellas. El artefacto xg_ajuste.csv es chico -> se commitea (como
calibracion.json) para que el server/colaborador lo use sin re-cosechar.

Uso:
    python -m src.enriquecer_xg            # construye xg_ajuste.csv + resumen
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from src import config
from src.fixtures import _candidatos_por_nacion, _codigo_nacion, emparejar_nombre
from src.jugadores_model import _norm

DB = config.DATA_PROC / "dsoccer_historico.db"
AJUSTE = config.DATA_PROC / "xg_ajuste.csv"

# Knobs (PRIORS tunables; los valida el re-tuneo de w en validar_statsbomb)
MIN_REMATES = 5      # remates internacionales minimos para confiar en el factor
K_XG = 3.0           # pseudo-cuenta: muestras chicas -> factor ~ 1
LO, HI = 0.6, 1.6    # clamp del ratio antes de GAMMA
GAMMA_XG = 0.5       # amortigua la transferencia internacional -> club

# Equipos StatsBomb cuyo nombre no resuelve fixtures._codigo_nacion (override minimo)
SB_OVERRIDES = {"jamaica": "JAM", "turkiye": "TUR", "czechia": "CZE", "south korea": "KOR"}


def _cod(equipo_sb: str) -> str | None:
    return SB_OVERRIDES.get(_norm(equipo_sb)) or _codigo_nacion(equipo_sb)


def _factor(xg: float, goles: float, remates: float) -> float:
    if remates < MIN_REMATES:
        return 1.0
    ratio = (xg + K_XG) / (goles + K_XG)
    return min(HI, max(LO, ratio)) ** GAMMA_XG


def _agg_statsbomb() -> pd.DataFrame:
    """Agrega xG/goles/remates por jugador (player_norm) desde la DB cosechada."""
    if not DB.exists():
        raise SystemExit(f"No existe {DB.name}. Corre antes: python -m src.ingesta_historica --desde 2018")
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("""
        SELECT player_norm, MAX(player) AS player, MAX(equipo) AS equipo,
               SUM(goles) AS goles, SUM(xg) AS xg, SUM(remates) AS remates,
               COUNT(DISTINCT match_id) AS n_part
        FROM jugador_partido_stats GROUP BY player_norm""", con)
    con.close()
    return df


def construir_ajuste() -> pd.DataFrame:
    sb = _agg_statsbomb()
    jug = pd.read_csv(config.DATA_PROC / "jugadores.csv")

    candidatos_cache: dict[str, dict] = {}
    filas = []
    for _, r in sb.iterrows():
        cod = _cod(r["equipo"])
        if not cod:
            continue
        cands = candidatos_cache.setdefault(cod, _candidatos_por_nacion(jug, cod))
        real, score = emparejar_nombre(r["player"], cands)
        if not real:
            continue
        filas.append({
            "player_modelo": real, "nacion": cod,
            "goles_int": int(r["goles"]), "xg_int": round(float(r["xg"]), 2),
            "remates_int": int(r["remates"]), "n_part": int(r["n_part"]),
            "score_match": round(score, 1),
            "factor_xg": round(_factor(r["xg"], r["goles"], r["remates"]), 3),
        })
    out = pd.DataFrame(filas)
    if not out.empty:
        # Si dos entradas StatsBomb matchean al mismo jugador del modelo, quedarse con la de
        # mas remates (mas info). Ordenar por remates desc y deduplicar.
        out = out.sort_values("remates_int", ascending=False).drop_duplicates("player_modelo")
        out = out.sort_values("factor_xg")
    return out


def cargar_ajuste(path=AJUSTE) -> dict:
    """{nombre_normalizado: factor_xg} para que entrenar_jugadores lo aplique. {} si no existe."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {_norm(n): float(f) for n, f in zip(df["player_modelo"], df["factor_xg"])}


def main() -> None:
    out = construir_ajuste()
    if out.empty:
        print("No se genero ningun ajuste (¿hay datos en la DB y matchean naciones?).")
        return
    out.to_csv(AJUSTE, index=False, encoding="utf-8")
    n_corr = (out["factor_xg"] != 1.0).sum()
    print(f"[xg] {len(out)} jugadores matcheados con jugadores.csv | {n_corr} con factor != 1.0")
    print(f"[xg] -> {AJUSTE.name}\n")
    print("=== mas castigados (suertudos: goles >> xG) ===")
    for _, r in out.head(6).iterrows():
        print(f"  {r.player_modelo:<26} {r.nacion}  goles={r.goles_int} xG={r.xg_int}  "
              f"factor={r.factor_xg}")
    print("\n=== mas premiados (generadores: xG >> goles) ===")
    for _, r in out.tail(6).iloc[::-1].iterrows():
        print(f"  {r.player_modelo:<26} {r.nacion}  goles={r.goles_int} xG={r.xg_int}  "
              f"factor={r.factor_xg}")


if __name__ == "__main__":
    main()
