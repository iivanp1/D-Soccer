"""Ingesta de datos a nivel JUGADOR desde FBref (via soccerdata).

A diferencia de ingest.py (datos por equipo/partido), esto baja estadisticas por
jugador y temporada: goles, asistencias, minutos, tiros, faltas, tarjetas y la
NACIONALIDAD de cada jugador (clave para armar selecciones del Mundial).

COBERTURA: Big-5 europeo (combinado) + MLS, Brasileirao, Saudi, Eredivisie y Primeira
para no dejar afuera a Messi, Cristiano y las joyas no-europeas. Cada liga usa su
propio formato de temporada (ver config.LIGAS_FBREF_EXTRA).

ROBUSTEZ (FBref bloquea por rate-limiting):
  - Pausas aleatorias agresivas (5-10s) entre ligas para no parecer un bot.
  - try/except por liga: si una falla (bloqueo/CAPTCHA/liga no soportada), se salta
    y el resto sigue. No se pierde lo ya bajado.
  - soccerdata cachea en ~/soccerdata/data: lo ya scrapeado NO se vuelve a pedir.
  - Esquema de salida FIJO: toda liga produce las mismas columnas (rellenando con NaN
    las que FBref no tenga para esa competicion), asi jugadores_model nunca se rompe.

Uso:
    python -m src.ingest_jugadores

Genera data/processed/jugadores.csv con una fila por jugador-equipo-temporada.
"""

from __future__ import annotations

import random
import sys
import time

import pandas as pd

from src import config

# Esquema de salida fijo. NO cambiar los nombres: jugadores_model.py y mundial_engine.py
# dependen de ellos. Toda liga se conforma a esta lista (columnas faltantes -> NaN).
COLUMNAS_SALIDA = [
    "league", "season", "team", "player", "nacion", "posicion", "edad",
    "partidos", "minutos", "noventas", "goles", "asistencias", "goles_sin_pen",
    "amarillas", "rojas", "goles_90", "asist_90", "tiros", "tiros_arco",
    "faltas_com", "faltas_rec", "intercep", "tackles_ganados",
]
CLAVE = ["league", "season", "team", "player"]


def _aplanar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana el MultiIndex de columnas de FBref a nombres simples unidos por '_'."""
    df = df.copy()
    df.columns = [
        "_".join(str(c) for c in col if c not in ("", None)).strip()
        if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df.reset_index()


def _buscar(cols: list[str], *tokens: str) -> str | None:
    """Primera columna cuyo nombre contiene TODOS los tokens (case-insensitive)."""
    for c in cols:
        bajo = c.lower()
        if all(t.lower() in bajo for t in tokens):
            return c
    return None


def _leer(fb, stat_type: str) -> pd.DataFrame | None:
    """Lee un tipo de stat; devuelve None si FBref no lo tiene para esa liga."""
    try:
        return _aplanar(fb.read_player_season_stats(stat_type=stat_type))
    except Exception as e:
        print(f"    ! sin '{stat_type}' para esta liga ({type(e).__name__})")
        return None


def _seleccionar(df: pd.DataFrame, mapeo: dict) -> pd.DataFrame:
    """Toma de df las columnas halladas en `mapeo` y las renombra a la clave limpia."""
    cols = list(df.columns)
    encontrado = {k: _buscar(cols, *toks) for k, toks in mapeo.items()}
    presentes = {k: v for k, v in encontrado.items() if v}
    out = df[list(presentes.values())].copy()
    out.columns = list(presentes.keys())
    return out


def _unificar(estandar: pd.DataFrame, shooting: pd.DataFrame | None,
              misc: pd.DataFrame | None) -> pd.DataFrame:
    """Combina los 3 tipos de stat en una fila por jugador con el esquema fijo."""
    ident_metr = {
        "league": ("league",), "season": ("season",), "team": ("team",),
        "player": ("player",), "nacion": ("nation",), "posicion": ("pos",),
        "edad": ("age",),
        "partidos": ("Playing Time_MP",), "minutos": ("Playing Time_Min",),
        "noventas": ("Playing Time_90s",),
        "goles": ("Performance_Gls",), "asistencias": ("Performance_Ast",),
        "goles_sin_pen": ("Performance_G-PK",),
        "amarillas": ("Performance_CrdY",), "rojas": ("Performance_CrdR",),
        "goles_90": ("Per 90", "Gls"), "asist_90": ("Per 90", "Ast"),
    }
    base = _seleccionar(estandar, ident_metr)

    if shooting is not None:
        sub = _seleccionar(shooting, {
            "league": ("league",), "season": ("season",), "team": ("team",),
            "player": ("player",),
            "tiros": ("Standard_Sh",), "tiros_arco": ("Standard_SoT",),
        })
        base = base.merge(sub, on=CLAVE, how="left")

    if misc is not None:
        sub = _seleccionar(misc, {
            "league": ("league",), "season": ("season",), "team": ("team",),
            "player": ("player",),
            "faltas_com": ("Performance_Fls",), "faltas_rec": ("Performance_Fld",),
            "intercep": ("Performance_Int",), "tackles_ganados": ("Performance_TklW",),
        })
        base = base.merge(sub, on=CLAVE, how="left")

    # Conformar al esquema fijo: agregar faltantes como NaN, ordenar columnas
    for col in COLUMNAS_SALIDA:
        if col not in base.columns:
            base[col] = pd.NA
    return base[COLUMNAS_SALIDA]


def _bajar_fuente(sd, liga: str, temporadas: list[str]) -> pd.DataFrame | None:
    """Baja y unifica una liga (con sus temporadas). None si ni siquiera hay 'standard'."""
    fb = sd.FBref(leagues=liga, seasons=temporadas)
    estandar = _leer(fb, "standard")
    if estandar is None or estandar.empty:
        print(f"  ! {liga}: sin datos base, se salta.")
        return None
    shooting = _leer(fb, "shooting")
    misc = _leer(fb, "misc")
    base = _unificar(estandar, shooting, misc)

    # Fix Bundesliga: en el Big-5 combinado, "Fußball-Bundesliga" (con ß) no traduce y
    # queda con league=NaN. Es la unica liga del combinado que falla, asi que rellenamos.
    if liga == config.LIGAS_FBREF:
        falt = base["league"].isna().sum()
        if falt:
            base["league"] = base["league"].fillna("GER-Bundesliga")
            print(f"    (rellenadas {falt} filas de Bundesliga con league=NaN)")

    print(f"  ok {liga}: {len(base)} filas")
    return base


def main() -> None:
    try:
        import soccerdata as sd
    except ImportError:
        print("Falta soccerdata. Instala con:  python -m pip install soccerdata")
        sys.exit(1)

    config.DATA_PROC.mkdir(parents=True, exist_ok=True)
    # Auto-instala el mapeo de ligas en soccerdata (MLS/Brasil/Saudi) -> reproducible al clonar.
    config.instalar_league_dict_fbref()

    # Fuentes: el Big-5 combinado + cada liga extra con SUS temporadas.
    fuentes = [(config.LIGAS_FBREF, config.TEMPORADAS_FBREF)]
    fuentes += list(config.LIGAS_FBREF_EXTRA.items())

    print(f"Bajando {len(fuentes)} fuentes desde FBref (pausas anti-bloqueo de 5-10s).")
    print("La 1a vez tarda; lo cacheado no se re-scrapea.\n")

    frames = []
    for i, (liga, temporadas) in enumerate(fuentes):
        if i > 0:
            pausa = random.uniform(5, 10)  # pausa agresiva entre ligas
            print(f"  ... pausa {pausa:.1f}s antes de la siguiente liga")
            time.sleep(pausa)
        try:
            base = _bajar_fuente(sd, liga, temporadas)
            if base is not None and not base.empty:
                frames.append(base)
        except Exception as e:
            print(f"  ! {liga} fallo por completo ({type(e).__name__}: {e}). Se salta.")

    if not frames:
        print("\nNo se pudo bajar ninguna liga. Revisa conexion / bloqueos de FBref.")
        sys.exit(1)

    todo = pd.concat(frames, ignore_index=True)
    todo = todo.dropna(subset=["player"]).drop_duplicates(subset=CLAVE).reset_index(drop=True)

    destino = config.DATA_PROC / "jugadores.csv"
    todo.to_csv(destino, index=False, encoding="utf-8")

    print(f"\nListo: {len(todo)} filas jugador-temporada -> {destino}")
    print(f"Jugadores unicos: {todo['player'].nunique()}  |  "
          f"Nacionalidades: {todo['nacion'].nunique()}  |  "
          f"Ligas: {todo['league'].nunique()}")
    print("Ligas incluidas:", sorted(todo["league"].dropna().unique().tolist()))


if __name__ == "__main__":
    main()
