"""Descarga y limpieza de datos historicos de football-data.co.uk.

Uso:
    python -m src.ingest

Baja un CSV por cada combinacion liga/temporada definida en config.py,
los junta, limpia las columnas y guarda un unico parquet/csv en data/processed.
"""

from __future__ import annotations

import io
import sys

import pandas as pd
import requests

from src import config


def _descargar_csv(liga: str, temporada: str) -> pd.DataFrame | None:
    """Baja un CSV de una liga/temporada. Devuelve None si falla."""
    url = f"{config.URL_BASE}/{temporada}/{liga}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! No se pudo bajar {liga} {temporada}: {e}")
        return None

    # Algunos CSV viejos vienen con codificacion latin-1 y filas basura al final.
    df = pd.read_csv(io.StringIO(resp.content.decode("latin-1")), on_bad_lines="skip")
    df["liga"] = liga
    df["temporada"] = temporada
    return df


def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Deja solo las columnas que nos importan, renombradas y con tipos correctos."""
    # Conservamos columnas que existan (algunas ligas/temporadas no traen todas)
    presentes = {orig: nuevo for orig, nuevo in config.COLUMNAS.items() if orig in df.columns}
    out = df[list(presentes.keys()) + ["liga", "temporada"]].rename(columns=presentes)

    # Fecha: football-data usa dd/mm/yy o dd/mm/yyyy
    out["fecha"] = pd.to_datetime(out["fecha"], dayfirst=True, errors="coerce")

    # Columnas numericas: forzar a numero, las faltantes quedan NaN
    cols_num = [c for c in out.columns if c not in ("fecha", "local", "visitante",
                                                    "resultado", "arbitro", "liga", "temporada")]
    for c in cols_num:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Tiramos filas sin lo minimo indispensable (equipos y goles)
    out = out.dropna(subset=["local", "visitante", "goles_local", "goles_visitante"])
    out["goles_local"] = out["goles_local"].astype(int)
    out["goles_visitante"] = out["goles_visitante"].astype(int)

    return out.sort_values("fecha").reset_index(drop=True)


def main() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.DATA_PROC.mkdir(parents=True, exist_ok=True)

    frames = []
    print(f"Descargando {len(config.LIGAS)} ligas x {len(config.TEMPORADAS)} temporadas...")
    for liga, nombre in config.LIGAS.items():
        for temp in config.TEMPORADAS:
            df = _descargar_csv(liga, temp)
            if df is not None and not df.empty:
                # guardamos el crudo por si queremos reprocesar sin re-descargar
                (config.DATA_RAW / f"{liga}_{temp}.csv").write_text(
                    df.to_csv(index=False), encoding="utf-8"
                )
                frames.append(df)
                print(f"  ok {nombre} {temp}: {len(df)} partidos")

    if not frames:
        print("No se descargo nada. Revisa tu conexion o los codigos de liga.")
        sys.exit(1)

    crudo = pd.concat(frames, ignore_index=True)
    limpio = _limpiar(crudo)

    destino = config.DATA_PROC / "partidos.csv"
    limpio.to_csv(destino, index=False, encoding="utf-8")
    print(f"\nListo: {len(limpio)} partidos limpios -> {destino}")
    print(f"Rango de fechas: {limpio['fecha'].min().date()} a {limpio['fecha'].max().date()}")
    print(f"Equipos distintos: {pd.concat([limpio['local'], limpio['visitante']]).nunique()}")


if __name__ == "__main__":
    main()
