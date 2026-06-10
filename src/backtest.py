"""Backtesting riguroso del modelo Dixon-Coles (validacion walk-forward).

La pregunta que responde este script: ¿el modelo realmente predice bien partidos
que NUNCA vio, o solo memoriza el pasado?

Metodo:
  - Simulacion historica walk-forward: para cada partido de la temporada de testeo,
    el modelo se entrena SOLO con partidos jugados estrictamente antes.
  - Por rendimiento, reentrenamos cada N dias (ventana semanal) en vez de partido
    a partido. Sigue sin haber fuga de informacion del futuro.

Metrica: Brier Score multiclase (3 vias: H/D/A). Mide CALIBRACION, no acierto binario.
  BS = (p_H - y_H)^2 + (p_D - y_D)^2 + (p_A - y_A)^2     (rango 0 a 2; menos = mejor)

Comparamos contra un benchmark (frecuencias historicas base) para ver si Dixon-Coles
aporta valor predictivo real o no.

Uso:
    python -m src.backtest                 # testea la temporada 2425, reentreno semanal
    python -m src.backtest 2425 7          # temporada y dias de reentreno explicitos
"""

from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from tabulate import tabulate

from src import config
from src.dixon_coles import DixonColes

# --- Parametros por defecto ---
TEMPORADA_TEST = "2425"   # temporada que se usa como conjunto de testeo (out-of-sample)
DIAS_REENTRENO = 7        # cada cuantos dias se reentrena el modelo


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def cargar_datos() -> pd.DataFrame:
    ruta = config.DATA_PROC / "partidos.csv"
    if not ruta.exists():
        print("No encuentro data/processed/partidos.csv. Corre antes: python -m src.ingest")
        sys.exit(1)
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    # 'temporada' se relee como int desde el CSV; la normalizamos a texto ('2425')
    # para poder compararla con los codigos de temporada del resto del codigo.
    df["temporada"] = df["temporada"].astype(str)
    return df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)


def resultado_real(goles_local: int, goles_visitante: int) -> str:
    """Devuelve 'H' (gana local), 'D' (empate) o 'A' (gana visitante)."""
    if goles_local > goles_visitante:
        return "H"
    if goles_local < goles_visitante:
        return "A"
    return "D"


def brier_score(p_h: float, p_d: float, p_a: float, real: str) -> float:
    """Brier Score multiclase de un partido. real in {'H','D','A'}."""
    y = {"H": 0.0, "D": 0.0, "A": 0.0}
    y[real] = 1.0
    return (p_h - y["H"]) ** 2 + (p_d - y["D"]) ** 2 + (p_a - y["A"]) ** 2


def frecuencias_base(df: pd.DataFrame) -> tuple[float, float, float]:
    """Frecuencia historica de H/D/A en un set de datos (para el benchmark)."""
    res = [resultado_real(gl, gv) for gl, gv in zip(df["goles_local"], df["goles_visitante"])]
    s = pd.Series(res)
    n = len(s)
    return (
        (s == "H").sum() / n,
        (s == "D").sum() / n,
        (s == "A").sum() / n,
    )


# --------------------------------------------------------------------------- #
# Backtest walk-forward
# --------------------------------------------------------------------------- #
def correr_backtest(df: pd.DataFrame,
                    temporada_test: str = TEMPORADA_TEST,
                    dias_reentreno: int = DIAS_REENTRENO) -> pd.DataFrame:
    """Corre la validacion walk-forward y devuelve un DataFrame con una fila por
    partido testeado (brier del modelo, brier del benchmark, liga, etc.)."""

    test = df[df["temporada"] == temporada_test].copy()
    if test.empty:
        print(f"No hay partidos de la temporada {temporada_test} en los datos.")
        sys.exit(1)

    # Benchmark: frecuencias base calculadas SOLO con datos previos al test (sin fuga).
    previos = df[df["fecha"] < test["fecha"].min()]
    if previos.empty:
        print("No hay datos previos a la temporada de testeo para calibrar el benchmark.")
        sys.exit(1)
    pb_h, pb_d, pb_a = frecuencias_base(previos)
    print(f"Benchmark (frecuencias historicas previas): "
          f"H={pb_h:.1%}  D={pb_d:.1%}  A={pb_a:.1%}\n")

    fecha_inicio = test["fecha"].min()
    fecha_fin = test["fecha"].max()
    print(f"Testeando temporada {temporada_test}: {len(test)} partidos, "
          f"{fecha_inicio.date()} a {fecha_fin.date()}")
    print(f"Reentrenando cada {dias_reentreno} dias (walk-forward)...\n")

    filas = []
    saltados = 0
    ventana_inicio = fecha_inicio
    n_reentrenos = 0

    while ventana_inicio <= fecha_fin:
        ventana_fin = ventana_inicio + timedelta(days=dias_reentreno)
        lote = test[(test["fecha"] >= ventana_inicio) & (test["fecha"] < ventana_fin)]

        if lote.empty:
            ventana_inicio = ventana_fin
            continue

        # Entrenar SOLO con lo jugado estrictamente antes del inicio de la ventana.
        entrenamiento = df[df["fecha"] < ventana_inicio]
        modelo = DixonColes().entrenar(entrenamiento)
        n_reentrenos += 1

        for _, p in lote.iterrows():
            local, visitante = p["local"], p["visitante"]
            real = resultado_real(p["goles_local"], p["goles_visitante"])

            # Si el modelo no conoce a algun equipo (recien ascendido, etc.) lo saltamos
            # para comparar modelo y benchmark exactamente sobre los mismos partidos.
            try:
                pred = modelo.predecir(local, visitante)
            except KeyError:
                saltados += 1
                continue

            bs_modelo = brier_score(pred["prob_local"], pred["prob_empate"],
                                    pred["prob_visitante"], real)
            bs_bench = brier_score(pb_h, pb_d, pb_a, real)

            filas.append({
                "fecha": p["fecha"],
                "liga": p["liga"],
                "local": local,
                "visitante": visitante,
                "real": real,
                "brier_modelo": bs_modelo,
                "brier_benchmark": bs_bench,
            })

        ventana_inicio = ventana_fin

    print(f"Reentrenos realizados: {n_reentrenos}")
    print(f"Partidos evaluados: {len(filas)}  |  saltados (equipo nuevo): {saltados}\n")
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def reportar(res: pd.DataFrame) -> None:
    if res.empty:
        print("No se evaluo ningun partido.")
        return

    bs_modelo = res["brier_modelo"].mean()
    bs_bench = res["brier_benchmark"].mean()
    mejora = (bs_bench - bs_modelo) / bs_bench * 100

    print("=" * 52)
    print("  RESULTADO GLOBAL (Brier Score, menor = mejor)")
    print("=" * 52)
    print(f"  Modelo Dixon-Coles : {bs_modelo:.4f}")
    print(f"  Benchmark base     : {bs_bench:.4f}")
    print(f"  Mejora del modelo  : {mejora:+.1f}%  "
          f"({'aporta valor' if mejora > 0 else 'NO aporta valor'})")
    print()

    # Desglose por liga
    por_liga = (res.groupby("liga")
                .agg(partidos=("real", "size"),
                     brier_modelo=("brier_modelo", "mean"),
                     brier_benchmark=("brier_benchmark", "mean"))
                .reset_index())
    por_liga["mejora_%"] = (por_liga["brier_benchmark"] - por_liga["brier_modelo"]) \
        / por_liga["brier_benchmark"] * 100
    por_liga["liga"] = por_liga["liga"].map(lambda c: config.LIGAS.get(c, c))
    por_liga = por_liga.sort_values("brier_modelo")

    print("Desglose por liga (ordenado de mas preciso a menos):")
    print(tabulate(por_liga, headers="keys", floatfmt=".4f", showindex=False))


def main() -> None:
    temporada = sys.argv[1] if len(sys.argv) >= 2 else TEMPORADA_TEST
    dias = int(sys.argv[2]) if len(sys.argv) >= 3 else DIAS_REENTRENO

    df = cargar_datos()
    res = correr_backtest(df, temporada_test=temporada, dias_reentreno=dias)
    reportar(res)


if __name__ == "__main__":
    main()
