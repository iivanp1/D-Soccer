"""Validacion walk-forward del modelo de estilos (tiros, corners, faltas, tarjetas).

Como aca predecimos CONTEOS (no resultados H/D/A), la metrica no es Brier sino el
MAE (Mean Absolute Error): en promedio, ¿por cuantas tarjetas/corners/etc le erramos?

Igual que backtest.py: para cada partido de la temporada de testeo, el modelo se
entrena solo con el pasado, reentrenando cada N dias.

Comparamos contra un benchmark naive: "predecir siempre el promedio de la liga".
Si el modelo no le gana al promedio, no aporta nada.

El test clave es TARJETAS: ahi se ve si el factor arbitro realmente sirve.

Uso:
    python -m src.backtest_estilos            # temporada 2425, reentreno semanal
    python -m src.backtest_estilos 2425 7
"""

from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from tabulate import tabulate

from src import config
from src.estilos_model import EstilosModel

TEMPORADA_TEST = "2425"
DIAS_REENTRENO = 7

# Metrica -> como obtener el valor REAL (total del partido) desde una fila del csv
METRICAS = {
    "tiros_total":    lambda r: r["tiros_local"] + r["tiros_visitante"],
    "corners_total":  lambda r: r["corners_local"] + r["corners_visitante"],
    "faltas_total":   lambda r: r["faltas_local"] + r["faltas_visitante"],
    "tarjetas_total": lambda r: (r["amarillas_local"] + r["rojas_local"]
                                 + r["amarillas_visitante"] + r["rojas_visitante"]),
}
# Metrica -> como obtener la prediccion (total) desde el dict del modelo
PRED_TOTAL = {
    "tiros_total":    lambda p: p["tiros_local"] + p["tiros_vis"],
    "corners_total":  lambda p: p["corners_local"] + p["corners_vis"],
    "faltas_total":   lambda p: p["faltas_total"],
    "tarjetas_total": lambda p: p["tarjetas_total"],
}


def cargar_datos() -> pd.DataFrame:
    ruta = config.DATA_PROC / "partidos.csv"
    if not ruta.exists():
        print("Falta data/processed/partidos.csv. Corre antes: python -m src.ingest")
        sys.exit(1)
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df["temporada"] = df["temporada"].astype(str)
    return df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)


def benchmark_promedios(entreno: pd.DataFrame) -> dict[str, float]:
    """Promedio historico de cada metrica (la prediccion naive)."""
    out = {}
    for m, fn in METRICAS.items():
        vals = entreno.apply(fn, axis=1)
        out[m] = float(vals.mean(skipna=True))
    return out


def correr_backtest(df: pd.DataFrame, temporada_test: str, dias_reentreno: int) -> pd.DataFrame:
    test = df[df["temporada"] == temporada_test].copy()
    if test.empty:
        print(f"No hay partidos de la temporada {temporada_test}.")
        sys.exit(1)

    fecha_inicio, fecha_fin = test["fecha"].min(), test["fecha"].max()
    print(f"Testeando temporada {temporada_test}: {len(test)} partidos, "
          f"{fecha_inicio.date()} a {fecha_fin.date()}")
    print(f"Reentrenando cada {dias_reentreno} dias. Metrica: MAE (menor = mejor)\n")

    filas = []
    ventana = fecha_inicio
    n_reentrenos = 0
    while ventana <= fecha_fin:
        fin = ventana + timedelta(days=dias_reentreno)
        lote = test[(test["fecha"] >= ventana) & (test["fecha"] < fin)]
        if lote.empty:
            ventana = fin
            continue

        entreno = df[df["fecha"] < ventana]
        modelo = EstilosModel().entrenar(entreno)
        base = benchmark_promedios(entreno)
        n_reentrenos += 1

        for _, p in lote.iterrows():
            pred = modelo.predecir_metricas(p["local"], p["visitante"],
                                            p.get("arbitro"), p["liga"])
            registro = {"liga": p["liga"]}
            for m in METRICAS:
                real = METRICAS[m](p)
                if pd.isna(real):
                    continue
                registro[f"ae_modelo_{m}"] = abs(PRED_TOTAL[m](pred) - real)
                registro[f"ae_bench_{m}"] = abs(base[m] - real)
            filas.append(registro)

        ventana = fin

    print(f"Reentrenos: {n_reentrenos}  |  partidos evaluados: {len(filas)}\n")
    return pd.DataFrame(filas)


def reportar(res: pd.DataFrame) -> None:
    if res.empty:
        print("Sin datos evaluados.")
        return

    filas = []
    for m in METRICAS:
        col_mod, col_ben = f"ae_modelo_{m}", f"ae_bench_{m}"
        if col_mod not in res:
            continue
        mae_mod = res[col_mod].mean()
        mae_ben = res[col_ben].mean()
        mejora = (mae_ben - mae_mod) / mae_ben * 100
        filas.append({
            "metrica": m,
            "MAE_modelo": mae_mod,
            "MAE_benchmark": mae_ben,
            "mejora_%": mejora,
            "veredicto": "aporta" if mejora > 0 else "NO aporta",
        })

    print("=" * 64)
    print("  RESULTADO (MAE = error absoluto medio por partido)")
    print("=" * 64)
    print(tabulate(pd.DataFrame(filas), headers="keys", floatfmt=".3f", showindex=False))
    print("\n(El test clave es 'tarjetas_total': mide si el factor arbitro sirve.)")


def main() -> None:
    temporada = sys.argv[1] if len(sys.argv) >= 2 else TEMPORADA_TEST
    dias = int(sys.argv[2]) if len(sys.argv) >= 3 else DIAS_REENTRENO
    df = cargar_datos()
    res = correr_backtest(df, temporada, dias)
    reportar(res)


if __name__ == "__main__":
    main()
