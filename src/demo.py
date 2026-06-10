"""Demo de punta a punta: carga datos, entrena Dixon-Coles y predice un partido.

Uso:
    python -m src.demo                       # usa dos equipos de ejemplo
    python -m src.demo "Real Madrid" "Barcelona"
"""

from __future__ import annotations

import sys

import pandas as pd
from tabulate import tabulate

from src import config
from src.dixon_coles import DixonColes


def cargar_datos() -> pd.DataFrame:
    ruta = config.DATA_PROC / "partidos.csv"
    if not ruta.exists():
        print("No encuentro data/processed/partidos.csv")
        print("Corre primero la ingesta:  python -m src.ingest")
        sys.exit(1)
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    return df


def main() -> None:
    df = cargar_datos()
    print(f"Datos: {len(df)} partidos, {df['fecha'].min().date()} a {df['fecha'].max().date()}\n")

    print("Entrenando Dixon-Coles...")
    modelo = DixonColes().entrenar(df)
    print(f"  ventaja de local (gamma): {modelo.gamma:.3f}")
    print(f"  correccion rho:           {modelo.rho:.3f}\n")

    print("Top 10 equipos por fuerza (ataque + defensa):")
    print(tabulate(modelo.ranking(10), headers="keys", floatfmt=".3f", showindex=False))
    print()

    # Equipos a predecir: de argumentos o dos por defecto que existan en los datos
    if len(sys.argv) >= 3:
        local, visitante = sys.argv[1], sys.argv[2]
    else:
        local, visitante = modelo.equipos[0], modelo.equipos[1]
        print(f"(sin argumentos, uso ejemplo: {local} vs {visitante})\n")

    pred = modelo.predecir(local, visitante)
    print(f"=== {pred['local']} vs {pred['visitante']} ===")
    print(f"  Gana {pred['local']:<18} {pred['prob_local']*100:5.1f}%")
    print(f"  Empate{'':<17} {pred['prob_empate']*100:5.1f}%")
    print(f"  Gana {pred['visitante']:<18} {pred['prob_visitante']*100:5.1f}%")
    print(f"  Marcador mas probable:  {pred['marcador_probable'][0]}-{pred['marcador_probable'][1]}")
    print(f"  Goles esperados:        {pred['goles_esp_local']:.2f} - {pred['goles_esp_visitante']:.2f}")
    print(f"  Prob. mas de 2.5 goles: {pred['prob_over_2_5']*100:.1f}%")


if __name__ == "__main__":
    main()
