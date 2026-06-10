"""Calibra el mapeo rating->goles del Motor Mundialista con datos REALES de selecciones.

Problema: con XIs de alto rating, el mapeo bruto infla los goles (ARG-FRA ~4.5 totales),
irreal para el contexto cerrado de un Mundial.

Solucion (honesta sobre que es y que no es riguroso):
  1. NIVEL (riguroso): medimos la media real de goles/equipo en los ultimos Mundiales
     (resultados de soccerdata/FBref). Ese es el ancla 'base_real'.
  2. ESCALA (de nuestros datos): centramos ataque/defensa en el promedio de las
     selecciones, arreglando que ataque (~3) y defensa (~1.5) esten en escalas distintas.
  3. COMPRESION (prior, NO ajustado): cuanto se separan los equipos fuertes de la media.
     No se puede ajustar sin las alineaciones historicas de cada Mundial ligadas a la
     forma de club de entonces (dato que no tenemos). Es un parametro elegido, no fiteado.

Uso:
    python -m src.calibrar_internacional

Guarda data/processed/calibracion.json, que mundial_engine.py carga automaticamente.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src import config
from src.jugadores_model import JugadoresModel

# Mundiales a usar como benchmark de la media real de goles
TORNEOS = [("INT-World Cup", "2022"), ("INT-World Cup", "2018")]
# Compresion: prior. Mas bajo = partidos mas cerrados (mas realista para un Mundial).
COMPRESION = 0.45


def _goles_de_score(score: str) -> tuple[int, int] | None:
    """Parsea 'X-Y' (ignorando penales tipo '1-1 (4-2)'). Devuelve (gl, gv) o None."""
    if not isinstance(score, str):
        return None
    nums = re.findall(r"\d+", score)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])  # resultado tras alargue, antes de penales
    return None


def media_real_mundiales() -> tuple[float, int]:
    """Media de goles por equipo por partido en los Mundiales definidos."""
    import soccerdata as sd

    goles, partidos = 0, 0
    for liga, temporada in TORNEOS:
        try:
            fb = sd.FBref(leagues=liga, seasons=[temporada])
            sched = fb.read_schedule()
        except Exception as e:
            print(f"  ! no pude leer {liga} {temporada}: {e}")
            continue
        for sc in sched["score"]:
            par = _goles_de_score(sc)
            if par:
                goles += par[0] + par[1]
                partidos += 1
        print(f"  {liga} {temporada}: {partidos} partidos acumulados, {goles} goles")

    if partidos == 0:
        # Respaldo documentado: media historica de Mundiales ~2.65 totales -> 1.33/equipo
        print("  ! sin datos scrapeados; uso respaldo de literatura (1.33 goles/equipo)")
        return 1.33, 0
    media_equipo = goles / (2 * partidos)
    return media_equipo, partidos


def main() -> None:
    print("1) Midiendo la media real de goles en Mundiales (datos FBref)...")
    media, n_part = media_real_mundiales()
    print(f"   -> media real: {media:.3f} goles/equipo/partido "
          f"({media*2:.2f} totales)  [{n_part} partidos]\n")

    print("2) Entrenando Motor Mundialista y calculando la 'seleccion promedio'...")
    df = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    jm = JugadoresModel().entrenar_jugadores(df)

    # Antes (sin calibrar)
    xi_arg, xi_fra = jm.seleccion_probable("ARG"), jm.seleccion_probable("FRA")
    antes = jm.predecir_partido_mundial(xi_arg, xi_fra)

    params = jm.calibrar(media, compresion=COMPRESION)
    print(f"   -> ataque_ref={params['atk_ref']:.2f}  defensa_ref={params['def_ref']:.2f}  "
          f"(promedio de {params['n_equipos_ref']} selecciones)")
    print(f"   -> compresion (prior): {params['compresion']}\n")

    # Despues (calibrado)
    despues = jm.predecir_partido_mundial(xi_arg, xi_fra)

    destino = config.DATA_PROC / "calibracion.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"3) Calibracion guardada -> {destino}\n")

    print("=== Efecto en ARG vs FRA (goles esperados) ===")
    print(f"   ANTES  (sin calibrar): {antes['goles_esp_local']:.2f} - {antes['goles_esp_visitante']:.2f}"
          f"  (total {antes['goles_esp_local']+antes['goles_esp_visitante']:.2f})")
    print(f"   DESPUES (calibrado):   {despues['goles_esp_local']:.2f} - {despues['goles_esp_visitante']:.2f}"
          f"  (total {despues['goles_esp_local']+despues['goles_esp_visitante']:.2f})")


if __name__ == "__main__":
    main()
