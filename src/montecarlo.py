"""Simulador de Montecarlo: juega un partido 10.000 veces, minuto a minuto.

Por que Montecarlo y no solo Poisson:
  El Poisson estatico asume que los eventos son INDEPENDIENTES y la tasa CONSTANTE.
  Eso es falso. Una roja temprana hunde a un equipo; el que gana se repliega. Montecarlo
  simula cada minuto con esas DEPENDENCIAS DINAMICAS (el "Factor Caos"), produciendo
  una distribucion realista de universos posibles en vez de un unico numero estatico.

Es AGNOSTICO al modelo: recibe un dict de tasas esperadas (de Dixon-Coles, del modelo
de estilos, del Motor Mundialista, o de un ML futuro) y devuelve la distribucion.

Como se maneja el tiempo (minuto a minuto):
  Una tasa esperada por partido (ej. 1.4 goles) se convierte en una probabilidad por
  minuto: p_min = tasa / 90. En cada uno de los 90 minutos se tira un "dado" (uniforme
  en [0,1]); si cae por debajo de p_min ajustada por el estado del partido, ocurre el
  evento. Vectorizamos las 10.000 simulaciones en arrays de numpy: 90 iteraciones sobre
  vectores de 10.000, en vez de 900.000 iteraciones sueltas. Rapido y con las
  dependencias dinamicas aplicadas como operaciones sobre arrays.

Uso:
    from src.montecarlo import SimuladorMontecarlo, construir_parametros
    params = construir_parametros(goles_local=1.8, goles_visitante=1.4,
                                  tarjetas_local=2.0, tarjetas_visitante=2.4,
                                  faltas_local=11, faltas_visitante=12,
                                  corners_local=5, corners_visitante=4)
    sim = SimuladorMontecarlo()
    res = sim.simular_partido(params, n_simulaciones=10000)
    sim.resumen(res)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MINUTOS = 90

# --- Factor Caos: cuanto cambian las dinamicas segun el estado del partido ---
RED_BOOST_RIVAL = 0.30   # el rival ataca 30% mas si tu equipo se queda con uno menos
RED_PENAL_ATAQUE = 0.25  # tu ataque cae 25% con un jugador menos
BONUS_VA_PERDIENDO = 0.12  # el que pierde arriesga mas: +12% de ataque
PENAL_VA_GANANDO = 0.10    # el que gana se repliega: -10% de ataque
FRAC_ROJA = 0.04         # ~4% de las tarjetas son rojas (reds/match ~0.15, cards/match ~4)


def construir_parametros(goles_local: float, goles_visitante: float,
                         tarjetas_local: float = 2.2, tarjetas_visitante: float = 2.2,
                         faltas_local: float = 11.0, faltas_visitante: float = 11.0,
                         corners_local: float = 5.0, corners_visitante: float = 5.0) -> dict:
    """Empaqueta las tasas esperadas (por partido) en el formato que espera el simulador."""
    return {
        "goles": (goles_local, goles_visitante),
        "tarjetas": (tarjetas_local, tarjetas_visitante),
        "faltas": (faltas_local, faltas_visitante),
        "corners": (corners_local, corners_visitante),
    }


@dataclass
class SimuladorMontecarlo:
    semilla: int | None = 42

    def simular_partido(self, parametros: dict, n_simulaciones: int = 10000) -> dict:
        rng = np.random.default_rng(self.semilla)
        N = n_simulaciones

        # Tasas por minuto (tasa_partido / 90). Cap a <1 por seguridad.
        gl_min, ga_min = (np.array(parametros["goles"]) / MINUTOS)
        tl_min, ta_min = (np.array(parametros["tarjetas"]) / MINUTOS)
        fl_min, fa_min = (np.array(parametros["faltas"]) / MINUTOS)
        cl_min, ca_min = (np.array(parametros["corners"]) / MINUTOS)

        # Estado de cada universo simulado (arrays de largo N)
        goles_l = np.zeros(N, dtype=np.int16)
        goles_v = np.zeros(N, dtype=np.int16)
        amar_l = np.zeros(N, dtype=np.int16); amar_v = np.zeros(N, dtype=np.int16)
        rojas_l = np.zeros(N, dtype=np.int16); rojas_v = np.zeros(N, dtype=np.int16)
        faltas_l = np.zeros(N, dtype=np.int16); faltas_v = np.zeros(N, dtype=np.int16)
        corner_l = np.zeros(N, dtype=np.int16); corner_v = np.zeros(N, dtype=np.int16)
        expuls_l = np.zeros(N, dtype=bool)  # tiene al menos un jugador expulsado
        expuls_v = np.zeros(N, dtype=bool)

        for _minuto in range(1, MINUTOS + 1):
            diff = goles_l - goles_v  # >0 gana local, <0 gana visitante

            # --- Multiplicador de ataque segun el marcador (el que pierde arriesga) ---
            mult_atk_l = np.ones(N)
            mult_atk_l[diff < 0] = 1 + BONUS_VA_PERDIENDO
            mult_atk_l[diff > 0] = 1 - PENAL_VA_GANANDO
            mult_atk_v = np.ones(N)
            mult_atk_v[diff > 0] = 1 + BONUS_VA_PERDIENDO
            mult_atk_v[diff < 0] = 1 - PENAL_VA_GANANDO

            # --- Factor Caos por expulsion: 10 vs 11 cambia el resto del partido ---
            mult_atk_l *= np.where(expuls_l, 1 - RED_PENAL_ATAQUE, 1.0)
            mult_atk_v *= np.where(expuls_v, 1 - RED_PENAL_ATAQUE, 1.0)
            # Si el rival tiene un expulsado, yo ataco mas
            tasa_gol_l = gl_min * mult_atk_l * np.where(expuls_v, 1 + RED_BOOST_RIVAL, 1.0)
            tasa_gol_v = ga_min * mult_atk_v * np.where(expuls_l, 1 + RED_BOOST_RIVAL, 1.0)

            # --- Goles: un "dado" por universo ---
            goles_l += (rng.random(N) < np.clip(tasa_gol_l, 0, 1)).astype(np.int16)
            goles_v += (rng.random(N) < np.clip(tasa_gol_v, 0, 1)).astype(np.int16)

            # --- Tarjetas: si ocurre, decidimos si es roja (expulsion) ---
            ev_card_l = rng.random(N) < tl_min
            roja_l = ev_card_l & (rng.random(N) < FRAC_ROJA)
            amar_l += (ev_card_l & ~roja_l).astype(np.int16)
            rojas_l += roja_l.astype(np.int16)
            expuls_l |= roja_l

            ev_card_v = rng.random(N) < ta_min
            roja_v = ev_card_v & (rng.random(N) < FRAC_ROJA)
            amar_v += (ev_card_v & ~roja_v).astype(np.int16)
            rojas_v += roja_v.astype(np.int16)
            expuls_v |= roja_v

            # --- Faltas y corners: conteos sin dependencia fuerte ---
            faltas_l += (rng.random(N) < fl_min).astype(np.int16)
            faltas_v += (rng.random(N) < fa_min).astype(np.int16)
            corner_l += (rng.random(N) < cl_min).astype(np.int16)
            corner_v += (rng.random(N) < ca_min).astype(np.int16)

        return self._consolidar(goles_l, goles_v, amar_l + rojas_l, amar_v + rojas_v,
                                rojas_l, rojas_v, faltas_l + faltas_v,
                                corner_l + corner_v, N)

    # ------------------------------------------------------------------ #
    def _consolidar(self, gl, gv, tl, tv, rl, rv, faltas_tot, corner_tot, N) -> dict:
        """Consolida los 10.000 universos en una matriz de frecuencias."""
        local = (gl > gv).mean()
        empate = (gl == gv).mean()
        visit = (gl < gv).mean()

        # Moda de marcadores (los mas frecuentes)
        marcadores = {}
        for a, b in zip(gl.tolist(), gv.tolist()):
            marcadores[(a, b)] = marcadores.get((a, b), 0) + 1
        top_marc = sorted(marcadores.items(), key=lambda kv: -kv[1])[:5]

        goles_tot = gl + gv
        tarj_tot = tl + tv

        def over(arr, linea):
            return float((arr > linea).mean())

        return {
            "n": N,
            "prob_local": float(local), "prob_empate": float(empate), "prob_visitante": float(visit),
            "marcadores_top": [((a, b), c / N) for (a, b), c in top_marc],
            "goles_esp": (float(gl.mean()), float(gv.mean())),
            "over_0_5_goles": over(goles_tot, 0.5),
            "over_1_5_goles": over(goles_tot, 1.5),
            "over_2_5_goles": over(goles_tot, 2.5),
            "over_3_5_goles": over(goles_tot, 3.5),
            "tarjetas_esp": float(tarj_tot.mean()),
            "over_2_5_tarjetas": over(tarj_tot, 2.5),
            "over_3_5_tarjetas": over(tarj_tot, 3.5),
            "over_4_5_tarjetas": over(tarj_tot, 4.5),
            "prob_alguna_roja": float(((rl + rv) > 0).mean()),
            "faltas_esp": float(faltas_tot.mean()),
            # Umbrales centrados en el nivel internacional (~28 faltas/partido tras calibrar)
            "over_24_5_faltas": over(faltas_tot, 24.5),
            "over_27_5_faltas": over(faltas_tot, 27.5),
            "over_30_5_faltas": over(faltas_tot, 30.5),
            "corners_esp": float(corner_tot.mean()),
            "over_8_5_corners": over(corner_tot, 8.5),
            "over_9_5_corners": over(corner_tot, 9.5),
            "over_10_5_corners": over(corner_tot, 10.5),
        }

    # ------------------------------------------------------------------ #
    def resumen(self, res: dict, local: str = "Local", visitante: str = "Visitante") -> None:
        print(f"=== Montecarlo: {res['n']:,} simulaciones ===")
        print(f"  Gana {local:<14} {res['prob_local']*100:5.1f}%")
        print(f"  Empate{'':<14} {res['prob_empate']*100:5.1f}%")
        print(f"  Gana {visitante:<14} {res['prob_visitante']*100:5.1f}%")
        print(f"  Goles esperados: {res['goles_esp'][0]:.2f} - {res['goles_esp'][1]:.2f}")
        print("  Marcadores mas probables:")
        for (a, b), p in res["marcadores_top"]:
            print(f"     {a}-{b}: {p*100:.1f}%")
        print(f"  Over 1.5 goles: {res['over_1_5_goles']*100:.1f}%  |  "
              f"Over 2.5: {res['over_2_5_goles']*100:.1f}%  |  Over 3.5: {res['over_3_5_goles']*100:.1f}%")
        print(f"  Tarjetas esperadas: {res['tarjetas_esp']:.1f}  |  "
              f"Over 3.5: {res['over_3_5_tarjetas']*100:.1f}%  |  Over 4.5: {res['over_4_5_tarjetas']*100:.1f}%")
        print(f"  Prob. alguna roja: {res['prob_alguna_roja']*100:.1f}%")
        print(f"  Corners esperados: {res['corners_esp']:.1f}  |  Over 9.5: {res['over_9_5_corners']*100:.1f}%")


if __name__ == "__main__":
    # Demo: tasas desde nuestros modelos VALIDADOS (Dixon-Coles + estilos) para un
    # partido de clubes, alimentando el Montecarlo.
    import sys
    import pandas as pd
    from src import config
    from src.dixon_coles import DixonColes
    from src.estilos_model import EstilosModel

    local = sys.argv[1] if len(sys.argv) > 1 else "Real Madrid"
    visitante = sys.argv[2] if len(sys.argv) > 2 else "Barcelona"

    df = pd.read_csv(config.DATA_PROC / "partidos.csv", parse_dates=["fecha"])
    df["temporada"] = df["temporada"].astype(str)

    dc = DixonColes().entrenar(df)
    est = EstilosModel().entrenar(df)
    pred_g = dc.predecir(local, visitante)
    pred_e = est.predecir_metricas(local, visitante, liga="SP1")

    params = construir_parametros(
        goles_local=pred_g["goles_esp_local"], goles_visitante=pred_g["goles_esp_visitante"],
        tarjetas_local=pred_e["tarjetas_local"], tarjetas_visitante=pred_e["tarjetas_vis"],
        faltas_local=pred_e["faltas_local"], faltas_visitante=pred_e["faltas_vis"],
        corners_local=pred_e["corners_local"], corners_visitante=pred_e["corners_vis"],
    )
    print(f"Tasas de entrada (de Dixon-Coles + estilos):")
    print(f"  goles {params['goles']}  tarjetas {tuple(round(x,2) for x in params['tarjetas'])}\n")

    sim = SimuladorMontecarlo()
    res = sim.simular_partido(params, n_simulaciones=10000)
    sim.resumen(res, local, visitante)
