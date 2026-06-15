"""Orquestador final de D-Soccer: fusiona el Motor Mundialista con el Montecarlo.

Pipeline completo de una prediccion de partido internacional:

    jugadores.csv ─▶ JugadoresModel ─▶ tasas base (goles, faltas, tarjetas)
                                            │
              EstilosModel (arbitro) ──────▶│  (factor observado/esperado, si aplica)
                                            ▼
                                  SimuladorMontecarlo (10.000 universos)
                                            ▼
                              Distribucion completa de mercado (gestion de riesgo)

Sobre el "Factor Caos sensible a jugadores" (rojas):
  Rastrear que jugador puntual se expulsa en cada uno de los 10.000 universos no aporta
  poder predictivo (una roja = jugar con 10). Usamos la opcion que el propio spec admite:
  "perdida de un peso defensivo promedio", que es justo lo que el Montecarlo ya modela
  (degrada el ataque del equipo con uno menos y potencia al rival el resto del partido).

Uso:
    python -m src.mundial_engine ARG FRA
    python -m src.mundial_engine ARG FRA --arbitro "Szymon Marciniak"
    python -m src.mundial_engine ARG FRA --xi-local "Lionel Messi,Julian Alvarez,..." \\
                                          --xi-visitante "Kylian Mbappe,..."
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.jugadores_model import JugadoresModel
from src.montecarlo import SimuladorMontecarlo, construir_parametros

# Corners: no tenemos dato por jugador, asi que repartimos un total internacional tipico
# segun la fuerza ofensiva relativa (heuristico declarado, no bottom-up).
CORNERS_TOTAL_BASE = 10.0


def _factor_arbitro(nombre: str | None) -> tuple[float, str]:
    """Factor de rigurosidad del arbitro via EstilosModel (observado/esperado).

    Los arbitros internacionales rara vez estan en nuestros datos de clubes; si no
    aparece, devolvemos 1.0 (neutro) y lo avisamos honestamente.
    """
    if not nombre:
        return 1.0, "no especificado -> neutro (1.00)"
    from src.estilos_model import EstilosModel
    df = pd.read_csv(config.DATA_PROC / "partidos.csv", parse_dates=["fecha"])
    est = EstilosModel().entrenar(df)
    if nombre in est.arbitros:
        f = est.arbitros[nombre]
        return f, f"{nombre}: factor {f:.2f} ({est.arbitros_n[nombre]} partidos en datos)"
    return 1.0, f"{nombre}: sin historial en datos de clubes -> neutro (1.00)"


def _formato_cuota(prob: float) -> str:
    """Cuota decimal implicita (1/prob) para contrastar contra cuotas comerciales."""
    return f"{1/prob:5.2f}" if prob > 1e-6 else "  inf"


def correr(nacion_local: str, nacion_visit: str,
           xi_local: list[str] | None, xi_visit: list[str] | None,
           arbitro: str | None, n_sims: int = 10000) -> None:
    # --- 1. Motor Mundialista: ratings de jugadores -> tasas base ---
    df_j = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    from src.enriquecer_xg import cargar_ajuste  # correccion por xG real (si xg_ajuste.csv existe)
    jm = JugadoresModel().entrenar_jugadores(df_j, ajuste_xg=cargar_ajuste())

    # Calibracion internacional (si existe): ancla los goles a la media real del Mundial
    import json
    cal_path = config.DATA_PROC / "calibracion.json"
    if cal_path.exists():
        with open(cal_path, encoding="utf-8") as f:
            jm.aplicar_calibracion(json.load(f))
        msg_cal = f"calibrado al Mundial (base {jm.base_real:.2f} goles/equipo)"
    else:
        msg_cal = "SIN calibrar (corre: python -m src.calibrar_internacional)"

    # Elo de selecciones (columna vertebral del hibrido). Si falla, sigue con jugadores solo.
    try:
        propio = config.DATA_PROC / "elo_propio.json"
        if propio.exists():  # nuestro Elo computado de la historia (preferido, refrescable)
            jm.cargar_elo(json.loads(propio.read_text(encoding="utf-8")))
        else:                # respaldo: Elo scrapeado de eloratings.net
            from src.elo import cargar_elo as _cargar_elo
            jm.cargar_elo(_cargar_elo())
    except Exception as e:
        print(f"  (sin Elo: {type(e).__name__} -> solo modelo de jugadores)")

    # Con imputacion por niveles, una seleccion con pocos (o cero) jugadores en el
    # dataset igual se completa con ratings sombra de su federacion. El motor corre
    # cualquier cruce del planeta.
    xi_l = xi_local or jm.seleccion_probable(nacion_local)
    xi_v = xi_visit or jm.seleccion_probable(nacion_visit)

    pred = jm.predecir_partido_mundial(xi_l, xi_v, nacion_local, nacion_visit)

    # --- 2. Arbitro: factor para FALTAS (StatsBomb intl, shrinkage bayesiano) ---
    #        + factor para TARJETAS (EstilosModel, datos de club -> siempre neutro para intl)
    from src.arbitros_faltas import cargar as _cargar_arb, factor_faltas as _get_factor_faltas
    _datos_arb = _cargar_arb()
    f_faltas, msg_faltas = _get_factor_faltas(arbitro, _datos_arb)

    disc_l = jm.disciplina_seleccion(xi_l, nacion_local, factor_faltas=f_faltas)
    disc_v = jm.disciplina_seleccion(xi_v, nacion_visit, factor_faltas=f_faltas)

    # Tarjetas: factor de rigurosidad del arbitro via EstilosModel (clubes).
    # Los arbitros internacionales casi nunca estan en datos de club -> retorna 1.0 (neutro).
    f_arb, msg_arb = _factor_arbitro(arbitro)
    tarj_l = disc_l["tarjetas"] * f_arb
    tarj_v = disc_v["tarjetas"] * f_arb

    # --- Corners: reparto heuristico segun fuerza ofensiva ---
    at_l, at_v = pred["fuerza_local"]["ataque"], pred["fuerza_visitante"]["ataque"]
    corners_l = CORNERS_TOTAL_BASE * at_l / (at_l + at_v)
    corners_v = CORNERS_TOTAL_BASE * at_v / (at_l + at_v)

    parametros = construir_parametros(
        goles_local=pred["goles_esp_local"], goles_visitante=pred["goles_esp_visitante"],
        tarjetas_local=tarj_l, tarjetas_visitante=tarj_v,
        faltas_local=disc_l["faltas"], faltas_visitante=disc_v["faltas"],
        corners_local=corners_l, corners_visitante=corners_v,
    )

    # --- 3. Montecarlo: 10.000 universos minuto a minuto ---
    res = SimuladorMontecarlo().simular_partido(parametros, n_simulaciones=n_sims)

    res["msg_faltas_arbitro"] = msg_faltas  # para Telegram y log
    _reporte(nacion_local, nacion_visit, xi_l, xi_v, pred, parametros, msg_arb, msg_cal, msg_faltas, res)
    return res  # para que el validador pueda registrar la prediccion


def _reporte(loc, vis, xi_l, xi_v, pred, params, msg_arb, msg_cal, msg_faltas, res) -> None:
    L = f"{loc}"; V = f"{vis}"
    print("=" * 60)
    print(f"  D-SOCCER | MOTOR MUNDIALISTA + MONTECARLO ({res['n']:,} sims)")
    print(f"  {L} vs {V}")
    print("=" * 60)
    fl, fv = pred["fuerza_local"], pred["fuerza_visitante"]
    print(f"  XI {L}: {fl['reales']} reales + {fl['sombra']} sombra  "
          f"(ataque {fl['ataque']:.2f} / defensa {fl['defensa']:.2f})")
    print(f"  XI {V}: {fv['reales']} reales + {fv['sombra']} sombra  "
          f"(ataque {fv['ataque']:.2f} / defensa {fv['defensa']:.2f})")
    print(f"  Arbitro (tarjetas): {msg_arb}")
    print(f"  Arbitro (faltas):   {msg_faltas}")
    print(f"  Mapeo a goles: {msg_cal}")
    h = pred.get("hibrido", {})
    if h.get("elo"):
        print(f"  Hibrido (w={h['w']}): Elo {h['elo'][0]:.2f}-{h['elo'][1]:.2f}  |  "
              f"jugadores {h['player'][0]:.2f}-{h['player'][1]:.2f}")
    print(f"  Tasas base -> goles {tuple(round(x,2) for x in params['goles'])} | "
          f"faltas {tuple(round(x,1) for x in params['faltas'])} | "
          f"tarjetas {tuple(round(x,2) for x in params['tarjetas'])}")
    print("  [!] PRIOR sin validar (selecciones; cubre 10 ligas, no todas las del mundo)\n")

    print("  -- 1X2 (prob | cuota implicita) ---------------------------")
    print(f"     Gana {L:<10} {res['prob_local']*100:5.1f}%   cuota {_formato_cuota(res['prob_local'])}")
    print(f"     Empate{'':<10} {res['prob_empate']*100:5.1f}%   cuota {_formato_cuota(res['prob_empate'])}")
    print(f"     Gana {V:<10} {res['prob_visitante']*100:5.1f}%   cuota {_formato_cuota(res['prob_visitante'])}")

    print("\n  -- Marcadores exactos mas repetidos -----------------------")
    for (a, b), p in res["marcadores_top"]:
        print(f"     {a}-{b}   {p*100:5.1f}%   cuota {_formato_cuota(p)}")

    print("\n  -- Over/Under (prob over | cuota over) ---------------------")
    filas = [
        ("Goles 1.5", res["over_1_5_goles"]), ("Goles 2.5", res["over_2_5_goles"]),
        ("Goles 3.5", res["over_3_5_goles"]),
        ("Faltas 24.5", res["over_24_5_faltas"]), ("Faltas 27.5", res["over_27_5_faltas"]),
        ("Faltas 30.5", res["over_30_5_faltas"]),
        ("Tarjetas 3.5", res["over_3_5_tarjetas"]), ("Tarjetas 4.5", res["over_4_5_tarjetas"]),
        ("Corners 8.5", res["over_8_5_corners"]), ("Corners 9.5", res["over_9_5_corners"]),
        ("Corners 10.5", res["over_10_5_corners"]),
    ]
    for nombre, p in filas:
        print(f"     {nombre:<14} over {p*100:5.1f}%  (cuota {_formato_cuota(p)}) | "
              f"under {(1-p)*100:5.1f}%  (cuota {_formato_cuota(1-p)})")
    print(f"\n     Prob. de alguna roja: {res['prob_alguna_roja']*100:.1f}%   "
          f"Goles esp.: {res['goles_esp'][0]:.2f}-{res['goles_esp'][1]:.2f}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Motor Mundialista + Montecarlo de D-Soccer")
    ap.add_argument("local", help="codigo de nacion local (ej. ARG)")
    ap.add_argument("visitante", help="codigo de nacion visitante (ej. FRA)")
    ap.add_argument("--xi-local", help="11 titulares separados por coma (opcional)")
    ap.add_argument("--xi-visitante", help="11 titulares separados por coma (opcional)")
    ap.add_argument("--arbitro", help="nombre del arbitro (opcional)")
    ap.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args()

    xi_l = [s.strip() for s in args.xi_local.split(",")] if args.xi_local else None
    xi_v = [s.strip() for s in args.xi_visitante.split(",")] if args.xi_visitante else None
    correr(args.local, args.visitante, xi_l, xi_v, args.arbitro, args.sims)


if __name__ == "__main__":
    main()
