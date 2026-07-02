"""Detector de VALOR: compara las probabilidades del modelo contra las cuotas reales
de las casas (API-Football) y calcula el Valor Esperado (EV) de cada apuesta.

  EV = prob_modelo * cuota_ofrecida - 1      ( >0 = apuesta con valor )

"Donde si y donde no" es ESTO: una formula, no una IA. La IA, si la sumamos, explica y
marca contexto, pero la DECISION es este calculo deterministico y transparente.

OJO HONESTO: el detector es tan bueno como el modelo. Hoy el Motor Mundialista todavia
sobre-valora a los debiles (sub-diferenciacion residual), asi que el "valor" que marque a
favor de un underdog suele ser NUESTRO error, no valor real. Se vuelve confiable cuando el
modelo este tuneado/validado con los datos del server. Para clubes (Dixon-Coles validado)
ya es util de una.

Uso:
    python -m src.valor <fixture_id>      # ej: 1489369 (Mexico-Sudafrica)
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from src.fixtures import _api_get

UMBRAL_VALOR = 0.05  # EV minimo para marcar una apuesta como "con valor" (5%)
SHARP_ID = 4         # Pinnacle: casa SHARP de referencia (linea afilada, margen bajo)

# Rango futbolistico de seguridad para los lambda deducidos (clamp anti-outlier).
_LAM_MIN, _LAM_MAX = 0.10, 6.0


def _demargin(odds: list[float | None]) -> list[float] | None:
    """De-margina un set de cuotas (1X2 o O/U) -> probabilidades 'justas' que suman 1.

    Quita la sobre-ronda (vig) de UNA casa normalizando las implicitas. None si falta
    alguna cuota. Es el metodo proporcional (basico); para Pinnacle, de margen chico, es
    suficiente. La prob de-marginada de la sharp es la mejor estimacion de 'verdad' que
    tenemos, y la vara honesta contra la que medir el edge del modelo.
    """
    # 'not (o > 0)' atrapa None, <=0 Y NaN (NaN>0 es False) en un solo chequeo.
    if not odds or any(o is None or not (o > 0) for o in odds):
        return None
    imp = [1.0 / o for o in odds]
    s = sum(imp)
    return [round(i / s, 4) for i in imp]


def cuotas_mercado(fixture_id: int) -> dict | None:
    """Cuotas de mercado de un fixture, con la SHARP (Pinnacle) aislada y de-marginada.

    Por mercado (Home/Draw/Away/Over/Under) devuelve:
      - mejor          : la cuota que MAS paga (para EJECUTAR la apuesta = line shopping)
      - pinnacle       : la cuota de Pinnacle (la sharp), o None si no la ofrece
      - prob_implicita : implicita promedio entre casas (con margen, solo informativa)
    Y a nivel partido:
      - sharp_1x2 : [p_local, p_empate, p_visit] de-marginadas de Pinnacle (la 'verdad')
      - sharp_ou  : [p_over, p_under] de-marginadas de Pinnacle
      - n_casas

    OJO con el sesgo de seleccion: tomar max(cuotas) de N casas infla el EV aparente (la
    casa outlier suele ser la lenta o la que sabe algo). Por eso el detector compara el
    modelo contra sharp_*, no contra 'mejor'. 'mejor' solo dice donde ejecutar.
    """
    resp = _api_get("odds", {"fixture": fixture_id})
    if not resp:
        return None
    bms = resp[0].get("bookmakers", [])

    cuotas = {"Home": [], "Draw": [], "Away": [], "Over": [], "Under": []}
    pin = {"Home": None, "Draw": None, "Away": None, "Over": None, "Under": None}
    for b in bms:
        es_pin = b.get("id") == SHARP_ID
        for bet in b.get("bets", []):
            nombre = bet["name"].lower()
            if nombre == "match winner":
                for v in bet["values"]:
                    if v["value"] in ("Home", "Draw", "Away"):
                        o = float(v["odd"]); cuotas[v["value"]].append(o)
                        if es_pin:
                            pin[v["value"]] = o
            elif "goals over/under" in nombre and "half" not in nombre:
                for v in bet["values"]:
                    if v["value"] == "Over 2.5":
                        o = float(v["odd"]); cuotas["Over"].append(o)
                        if es_pin:
                            pin["Over"] = o
                    elif v["value"] == "Under 2.5":
                        o = float(v["odd"]); cuotas["Under"].append(o)
                        if es_pin:
                            pin["Under"] = o

    def resumen(k):
        lista = cuotas[k]
        if not lista:
            return None
        return {"mejor": max(lista), "pinnacle": pin[k],
                "prob_implicita": sum(1 / o for o in lista) / len(lista)}

    out = {k: resumen(k) for k in cuotas}
    out["n_casas"] = len(bms)
    out["sharp_1x2"] = _demargin([pin["Home"], pin["Draw"], pin["Away"]])
    out["sharp_ou"] = _demargin([pin["Over"], pin["Under"]])
    return out


# =========================================================================== #
#  INGENIERIA INVERSA DESDE PINNACLE  (EL ANCLA)
# =========================================================================== #
#  En vez de calcular los goles desde cero, deducimos el lambda (goles esperados)
#  que el mercado sharp tiene IMPLICITO en sus cuotas, y lo usamos como ancla. El
#  1X2 da el REPARTO (quien es favorito); el Over/Under da la MAGNITUD (cuantos
#  goles). Se invierte resolviendo el problema inverso de Poisson por minimos
#  cuadrados (no hay forma cerrada).
# =========================================================================== #
def _matriz_kn(lam_h: float, lam_a: float, frac3: float, max_goals: int = 10) -> np.ndarray:
    """Matriz de marcadores bajo bivariate Poisson Karlis-Ntzoufras (X=W1+W3, Y=W2+W3).

    lambda3 = frac3 * min(lam_h, lam_a); las MARGINALES se preservan. frac3=0 -> Poisson
    independiente clasico. Es el MISMO modelo generativo que simula el Montecarlo (frac3),
    asi inversion y simulacion son consistentes (sin doble correccion).
    """
    g = np.arange(max_goals + 1)
    l3 = frac3 * min(lam_h, lam_a)
    l1, l2 = max(lam_h - l3, 1e-9), max(lam_a - l3, 1e-9)
    p1, p2, p3 = poisson.pmf(g, l1), poisson.pmf(g, l2), poisson.pmf(g, l3)
    m = np.zeros((max_goals + 1, max_goals + 1))
    for k in range(max_goals + 1):
        if p3[k] < 1e-12:
            break
        m[k:, k:] += p3[k] * np.outer(p1[:max_goals + 1 - k], p2[:max_goals + 1 - k])
    return m / m.sum()


def invertir_pinnacle_1x2(sharp_1x2: list[float] | None, max_goals: int = 10,
                          frac3: float | None = None) -> tuple[float, float] | None:
    """Deduce (lam_local, lam_visitante) implicitos del 1X2 de-marginado de Pinnacle.

    Minimiza el MSE entre el 1X2 del modelo generativo (bivariate Poisson KN, el mismo que
    simula el motor) y las tres probabilidades de Pinnacle. Devuelve None si el input es
    invalido. frac3=None -> usa config.FRAC3_GOLES_COMUNES (produccion).
    """
    if not sharp_1x2 or len(sharp_1x2) != 3 or any(p is None for p in sharp_1x2):
        return None
    s = float(sum(sharp_1x2))
    if s <= 0:
        return None
    if frac3 is None:
        from src import config
        frac3 = config.FRAC3_GOLES_COMUNES
    p_h, p_d, p_a = (p / s for p in sharp_1x2)  # re-normaliza por las dudas

    def _loss(log_lams: np.ndarray) -> float:
        lam_h, lam_a = np.exp(log_lams)
        m = _matriz_kn(lam_h, lam_a, frac3, max_goals)
        ph, pd_, pa = float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())
        return (ph - p_h) ** 2 + (pd_ - p_d) ** 2 + (pa - p_a) ** 2

    r = minimize(_loss, np.log([1.3, 1.1]), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 2000})
    lam_h, lam_a = np.exp(r.x)
    return (float(np.clip(lam_h, _LAM_MIN, _LAM_MAX)),
            float(np.clip(lam_a, _LAM_MIN, _LAM_MAX)))


def invertir_pinnacle_ou(p_over: float | None, linea: float = 2.5) -> float | None:
    """Deduce el lambda_TOTAL implicito de la prob Over de-marginada de Pinnacle.

    Under = (total de goles <= floor(linea)). Resuelve 1 - CDF_Poisson(k; lam) = p_over.
    (Se usa solo como semilla del re-escalado KN de lambda_pinnacle; el ajuste final del
    total se hace bajo el modelo KN completo para consistencia con la simulacion.)
    """
    if p_over is None or not (0.0 < p_over < 1.0):
        return None
    k = int(np.floor(linea))

    def _loss(lam: np.ndarray) -> float:
        l = max(float(lam[0]), 1e-6)
        return (1.0 - float(poisson.cdf(k, l)) - p_over) ** 2

    r = minimize(_loss, [2.4], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 1000})
    return float(np.clip(r.x[0], 2 * _LAM_MIN, 2 * _LAM_MAX))


def lambda_pinnacle(mercado: dict | None, frac3: float | None = None) -> tuple[float, float] | None:
    """De la salida de cuotas_mercado() deduce el ancla (lam_local, lam_visit) de Pinnacle.

    El 1X2 da el REPARTO local/visita; si hay O/U, se re-escala el total para que el modelo
    KN reproduzca la prob Over del mercado (busqueda 1-D bajo el MISMO modelo generativo que
    simula el motor -> sin doble correccion). None si no hay 1X2 sharp.
    """
    if not mercado:
        return None
    if frac3 is None:
        from src import config
        frac3 = config.FRAC3_GOLES_COMUNES
    lam = invertir_pinnacle_1x2(mercado.get("sharp_1x2"), frac3=frac3)
    if lam is None:
        return None
    lam_h, lam_a = lam
    sou = mercado.get("sharp_ou")
    if sou and sou[0] and 0.0 < sou[0] < 1.0:
        p_over = sou[0]

        def _loss_esc(le: np.ndarray) -> float:
            s = float(np.exp(le[0]))
            m = _matriz_kn(lam_h * s, lam_a * s, frac3)
            p_under = float(sum(m[i, j] for i in range(3) for j in range(3) if i + j <= 2))
            return (1.0 - p_under - p_over) ** 2

        r = minimize(_loss_esc, [0.0], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 500})
        esc = float(np.exp(r.x[0]))
        if 0.5 < esc < 2.0:  # sanity: si el O/U pide algo absurdo, quedarse con el 1X2
            lam_h, lam_a = lam_h * esc, lam_a * esc
    return round(lam_h, 3), round(lam_a, 3)


def _ev(prob_modelo: float, cuota: float) -> float:
    return prob_modelo * cuota - 1.0


def analizar_valor(fixture_id: int) -> None:
    from src.fixtures import correr_partido_auto

    # Una sola consulta a /odds: deducimos el ancla de Pinnacle y la reusamos para el motor
    # (evita gastar cuota dos veces) y para la tabla de EV.
    mercado = cuotas_mercado(fixture_id)
    ancla = lambda_pinnacle(mercado) if mercado else None

    # anclar=False: ya calculamos el ancla de la misma consulta de cuotas (no re-fetch).
    info = correr_partido_auto(fixture_id, ancla=ancla, anclar=False)
    if info is None:
        return
    res = info["res"]
    if not mercado:
        print("\nNo hay cuotas disponibles para este fixture.")
        return
    if res.get("msg_ancla"):
        print(f"\n  Ancla de mercado -> {res['msg_ancla']}")

    sharp = mercado.get("sharp_1x2")
    sharp_ou = mercado.get("sharp_ou")
    hay_sharp = sharp is not None

    print(f"\n{'='*72}")
    print(f"  DETECTOR DE VALOR  ({mercado['n_casas']} casas | sharp: "
          f"{'Pinnacle' if hay_sharp else 'NO DISPONIBLE'})")
    print(f"{'='*72}")
    if not hay_sharp:
        print("  [!] Pinnacle no cotiza este partido -> sin vara sharp, el EV no es confiable.")
    print(f"  {'Mercado':<13}{'modelo':>8}{'sharp':>8}{'best':>7}{'EV-mod':>9}{'EV-sharp':>10}")
    print("  " + "-" * 70)

    # (nombre, p_modelo, p_sharp_demarginada, market_dict)
    filas = [
        ("Gana local", res["prob_local"], sharp[0] if sharp else None, mercado.get("Home")),
        ("Empate", res["prob_empate"], sharp[1] if sharp else None, mercado.get("Draw")),
        ("Gana visita", res["prob_visitante"], sharp[2] if sharp else None, mercado.get("Away")),
        ("Over 2.5", res["over_2_5_goles"], sharp_ou[0] if sharp_ou else None, mercado.get("Over")),
        ("Under 2.5", 1 - res["over_2_5_goles"], sharp_ou[1] if sharp_ou else None, mercado.get("Under")),
    ]
    edge_modelo, line_shop = [], []
    for nombre, p_mod, p_sharp, mkt in filas:
        if not mkt:
            continue
        best = mkt["mejor"]
        ev_mod = _ev(p_mod, best)
        ev_sharp = _ev(p_sharp, best) if p_sharp else None
        ss = f"{p_sharp*100:7.1f}%" if p_sharp else "    s/d"
        es = f"{ev_sharp*100:>+9.1f}%" if ev_sharp is not None else "      s/d"
        # EDGE MODELO: el modelo discrepa de la sharp HACIA ARRIBA y la mejor cuota paga.
        es_edge = p_sharp is not None and p_mod > p_sharp and ev_mod > UMBRAL_VALOR
        # LINE SHOPPING: la mejor cuota esta mal preciada vs Pinnacle (no depende del modelo).
        es_line = ev_sharp is not None and ev_sharp > 0
        marca = ("  <-EDGE" if es_edge else "") + (" <-LINE" if es_line else "")
        print(f"  {nombre:<13}{p_mod*100:7.1f}%{ss}{best:>7.2f}{ev_mod*100:>+8.1f}%{es}{marca}")
        if es_edge:
            edge_modelo.append((nombre, ev_mod, best, p_mod - p_sharp))
        if es_line:
            line_shop.append((nombre, ev_sharp, best))

    print()
    if edge_modelo:
        print("  EDGE DEL MODELO (modelo > sharp de-marginada, ordenado por EV):")
        for nombre, ev, cuota, dp in sorted(edge_modelo, key=lambda x: -x[1]):
            print(f"     {nombre:<13} EV {ev*100:+.1f}%  @ {cuota:.2f}  ({dp*100:+.1f}pp vs sharp)")
        print("  [!] Asume que el modelo le gana a Pinnacle. Para SELECCIONES es una HIPOTESIS")
        print("      sin validar -> tentativo, no dinero seguro. Confirmar con CLV (validacion).")
    else:
        print("  Sin edge del modelo vs la sharp. (Lo esperable: Pinnacle es dificil de batir.)")
    if line_shop:
        print("\n  LINE SHOPPING (mejor cuota mal preciada vs Pinnacle, NO depende del modelo):")
        for nombre, ev, cuota in sorted(line_shop, key=lambda x: -x[1]):
            print(f"     {nombre:<13} +{ev*100:.1f}% vs sharp  @ {cuota:.2f}")


def _self_test_ancla() -> None:
    """Prueba OFFLINE (sin API) la ingenieria inversa: round-trip lambda -> 1X2 -> lambda.

    Genera el 1X2/OU del modelo KN (frac3 de produccion) con un lambda CONOCIDO y verifica
    que la inversion lo recupere. Si el error es chico, la matematica del ancla es correcta
    Y la inversion es consistente con el modelo que simula el motor.
    """
    from src import config
    frac3 = config.FRAC3_GOLES_COMUNES
    print(f"=== SELF-TEST ancla Pinnacle (round-trip lambda, KN frac3={frac3}) ===\n")
    casos = [(1.80, 1.00), (1.35, 1.35), (2.40, 0.70), (1.10, 1.60)]
    ok = True
    for lam_h, lam_a in casos:
        m = _matriz_kn(lam_h, lam_a, frac3)
        p_h, p_d, p_a = float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())
        rec = invertir_pinnacle_1x2([p_h, p_d, p_a], frac3=frac3)
        err = max(abs(rec[0] - lam_h), abs(rec[1] - lam_a))
        # Round-trip completo via lambda_pinnacle (1X2 + OU sintetico del mismo modelo).
        p_under = float(sum(m[i, j] for i in range(3) for j in range(3) if i + j <= 2))
        merc = {"sharp_1x2": [p_h, p_d, p_a], "sharp_ou": [1 - p_under, p_under]}
        lp = lambda_pinnacle(merc, frac3=frac3)
        err_lp = max(abs(lp[0] - lam_h), abs(lp[1] - lam_a))
        flag = "OK " if (err < 0.02 and err_lp < 0.03) else "XX "
        ok = ok and flag == "OK "
        print(f"  {flag} real ({lam_h:.2f},{lam_a:.2f})  1X2[{p_h:.3f}/{p_d:.3f}/{p_a:.3f}]"
              f"  -> 1x2 ({rec[0]:.3f},{rec[1]:.3f}) err {err:.4f}"
              f"  | +OU ({lp[0]:.3f},{lp[1]:.3f}) err {err_lp:.4f}")
    # Mezcla (interpolacion del motor) a modo ilustrativo.
    alpha = 0.35
    lam_mod, lam_pin = (1.20, 1.20), (1.85, 0.95)
    mix = tuple(alpha * p + (1 - alpha) * mo for p, mo in zip(lam_pin, lam_mod))
    print(f"\n  Interpolacion alpha={alpha}: modelo {lam_mod} + Pinnacle {lam_pin} "
          f"-> hibrido ({mix[0]:.3f},{mix[1]:.3f})")
    print(f"\n  Resultado: {'TODOS OK' if ok else 'HAY FALLOS'}")


def main() -> None:
    if "--test-ancla" in sys.argv:
        _self_test_ancla()
        return
    if len(sys.argv) < 2:
        print("Uso: python -m src.valor <fixture_id>   |   python -m src.valor --test-ancla")
        return
    analizar_valor(int(sys.argv[1]))


if __name__ == "__main__":
    main()
