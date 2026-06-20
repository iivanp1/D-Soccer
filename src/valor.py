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

from src.fixtures import _api_get

UMBRAL_VALOR = 0.05  # EV minimo para marcar una apuesta como "con valor" (5%)
SHARP_ID = 4         # Pinnacle: casa SHARP de referencia (linea afilada, margen bajo)


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


def _ev(prob_modelo: float, cuota: float) -> float:
    return prob_modelo * cuota - 1.0


def analizar_valor(fixture_id: int) -> None:
    from src.fixtures import correr_partido_auto

    info = correr_partido_auto(fixture_id)
    if info is None:
        return
    res = info["res"]
    mercado = cuotas_mercado(fixture_id)
    if not mercado:
        print("\nNo hay cuotas disponibles para este fixture.")
        return

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


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m src.valor <fixture_id>")
        return
    analizar_valor(int(sys.argv[1]))


if __name__ == "__main__":
    main()
