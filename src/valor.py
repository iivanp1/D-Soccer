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


def cuotas_mercado(fixture_id: int) -> dict | None:
    """Cuotas de mercado de un fixture: mejor cuota ofrecida + probabilidad de consenso.

    Promedia la probabilidad implicita entre todas las casas (consenso) y toma la MEJOR
    cuota disponible (la que mas paga) para el calculo de EV.
    """
    resp = _api_get("odds", {"fixture": fixture_id})
    if not resp:
        return None
    bms = resp[0].get("bookmakers", [])

    cuotas = {"Home": [], "Draw": [], "Away": [], "Over": [], "Under": []}
    for b in bms:
        for bet in b.get("bets", []):
            nombre = bet["name"].lower()
            if nombre == "match winner":
                for v in bet["values"]:
                    if v["value"] in ("Home", "Draw", "Away"):
                        cuotas[v["value"]].append(float(v["odd"]))
            elif "goals over/under" in nombre and "half" not in nombre:
                for v in bet["values"]:
                    if v["value"] == "Over 2.5":
                        cuotas["Over"].append(float(v["odd"]))
                    elif v["value"] == "Under 2.5":
                        cuotas["Under"].append(float(v["odd"]))

    def resumen(lista):
        if not lista:
            return None
        mejor = max(lista)                       # la cuota que mas paga
        prob_imp = sum(1 / o for o in lista) / len(lista)  # implicita promedio (con margen)
        return {"mejor": mejor, "prob_implicita": prob_imp}

    out = {k: resumen(v) for k, v in cuotas.items()}
    out["n_casas"] = len(bms)
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

    print(f"\n{'='*60}")
    print(f"  DETECTOR DE VALOR  ({mercado['n_casas']} casas)")
    print(f"{'='*60}")
    print(f"  {'Mercado':<14}{'modelo':>8}{'mercado':>9}{'mejor cuota':>13}{'EV':>9}")
    print("  " + "-" * 56)

    filas = [
        ("Gana local", res["prob_local"], mercado.get("Home")),
        ("Empate", res["prob_empate"], mercado.get("Draw")),
        ("Gana visita", res["prob_visitante"], mercado.get("Away")),
        ("Over 2.5", res["over_2_5_goles"], mercado.get("Over")),
        ("Under 2.5", 1 - res["over_2_5_goles"], mercado.get("Under")),
    ]
    apuestas_valor = []
    for nombre, p_mod, mkt in filas:
        if not mkt:
            continue
        ev = _ev(p_mod, mkt["mejor"])
        marca = "  <-- VALOR" if ev > UMBRAL_VALOR else ""
        print(f"  {nombre:<14}{p_mod*100:7.1f}%{mkt['prob_implicita']*100:8.1f}%"
              f"{mkt['mejor']:>13.2f}{ev*100:>+8.1f}%{marca}")
        if ev > UMBRAL_VALOR:
            apuestas_valor.append((nombre, ev, mkt["mejor"]))

    print()
    if apuestas_valor:
        apuestas_valor.sort(key=lambda x: -x[1])
        print("  Apuestas con VALOR (segun el modelo, ordenadas por EV):")
        for nombre, ev, cuota in apuestas_valor:
            print(f"     {nombre:<14} EV {ev*100:+.1f}%  @ {cuota:.2f}")
        print("\n  [!] El valor es tan confiable como el modelo. Si marca un underdog y el modelo")
        print("      todavia no esta validado, probablemente sea nuestro sesgo, no valor real.")
    else:
        print("  Sin apuestas con valor (el mercado cubre todo). PASAR.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m src.valor <fixture_id>")
        return
    analizar_valor(int(sys.argv[1]))


if __name__ == "__main__":
    main()
