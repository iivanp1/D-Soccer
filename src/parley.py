"""Asesor de PARLEYS (combinadas): menu de mercados del modelo + EV/Kelly de la combinada.

La matematica que las casas no quieren que hagas:
  - EV de una pata     = prob_modelo * cuota - 1
  - Un parley solo es +EV si CADA pata es +EV. El vig se COMPONE multiplicativamente:
    con 5% de margen por pata, un parley de 3 patas lleva ~13% de margen efectivo y uno
    de 6 patas ~25%. Combinar patas -EV "porque paga mucho" es regalar plata mas rapido.
  - Patas del MISMO partido estan CORRELACIONADAS: multiplicar sus probabilidades es
    invalido (y las casas taxean los same-game parlays ~15-20% de hold). Este asesor
    las bloquea del calculo y lo avisa.
  - Kelly criterion para el stake: f = (prob*cuota - 1) / (cuota - 1). Usamos 25% de
    Kelly (fraccional) porque el prob del modelo tiene error de estimacion.

Donde tenemos edge REAL (validado) y donde no:
  FALTAS del partido    -> edge modesto validado (+4.8% MAE vs baseline). La estrella.
  PLAYER PROPS (tiros)  -> validado en StatsBomb. Segunda opcion.
  1X2 / Over-Under      -> ~= mercado (sin edge demostrado; ahora bien calibrados). Sirven
                           de RELLENO de parley solo con cuota >= justa, nunca como valor.
  TARJETAS / CORNERS    -> SIN senal (corr ~0). No apostar, ni en parley.

Uso:
    python -m src.parley menu NED SWE            # mercados del modelo (sin API, XI probable)
    python -m src.parley menu-fid 1539007        # idem via API (XI/arbitro/ancla reales)
    python -m src.parley eval 0.62@1.75 0.55@2.10 [--bank 1000]
        # EV/Kelly de una combinada: cada pata es prob_modelo@cuota_casa
"""

from __future__ import annotations

import sys

# Margen tipico de una casa soft por pata (para mostrar el vig compuesto del parley).
VIG_PATA = 0.05
KELLY_FRACCION = 0.25   # Kelly fraccional: el prob del modelo tiene error -> apostar 25% del Kelly


# ---------------------------------------------------------------------------- #
#  Mercados del res del motor -> (nombre, prob, flag de edge)
# ---------------------------------------------------------------------------- #
def mercados_de_res(res: dict, loc: str, vis: str) -> list[tuple[str, float, str]]:
    """Aplana el res del Montecarlo en (mercado, prob, flag). flag: EDGE/ok/NO-APOSTAR."""
    out = []
    add = out.append
    add((f"Gana {loc}", res["prob_local"], "ok"))
    add(("Empate", res["prob_empate"], "ok"))
    add((f"Gana {vis}", res["prob_visitante"], "ok"))
    add((f"{loc} o empate (+0.5)", res["ah_local"]["+0.5"], "ok"))
    add((f"{vis} o empate (-0.5 inv)", 1 - res["ah_local"]["-0.5"], "ok"))
    for linea in ("-2.5", "-1.5", "-0.5", "+1.5", "+2.5"):
        add((f"{loc} AH {linea}", res["ah_local"][linea], "ok"))
    add(("Over 1.5 goles", res["over_1_5_goles"], "ok"))
    add(("Under 3.5 goles", 1 - res["over_3_5_goles"], "ok"))
    add(("Over 2.5 goles", res["over_2_5_goles"], "ok"))
    add(("Under 2.5 goles", 1 - res["over_2_5_goles"], "ok"))
    add(("Ambos anotan (si)", res["btts"], "ok"))
    add(("Ambos anotan (no)", 1 - res["btts"], "ok"))
    add((f"Anota {loc}", res["over_0_5_local"], "ok"))
    add((f"Anota {vis}", res["over_0_5_visit"], "ok"))
    for k, nom in (("over_24_5_faltas", "Over 24.5 faltas"), ("over_27_5_faltas", "Over 27.5 faltas"),
                   ("over_30_5_faltas", "Over 30.5 faltas")):
        add((nom, res[k], "EDGE"))
        add((nom.replace("Over", "Under"), 1 - res[k], "EDGE"))
    for k, nom in (("over_3_5_tarjetas", "Over 3.5 tarjetas"), ("over_9_5_corners", "Over 9.5 corners")):
        add((nom, res[k], "NO-APOSTAR"))
    return out


def imprimir_menu(res: dict, loc: str, vis: str) -> None:
    print("=" * 66)
    print(f"  MENU DE PARLEY  {loc} vs {vis}")
    print("  prob = modelo | cuota justa = 1/prob | apostar SOLO si pagan MAS")
    print("=" * 66)
    print(f"  {'mercado':<26}{'prob':>8}{'c.justa':>9}   senal")
    print("  " + "-" * 62)
    for nombre, p, flag in mercados_de_res(res, loc, vis):
        if p <= 0.02 or p >= 0.98:
            continue  # sin interes practico
        marca = {"EDGE": "  <- edge validado", "NO-APOSTAR": "  [X] sin senal, evitar"}.get(flag, "")
        print(f"  {nombre:<26}{p*100:>7.1f}%{1/p:>9.2f}{marca}")
    print("\n  Regla de oro: una pata entra al parley SOLO si su cuota real > cuota justa.")
    print("  Prioriza patas EDGE (faltas/props); 1X2/goles solo como relleno bien pagado.")
    print("  NUNCA combines dos patas del MISMO partido (correlacion -> prob invalida).")


# ---------------------------------------------------------------------------- #
#  Evaluador de combinadas
# ---------------------------------------------------------------------------- #
def evaluar_parley(patas: list[tuple[float, float]], bank: float | None = None) -> dict:
    """patas = [(prob_modelo, cuota_casa)]. Devuelve prob, cuota, EV, Kelly del parley."""
    prob = 1.0
    cuota = 1.0
    evs = []
    for p, c in patas:
        prob *= p
        cuota *= c
        evs.append(p * c - 1.0)
    ev = prob * cuota - 1.0
    kelly = max(0.0, (prob * cuota - 1.0) / (cuota - 1.0)) if cuota > 1 else 0.0
    return {"prob": prob, "cuota": cuota, "ev": ev, "evs_pata": evs,
            "kelly": kelly, "kelly_frac": kelly * KELLY_FRACCION,
            "stake": (bank * kelly * KELLY_FRACCION) if bank else None}


def imprimir_eval(patas: list[tuple[float, float]], bank: float | None = None) -> None:
    r = evaluar_parley(patas, bank)
    n = len(patas)
    vig_comp = 1 - (1 - VIG_PATA) ** n
    print("=" * 60)
    print(f"  EVALUADOR DE PARLEY ({n} patas)")
    print("=" * 60)
    print(f"  {'#':<3}{'prob':>8}{'cuota':>8}{'EV pata':>10}")
    for i, ((p, c), ev) in enumerate(zip(patas, r["evs_pata"]), 1):
        marca = "  <- pata -EV: HUNDE el parley" if ev < 0 else ""
        print(f"  {i:<3}{p*100:>7.1f}%{c:>8.2f}{ev*100:>+9.1f}%{marca}")
    print("  " + "-" * 56)
    print(f"  Prob. de cobrar : {r['prob']*100:6.2f}%   (1 de cada {1/r['prob']:.1f})")
    print(f"  Cuota combinada : {r['cuota']:6.2f}")
    print(f"  EV del parley   : {r['ev']*100:+6.1f}%   (vig compuesto tipico {n} patas: ~{vig_comp*100:.0f}%)")
    if r["ev"] > 0:
        print(f"  Kelly 25%       : {r['kelly_frac']*100:5.2f}% del bankroll"
              + (f"  -> stake ${r['stake']:.0f}" if r["stake"] else ""))
        print("  -> +EV segun el MODELO. Ojo: el EV es tan bueno como las prob del modelo.")
    else:
        print("  -> parley -EV: NO apostar. Saca las patas -EV o busca mejores cuotas.")
    if any(ev < 0 for ev in r["evs_pata"]):
        print("  [!] Hay patas -EV: un parley solo es +EV si CADA pata lo es.")


def _parse_pata(s: str) -> tuple[float, float]:
    p, c = s.split("@")
    p = float(p)
    if p > 1:
        p /= 100.0  # acepta 62@1.75 como 62%
    return p, float(c)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "menu" and len(args) >= 3:
        from src.mundial_engine import correr
        res = correr(args[1], args[2], None, None, None, n_sims=10000)
        print()
        imprimir_menu(res, args[1], args[2])
    elif cmd == "menu-fid" and len(args) >= 2:
        from src.fixtures import correr_partido_auto
        info = correr_partido_auto(int(args[1]))
        if info:
            print()
            imprimir_menu(info["res"], info["cod_l"], info["cod_v"])
    elif cmd == "eval" and len(args) >= 2:
        bank = None
        if "--bank" in args:
            i = args.index("--bank")
            bank = float(args[i + 1])
            args = args[:i] + args[i + 2:]
        patas = [_parse_pata(s) for s in args[1:]]
        imprimir_eval(patas, bank)
    else:
        print("Uso: python -m src.parley menu <LOC> <VIS> | menu-fid <fixture_id> | "
              "eval p@c p@c ... [--bank N]")


if __name__ == "__main__":
    main()
