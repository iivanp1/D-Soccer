"""Registro de FORMA: como llegan las selecciones (ultimos partidos + perfil del plantel).

Que muestra por seleccion:
  1. FORMA RECIENTE (results.csv, gratis, 49k partidos hasta hoy): ultimos N partidos
     internacionales con W/D/L, goles a favor/en contra, % over 2.5, % ambos anotan, racha.
  2. PERFIL DEL PLANTEL (jugadores.csv + XI probable): expectativa de disparo del equipo,
     punteria (conversion goles/90 del XI), propension a faltas y tarjetas.

HONESTIDAD ESTADISTICA (que dice la literatura):
  - La forma W/D/L reciente es un predictor DEBIL (correlacion ~0.15-0.25 con el resultado
    siguiente; mucho ruido y regresion a la media). Este registro es CONTEXTO, no senal.
  - Lo que si predice: la calidad del plantel (nuestro Elo+jugadores) y los tiros/xG
    recientes. El numero duro para apostar sigue siendo el del motor (mundial_engine).

Uso:
    python -m src.forma NED SWE      # registro comparado de ambas
    python -m src.forma ARG          # una sola
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src import config
from src.fixtures import PAIS_API_A_CODIGO
from src.jugadores_model import _norm

RESULTS = config.DATA_RAW / "results.csv"
N_PARTIDOS = 8


def _nombres_de(codigo: str) -> set[str]:
    """Nombres (normalizados) que results.csv puede usar para un codigo FBref."""
    return {n for n, c in PAIS_API_A_CODIGO.items() if c == codigo}


def forma_reciente(codigo: str, n: int = N_PARTIDOS) -> dict | None:
    """Ultimos n partidos internacionales CON resultado del pais (results.csv)."""
    if not RESULTS.exists():
        return None
    df = pd.read_csv(RESULTS).dropna(subset=["home_score", "away_score"])
    nombres = _nombres_de(codigo)
    es_local = df["home_team"].map(lambda x: _norm(x) in nombres)
    es_visit = df["away_team"].map(lambda x: _norm(x) in nombres)
    sub = df[es_local | es_visit].sort_values("date").tail(n)
    if sub.empty:
        return None

    filas, w = [], 0
    gf_t = gc_t = overs = btts = 0
    for _, r in sub.iterrows():
        local = _norm(r["home_team"]) in nombres
        gf, gc = (r["home_score"], r["away_score"]) if local else (r["away_score"], r["home_score"])
        gf, gc = int(gf), int(gc)
        rival = r["away_team"] if local else r["home_team"]
        res = "W" if gf > gc else ("D" if gf == gc else "L")
        w += res == "W"
        gf_t += gf; gc_t += gc
        overs += (gf + gc) > 2.5
        btts += gf > 0 and gc > 0
        filas.append({"fecha": r["date"], "rival": rival, "gf": gf, "gc": gc, "res": res,
                      "torneo": r["tournament"]})
    racha = "".join(f["res"] for f in filas)
    return {"partidos": filas, "racha": racha, "w": w, "n": len(filas),
            "gf": gf_t, "gc": gc_t, "over_pct": overs / len(filas), "btts_pct": btts / len(filas)}


def perfil_plantel(codigo: str, jm=None, dfj=None) -> dict | None:
    """Expectativa de disparo, punteria y disciplina del XI probable."""
    if jm is None:
        from src.jugadores_model import JugadoresModel
        from src.enriquecer_xg import cargar_ajuste
        dfj = pd.read_csv(config.DATA_PROC / "jugadores.csv")
        jm = JugadoresModel().entrenar_jugadores(dfj, ajuste_xg=cargar_ajuste())
    xi = jm.seleccion_probable(codigo)
    if not xi:
        return None
    idx = [_norm(p) for p in xi]
    sub = jm.jugadores[jm.jugadores.index.isin(idx)]
    disc = jm.disciplina_seleccion(xi, codigo, factor_faltas=config.ESCALA_FALTAS_WC)
    # Punteria del XI en datos de club: goles sin penal vs tiros al arco (agregado)
    d = dfj[dfj["player"].map(lambda x: _norm(str(x))).isin(idx)]
    sot = d["tiros_arco"].sum()
    npg = d["goles_sin_pen"].sum()
    return {
        "xi": list(sub["nombre"]) if "nombre" in sub.columns else xi,
        "n_reales": len(sub),
        "ataque": float(sub["ofensivo"].sum()),
        "defensa": float(sub["defensivo"].sum()),
        "faltas_esp": disc["faltas"],
        "tarjetas_esp": disc["tarjetas"],
        "sot_total": float(sot), "conversion": float(npg / sot) if sot > 0 else None,
    }


def imprimir_registro(codigo: str, jm=None, dfj=None) -> None:
    print("=" * 64)
    print(f"  REGISTRO DE FORMA: {codigo}")
    print("=" * 64)
    f = forma_reciente(codigo)
    if f:
        print(f"  Ultimos {f['n']}: {f['racha']}  ({f['w']}W {sum(1 for c in f['racha'] if c=='D')}D "
              f"{sum(1 for c in f['racha'] if c=='L')}L) | GF-GC {f['gf']}-{f['gc']}")
        print(f"  Over 2.5 en sus partidos: {f['over_pct']*100:.0f}% | Ambos anotan: {f['btts_pct']*100:.0f}%")
        for p in f["partidos"][-5:]:
            print(f"    {p['fecha']}  {p['res']} {p['gf']}-{p['gc']} vs {p['rival']:<20} ({p['torneo']})")
    else:
        print("  (sin resultados en results.csv; correr python -m src.elo_history --refrescar)")
    pp = perfil_plantel(codigo, jm, dfj)
    if pp:
        print(f"\n  PLANTEL (XI probable, {pp['n_reales']} con datos de club):")
        print(f"    Ataque {pp['ataque']:.2f} | Defensa {pp['defensa']:.2f} | "
              f"Faltas esp {pp['faltas_esp']:.1f} | Tarjetas esp {pp['tarjetas_esp']:.1f}")
        if pp["conversion"] is not None:
            print(f"    Punteria del XI (goles/tiro al arco, clubes): {pp['conversion']*100:.1f}%")
    print("\n  [!] La forma W/D/L reciente es senal DEBIL (lit.: corr ~0.2, mucho ruido).")
    print("      Es contexto para leer el partido; el numero para apostar es el del motor.")


def main() -> None:
    codigos = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not codigos:
        print("Uso: python -m src.forma <COD> [COD2]   (ej: python -m src.forma NED SWE)")
        return
    # Un solo entrenamiento del modelo para todas las selecciones pedidas
    from src.jugadores_model import JugadoresModel
    from src.enriquecer_xg import cargar_ajuste
    dfj = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    jm = JugadoresModel().entrenar_jugadores(dfj, ajuste_xg=cargar_ajuste())
    for c in codigos:
        imprimir_registro(c.upper(), jm, dfj)
        print()


if __name__ == "__main__":
    main()
