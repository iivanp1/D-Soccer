"""Simulador de BANKROLL: cuanto tarda (y cuanto duele) convertir X en Y apostando con edge.

La pregunta que responde: "quiero convertir 5.000 en 5.000.000 (1000x), que tan realista es?"
La respuesta depende de SOLO tres numeros:
  1. El EV real por apuesta (edge vs la cuota que te pagan). Los pros sostienen 2-5%.
  2. Cuantas apuestas con edge encontras por semana (VOLUMEN -> por eso clubes > Mundial).
  3. La fraccion de Kelly que apostas (full Kelly = crecimiento maximo pero drawdowns
     brutales; 25-50% de Kelly es lo defendible cuando el edge es estimado con error).

El promedio esconde lo importante: el drawdown y el riesgo de ruina solo aparecen SIMULANDO.

Uso:
    python -m src.bankroll                            # tabla de escenarios (EV 3/5/10%)
    python -m src.bankroll --bank 5000 --meta 5000000 --ev 0.05 --cuota 1.90 \
                           --kelly 0.5 --semana 10 --anos 20
"""

from __future__ import annotations

import sys

import numpy as np


def simular(bank0: float, meta: float, ev: float, cuota: float, kelly_frac: float,
            apuestas_semana: float, anos: float, n_sims: int = 20000,
            semilla: int = 42) -> dict:
    """Simula n_sims trayectorias de bankroll con stake Kelly-fraccional fijo.

    p implicita del edge: EV = p*cuota - 1  ->  p = (1+EV)/cuota.
    Stake por apuesta: f = kelly_frac * kelly_full, con kelly_full = EV/(cuota-1).
    RUINA operativa: si el bank cae bajo el 5% del inicial, se considera quemado
    (no hay stake minimo viable) y deja de operar.
    """
    p = (1.0 + ev) / cuota
    if not (0 < p < 1):
        raise ValueError(f"EV {ev} y cuota {cuota} implican p={p:.3f} (imposible)")
    kelly_full = ev / (cuota - 1.0)
    f = max(0.0, kelly_frac * kelly_full)
    n_bets = int(round(anos * 52 * apuestas_semana))

    rng = np.random.default_rng(semilla)
    b = np.full(n_sims, float(bank0))
    peak = b.copy()
    max_dd = np.zeros(n_sims)
    alcanzo = np.zeros(n_sims, dtype=bool)     # llego a la meta (y se retira)
    quemado = np.zeros(n_sims, dtype=bool)     # cayo bajo el 5% del bank inicial
    t_alcanzo = np.full(n_sims, -1.0)          # en que apuesta alcanzo la meta

    gan = 1.0 + f * (cuota - 1.0)
    per = 1.0 - f
    piso = 0.05 * bank0

    for i in range(n_bets):
        vivo = ~alcanzo & ~quemado
        if not vivo.any():
            break
        win = rng.random(n_sims) < p
        b = np.where(vivo, b * np.where(win, gan, per), b)
        peak = np.maximum(peak, b)
        max_dd = np.maximum(max_dd, 1.0 - b / peak)
        nuevo_meta = vivo & (b >= meta)
        t_alcanzo[nuevo_meta] = i + 1
        alcanzo |= nuevo_meta
        quemado |= vivo & (b < piso)

    hits = t_alcanzo[t_alcanzo > 0]
    return {
        "p": p, "kelly_full": kelly_full, "f": f, "n_bets": n_bets,
        "prob_meta": float(alcanzo.mean()),
        "prob_quemado": float(quemado.mean()),
        "prob_mitad": float((max_dd >= 0.5).mean()),   # sufrio un drawdown >= 50%
        "mediana_final": float(np.median(b)),
        "p90_final": float(np.percentile(b, 90)),
        "anos_mediana_meta": float(np.median(hits) / (52 * apuestas_semana)) if len(hits) else None,
    }


def reporte(bank0, meta, ev, cuota, kelly_frac, semana, anos, n_sims=20000) -> None:
    r = simular(bank0, meta, ev, cuota, kelly_frac, semana, anos, n_sims)
    print("=" * 64)
    print(f"  BANKROLL: {bank0:,.0f} -> meta {meta:,.0f}  ({meta/bank0:,.0f}x)")
    print(f"  EV {ev*100:+.1f}% por apuesta @ {cuota:.2f} (p={r['p']*100:.1f}%) | "
          f"Kelly {kelly_frac*100:.0f}% (stake {r['f']*100:.2f}% del bank)")
    print(f"  {semana:.0f} apuestas/semana durante {anos:.0f} anos ({r['n_bets']:,} apuestas, "
          f"{n_sims:,} simulaciones)")
    print("=" * 64)
    print(f"  P(alcanzar la meta)         : {r['prob_meta']*100:5.1f}%"
          + (f"   (mediana de los que llegan: {r['anos_mediana_meta']:.1f} anos)"
             if r['anos_mediana_meta'] else ""))
    print(f"  P(quemar la banca, <5%)     : {r['prob_quemado']*100:5.1f}%")
    print(f"  P(sufrir drawdown >= 50%)   : {r['prob_mitad']*100:5.1f}%")
    print(f"  Bank final mediano          : {r['mediana_final']:,.0f}")
    print(f"  Bank final p90 (optimista)  : {r['p90_final']:,.0f}")


def escenarios(bank0=5000.0, meta=5_000_000.0) -> None:
    print(f"\n  QUE HACE FALTA PARA {meta/bank0:,.0f}x  (cuota tipica 1.90, Kelly 50%, 20 anos max)\n")
    print(f"  {'EV/apuesta':>11}{'ap/semana':>11}{'P(meta)':>9}{'P(quemado)':>12}{'DD>=50%':>9}{'anos tip.':>10}")
    for ev, semana in [(0.03, 10), (0.05, 10), (0.05, 30), (0.10, 10), (0.10, 30)]:
        r = simular(bank0, meta, ev, 1.90, 0.5, semana, 20, n_sims=8000)
        anos = f"{r['anos_mediana_meta']:.1f}" if r["anos_mediana_meta"] else "  >20"
        print(f"  {ev*100:>10.0f}%{semana:>11.0f}{r['prob_meta']*100:>8.1f}%"
              f"{r['prob_quemado']*100:>11.1f}%{r['prob_mitad']*100:>8.1f}%{anos:>10}")
    print("""
  LECTURA HONESTA:
  - Con el edge REALISTA de un buen modelo (3-5% EV), 1000x toma DECADAS a 10
    apuestas/semana. El volumen lo cambia todo: 30/semana (clubes, varias ligas)
    acorta a ~5-8 anos con 5% EV. El Mundial solo NO alcanza: son ~100 partidos.
  - Subir el stake (Kelly alto) NO acorta el camino gratis: dispara P(quemado)
    y drawdowns >=50% que en la practica nadie aguanta sin abandonar/desviarse.
  - Antes de sonar con el 1000x: CONFIRMAR el edge con CLV (validacion reporte).
    Un EV de +5% "segun el modelo" que el CLV no confirma es 0% real.
  - Las casas LIMITAN a los ganadores: parte del plan a anos es rotar casas /
    line shopping. Pinnacle no limita pero su linea es la mas dura de batir.""")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        escenarios()
        return

    def _get(flag, default):
        return float(args[args.index(flag) + 1]) if flag in args else default

    reporte(
        bank0=_get("--bank", 5000), meta=_get("--meta", 5_000_000),
        ev=_get("--ev", 0.05), cuota=_get("--cuota", 1.90),
        kelly_frac=_get("--kelly", 0.5), semana=_get("--semana", 10),
        anos=_get("--anos", 20),
    )


if __name__ == "__main__":
    main()
