"""Valida el Motor Mundialista sobre los partidos historicos de StatsBomb y TUNEA w.

Es el pago del cosechador (ingesta_historica.py): en vez de tunear w con 8 partidos del harvest
de API-Football (donde el LOO no generalizaba), ahora se usan cientos de partidos REALES de
selecciones con alineacion confirmada. Para cada partido: se mapea a codigos FBref, se cruzan los
XI titulares contra jugadores.csv, se predice con el hibrido (Elo+jugadores) y se mide Brier vs el
resultado real. Despues se barre w en [0,1] buscando el que minimiza el Brier (+ LOO).

PERIOD-CORRECT (sin fuga de futuro): nuestros ratings de club son 2324/2024, asi que solo se
validan torneos de 2024 (Euro 2024, Copa America 2024) y AFCON 2023 (ene-feb 2024). Los Mundiales
2018/2022 y Euro 2020 quedan fuera: no tenemos su forma de club de esa epoca. AFCON 2023 tiene una
fuga LEVE (usa la 2324 completa, que termina despues del torneo); --solo-2024 da el subset limpio.

EFICIENCIA: las lambdas de cada partido (lado jugadores y lado Elo) NO dependen de w. Se precomputan
una vez (prediciendo con w=0 y w=1) y todo el barrido + LOO es aritmetica sobre una matriz de Brier.

NO modifica el modelo ni validacion.py: los reusa read-only. Necesita la DB cosechada
(python -m src.ingesta_historica --desde 2023).

Uso:
    python -m src.validar_statsbomb               # 2024 + AFCON 2023
    python -m src.validar_statsbomb --solo-2024   # solo Euro/Copa America 2024 (sin fuga)
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys

import numpy as np
import pandas as pd

from src import config
from src.backtest import brier_score
from src.enriquecer_xg import cargar_ajuste
from src.fixtures import _codigo_nacion, armar_xi
from src.jugadores_model import JugadoresModel, _norm
from src.montecarlo import ZIP_LAMBDA_DEF
from src.tunear_w import GRID, W_ACTUAL

# Grids del tuneo de priors (rangos pedidos por el usuario). alpha (ancla Pinnacle) NO entra:
# necesita cuotas de mercado por partido historico y StatsBomb no las tiene (ver tunear_priors).
PI_GRID = [0.05, 0.075, 0.10, 0.125, 0.15]
RHO_GRID = [-0.08, -0.10, -0.12, -0.14, -0.16, -0.18, -0.20]

DB = config.DATA_PROC / "dsoccer_historico.db"
SEASONS_CLUB = ["2324", "2024"]            # forma de club disponible hasta ~mediados de 2024
SEASONS_SB = ("2023", "2024")              # torneos StatsBomb period-correct
# Nombres StatsBomb que no estan en fixtures.PAIS_API_A_CODIGO (override local, no toca produccion)
SB_OVERRIDES = {
    "jamaica": "JAM", "turkiye": "TUR", "czechia": "CZE", "ivory coast": "CIV",
    "cote d'ivoire": "CIV", "dr congo": "COD", "cape verde": "CPV", "guinea": "GUI",
    "equatorial guinea": "EQG", "mauritania": "MTN", "namibia": "NAM", "tanzania": "TAN",
    "angola": "ANG", "mozambique": "MOZ", "zambia": "ZAM", "gambia": "GAM",
    "burkina faso": "BFA", "south korea": "KOR",
}


_FACT = np.array([math.factorial(k) for k in range(20)], dtype=float)


def _pois(g: np.ndarray, lam: float) -> np.ndarray:
    """Poisson pmf vectorizada SIN scipy (para correr en el venv minimo del server, que no
    tiene scipy). g = array de enteros 0..k; pmf(k) = e^-lam * lam^k / k!."""
    return np.exp(-lam) * np.power(float(lam), g) / _FACT[g]


def _cod(nombre_sb: str) -> str | None:
    return SB_OVERRIDES.get(_norm(nombre_sb)) or _codigo_nacion(nombre_sb)


def _modelo() -> tuple[JugadoresModel, pd.DataFrame]:
    """Motor period-correct: forma de club 2324/2024 + Elo propio. Mismo patron que harvest."""
    df = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    df = df[df["season"].astype(str).isin(SEASONS_CLUB)]
    jm = JugadoresModel().entrenar_jugadores(df, ajuste_xg=cargar_ajuste())
    cal = json.loads((config.DATA_PROC / "calibracion.json").read_text(encoding="utf-8"))
    jm.calibrar(cal["base_real"], compresion=cal["compresion"])
    jm.cargar_elo(json.loads((config.DATA_PROC / "elo_propio.json").read_text(encoding="utf-8")))
    return jm, df


def _titulares(con, match_id: int, equipo: str) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT player FROM alineaciones WHERE match_id=? AND equipo=? AND es_titular=1",
        (match_id, equipo))]


def _muestra(con, dfj: pd.DataFrame, solo_2024: bool) -> tuple[list[dict], dict]:
    seasons = ("2024",) if solo_2024 else SEASONS_SB
    placeholders = ",".join("?" * len(seasons))
    q = (f"SELECT match_id, equipo_local, equipo_visitante, resultado, "
         f"goles_local, goles_visitante, season "
         f"FROM partidos WHERE season IN ({placeholders})")
    muestra, sin_mapeo = [], {}
    reales = 0
    for mid, local, visit, real, gl, gv, season in con.execute(q, seasons):
        if real not in ("H", "D", "A"):
            continue
        cl, cv = _cod(local), _cod(visit)
        if not cl or not cv:
            for nom, cod in ((local, cl), (visit, cv)):
                if not cod:
                    sin_mapeo[nom] = sin_mapeo.get(nom, 0) + 1
            continue
        xi_l = armar_xi(_titulares(con, mid, local), cl, dfj)["xi_real"]
        xi_v = armar_xi(_titulares(con, mid, visit), cv, dfj)["xi_real"]
        reales += len(xi_l) + len(xi_v)
        muestra.append({"cl": cl, "cv": cv, "xi_l": xi_l, "xi_v": xi_v, "real": real,
                        "gl": gl, "gv": gv})
    meta = {"sin_mapeo": sin_mapeo,
            "xi_real_prom": reales / (2 * len(muestra)) if muestra else 0}
    return muestra, meta


def _matriz_brier(jm: JugadoresModel, muestra: list[dict]) -> np.ndarray:
    """Matriz n_partidos x len(GRID) de Brier. Precomputa lambdas (w=0 lado jugadores, w=1 lado
    Elo) una sola vez por partido; el resto del grid es mezcla lineal + Poisson (barato)."""
    jm.w_elo = 0.0
    pl = [jm.predecir_partido_mundial(m["xi_l"], m["xi_v"], m["cl"], m["cv"]) for m in muestra]
    jm.w_elo = 1.0
    pe = [jm.predecir_partido_mundial(m["xi_l"], m["xi_v"], m["cl"], m["cv"]) for m in muestra]

    g = np.arange(9)
    B = np.empty((len(muestra), len(GRID)))
    for i, (a, b, m) in enumerate(zip(pl, pe, muestra)):
        lp = (a["goles_esp_local"], a["goles_esp_visitante"])  # lado jugadores (w=0)
        le = (b["goles_esp_local"], b["goles_esp_visitante"])  # lado Elo       (w=1)
        for k, w in enumerate(GRID):
            lam_l = w * le[0] + (1 - w) * lp[0]
            lam_v = w * le[1] + (1 - w) * lp[1]
            mat = np.outer(_pois(g, lam_l), _pois(g, lam_v))
            mat /= mat.sum()
            B[i, k] = brier_score(float(np.tril(mat, -1).sum()), float(np.trace(mat)),
                                  float(np.triu(mat, 1).sum()), m["real"])
    return B


def tunear(solo_2024: bool = False) -> None:
    if not DB.exists():
        print(f"No existe {DB.name}. Corre primero: python -m src.ingesta_historica --desde 2023")
        return
    jm, dfj = _modelo()
    con = sqlite3.connect(DB)
    muestra, meta = _muestra(con, dfj, solo_2024)
    con.close()
    if not muestra:
        print("Muestra vacia (¿la DB no tiene torneos 2023/2024 todavia?).")
        return

    print("=" * 60)
    print(f"  TUNEO DE w SOBRE StatsBomb  |  {len(muestra)} partidos "
          f"({'solo 2024' if solo_2024 else '2024 + AFCON23'})")
    print(f"  XI real cruzado promedio: {meta['xi_real_prom']:.1f}/11 (el resto -> sombra)")
    if meta["sin_mapeo"]:
        top = sorted(meta["sin_mapeo"].items(), key=lambda x: -x[1])[:6]
        print(f"  selecciones SIN mapeo (se saltean): {top}")
    print("=" * 60)

    B = _matriz_brier(jm, muestra)
    curva = list(zip(GRID, B.mean(axis=0)))
    w_opt, b_opt = min(curva, key=lambda t: t[1])
    d = dict(curva)

    print(f"\n  {'w':>5}   {'Brier':>7}")
    for w, b in curva:
        marca = ("  <- optimo" if w == w_opt else ("  <- actual" if abs(w - W_ACTUAL) < 1e-9 else
                 ("  (jugadores)" if w == 0.0 else ("  (Elo)" if w == 1.0 else ""))))
        print(f"  {w:>5.2f}   {b:>7.4f}{marca}")

    print(f"\n  w optimo (in-sample) : {w_opt:.2f}  (Brier {b_opt:.4f})")
    print(f"  w actual (0.65)      : {d[W_ACTUAL]:.4f}")
    print(f"  jugadores puro (w=0) : {d[0.0]:.4f}")
    print(f"  Elo puro (w=1)       : {d[1.0]:.4f}")

    # Leave-one-out exacto y barato: para cada partido, el w que minimiza el Brier de los OTROS
    # (col total - fila i) y se evalua en el partido excluido.
    col = B.sum(axis=0)
    loo = float(np.mean([B[i, int(np.argmin(col - B[i]))] for i in range(len(muestra))]))
    print(f"\n  Brier LOO (w fuera de muestra): {loo:.4f}")
    print(f"  -> {'el tuneo GENERALIZA: mover w hacia ' + f'{w_opt:.2f}' if loo < d[W_ACTUAL] else 'NO supera a w=0.65 (mantener)'}")
    print(f"\n  (n={len(muestra)}: muestra seria. AFCON23 tiene fuga LEVE; --solo-2024 = limpio.)")


BENCH_1X2 = (0.40, 0.27, 0.33)  # benchmark naive (mismo que validacion.py)


def _calibracion(pares: list[tuple[float, int]], n_bins: int = 5) -> None:
    """Mini tabla de calibracion: bin de prob predicha vs frecuencia observada (y conteo)."""
    ancho = 1.0 / n_bins
    print(f"    {'bin':<11}{'pred':>7}{'obs':>7}{'n':>6}")
    for b in range(n_bins):
        lo, hi = b * ancho, (b + 1) * ancho
        gr = [(p, y) for p, y in pares if lo <= p < hi or (b == n_bins - 1 and p >= hi)]
        if not gr:
            continue
        pred = sum(p for p, _ in gr) / len(gr)
        obs = sum(y for _, y in gr) / len(gr)
        flag = ""
        if len(gr) >= 8 and pred - obs > 0.10:
            flag = "  <- sobreconfia"
        elif len(gr) >= 8 and obs - pred > 0.10:
            flag = "  <- subconfia"
        print(f"    [{lo:.1f}-{hi:.1f})  {pred*100:6.1f}%{obs*100:6.1f}%{len(gr):>6}{flag}")


def escanear_mercados(solo_2024: bool = False) -> None:
    """EDGE RETROSPECTIVO por mercado sobre StatsBomb (sin cuotas): mide si la prediccion de
    cada mercado tiene RESOLUCION (discrimina mejor que la base rate) y CALIBRACION -- la
    precondicion de que pueda existir edge. Usa w=W_ACTUAL (produccion). 1X2 y Over/Under 2.5.
    No mide edge vs MERCADO (eso es forward, con CLV); mide si el mercado es predecible."""
    if not DB.exists():
        print(f"No existe {DB.name}. Corre: python -m src.ingesta_historica --desde 2023")
        return
    jm, dfj = _modelo()
    con = sqlite3.connect(DB)
    muestra, meta = _muestra(con, dfj, solo_2024)
    con.close()
    muestra = [m for m in muestra if m["gl"] is not None and m["gv"] is not None]
    if not muestra:
        print("Muestra vacia (¿la DB no tiene torneos 2023/2024?).")
        return

    # Usa el w de PRODUCCION (default del modelo, hoy 0.85), NO el W_ACTUAL viejo de tunear_w.
    w = jm.w_elo
    # precompute lado jugadores (w=0) y lado Elo (w=1) una vez; despues mezcla lineal.
    jm.w_elo = 0.0
    pl = [jm.predecir_partido_mundial(m["xi_l"], m["xi_v"], m["cl"], m["cv"]) for m in muestra]
    jm.w_elo = 1.0
    pe = [jm.predecir_partido_mundial(m["xi_l"], m["xi_v"], m["cl"], m["cv"]) for m in muestra]
    g = np.arange(11)

    p1x2, real1x2, pover, realover = [], [], [], []
    for a, b, m in zip(pl, pe, muestra):
        lam_l = w * b["goles_esp_local"] + (1 - w) * a["goles_esp_local"]
        lam_v = w * b["goles_esp_visitante"] + (1 - w) * a["goles_esp_visitante"]
        mat = np.outer(_pois(g, lam_l), _pois(g, lam_v))
        mat /= mat.sum()
        p1x2.append((float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())))
        real1x2.append(m["real"])
        p_under = float(sum(mat[i, j] for i in range(11) for j in range(11) if i + j <= 2))
        pover.append(1.0 - p_under)
        realover.append(1 if (m["gl"] + m["gv"]) >= 3 else 0)

    n = len(muestra)
    print("=" * 64)
    print(f"  ESCANEO DE MERCADOS (retrospectivo, StatsBomb)  |  n={n}, w={w}")
    print(f"  XI real cruzado: {meta['xi_real_prom']:.1f}/11 (resto -> sombra)")
    print("=" * 64)

    # --- 1X2 ---
    bs_mod = float(np.mean([brier_score(*p, r) for p, r in zip(p1x2, real1x2)]))
    bs_ben = float(np.mean([brier_score(*BENCH_1X2, r) for r in real1x2]))
    mejora = (1 - bs_mod / bs_ben) * 100 if bs_ben > 0 else 0
    print(f"\n  [1X2]  Brier modelo {bs_mod:.4f}  vs benchmark naive {bs_ben:.4f}  "
          f"({'RESOLUCION +%.1f%%' % mejora if bs_mod < bs_ben else 'sin mejora'})")
    print("  (OJO: vs benchmark, NO vs mercado. El Elo ya ~= mercado en 1X2; aca solo se ve")
    print("   que el modelo discrimina mejor que las frecuencias base.)")
    idx = {"H": 0, "D": 1, "A": 2}
    _calibracion([(p[j], int(idx[r] == j)) for p, r in zip(p1x2, real1x2) for j in range(3)])

    # --- Over/Under 2.5 goles ---
    base = float(np.mean(realover))
    bs_mod_ou = float(np.mean([(p - y) ** 2 for p, y in zip(pover, realover)]))
    bs_base = float(np.mean([(base - y) ** 2 for y in realover]))
    skill = (1 - bs_mod_ou / bs_base) * 100 if bs_base > 0 else 0
    print(f"\n  [Over/Under 2.5 goles]  base rate over: {base*100:.0f}%  (over real en la muestra)")
    print(f"     Brier modelo {bs_mod_ou:.4f}  vs base-rate constante {bs_base:.4f}  -> skill {skill:+.1f}%")
    if bs_mod_ou < bs_base:
        print("     -> el modelo DISCRIMINA over/under: hay resolucion -> vale testear forward vs sharp.")
    else:
        print("     -> el modelo NO discrimina over/under: SIN resolucion -> no apostar totales.")
    _calibracion(list(zip(pover, realover)))
    print(f"\n  (Binary Brier de O/U NO es comparable al 3-way de 1X2: comparar modelo vs su baseline.)")
    print(f"  (n={n}: {'muestra seria' if n >= 80 else 'muestra chica, leer con pinzas'}. "
          f"Edge vs MERCADO solo lo confirma el CLV forward.)")


def _ajustar_matriz(mat: np.ndarray, lam_l: float, lam_v: float,
                    pi_zip: float, rho: float) -> np.ndarray:
    """Aplica ZIP + Dixon-Coles a una matriz Poisson normalizada. Replica ANALITICA (exacta, sin
    ruido MC) de lo que hace el Montecarlo: ZIP mezcla (1-pi)*Poisson + pi*Z con Z topado en <=1 gol
    (0-0/1-0/0-1, reparto 0-vs-1 via ZIP_LAMBDA_DEF, el gol al mas fuerte); DC reponddera los 4
    marcadores bajos (rho<0 infla 0-0/1-1, desinfla 1-0/0-1) y renormaliza."""
    m = mat.copy()
    if pi_zip > 0:
        p_un = ZIP_LAMBDA_DEF / (1.0 + ZIP_LAMBDA_DEF)  # P(1 gol | defensivo)
        tot = lam_l + lam_v
        share = lam_l / tot if tot > 0 else 0.5
        m = (1.0 - pi_zip) * m
        m[0, 0] += pi_zip * (1.0 - p_un)
        m[1, 0] += pi_zip * p_un * share
        m[0, 1] += pi_zip * p_un * (1.0 - share)
    if rho != 0.0:
        m[0, 0] *= (1.0 - lam_l * lam_v * rho)
        m[1, 0] *= (1.0 + lam_v * rho)
        m[0, 1] *= (1.0 + lam_l * rho)
        m[1, 1] *= (1.0 - rho)
    s = m.sum()
    return m / s if s > 0 else m


def _brier_conjunto(mat: np.ndarray, real: str, over_real: int) -> tuple[float, float]:
    """(Brier 1X2 3-way one-hot [0-2], Brier O/U 2.5 binario 2-comp [0-2]). La suma es el objetivo.
    Ambos en la misma escala 0-2 para que pi (mueve O/U) y rho (mueve 1X2) pesen parejo."""
    p_h = float(np.tril(mat, -1).sum())
    p_d = float(np.trace(mat))
    p_a = float(np.triu(mat, 1).sum())
    b1 = brier_score(p_h, p_d, p_a, real)
    p_under = float(mat[0, 0] + mat[0, 1] + mat[0, 2] + mat[1, 0] + mat[1, 1] + mat[2, 0])
    p_over = 1.0 - p_under
    b_ou = (p_over - over_real) ** 2 + (p_under - (1 - over_real)) ** 2
    return b1, b_ou


def tunear_priors(solo_2024: bool = False) -> None:
    """Grid search de PI_ZIP_SELECCIONES x RHO_DIXON_COLES sobre StatsBomb, minimizando el Brier
    conjunto (1X2 + O/U 2.5) del MODELO PURO (sin ancla; w=0.85 produccion). alpha (ancla Pinnacle)
    NO se tunea aca: necesita cuotas de mercado por partido y StatsBomb no las tiene -> se tunea
    forward con predicciones_log (pin_*/cierre_*). Las lambdas por partido se precomputan una vez."""
    if not DB.exists():
        print(f"No existe {DB.name}. Corre: python -m src.ingesta_historica --desde 2023")
        return
    jm, dfj = _modelo()
    con = sqlite3.connect(DB)
    muestra, meta = _muestra(con, dfj, solo_2024)
    con.close()
    muestra = [m for m in muestra if m["gl"] is not None and m["gv"] is not None]
    if not muestra:
        print("Muestra vacia (¿la DB no tiene torneos 2023/2024?).")
        return

    w = jm.w_elo  # produccion (0.85). Lado jugadores (w=0) y Elo (w=1) se precomputan una vez.
    jm.w_elo = 0.0
    pl = [jm.predecir_partido_mundial(m["xi_l"], m["xi_v"], m["cl"], m["cv"]) for m in muestra]
    jm.w_elo = 1.0
    pe = [jm.predecir_partido_mundial(m["xi_l"], m["xi_v"], m["cl"], m["cv"]) for m in muestra]
    jm.w_elo = w

    g = np.arange(11)
    base = []  # (mat_poisson_normalizada, lam_l, lam_v, real, over_real)
    for a, b, m in zip(pl, pe, muestra):
        lam_l = w * b["goles_esp_local"] + (1 - w) * a["goles_esp_local"]
        lam_v = w * b["goles_esp_visitante"] + (1 - w) * a["goles_esp_visitante"]
        mat = np.outer(_pois(g, lam_l), _pois(g, lam_v))
        mat /= mat.sum()
        base.append((mat, lam_l, lam_v, m["real"], 1 if (m["gl"] + m["gv"]) >= 3 else 0))

    n = len(base)

    def score(pi: float, rho: float) -> tuple[float, float, float, float]:
        """(joint, brier_1x2, brier_ou, prob_over_media) sobre la muestra."""
        b1 = bou = pov = 0.0
        for mat, ll, lv, real, ov in base:
            adj = _ajustar_matriz(mat, ll, lv, pi, rho)
            x1, xo = _brier_conjunto(adj, real, ov)
            b1 += x1; bou += xo
            pov += float(adj[0, 0] + adj[0, 1] + adj[0, 2] + adj[1, 0] + adj[1, 1] + adj[2, 0])
        return (b1 + bou) / n, b1 / n, bou / n, 1.0 - pov / n

    # Tensor de Brier conjunto para grid + LOO
    B = np.empty((n, len(PI_GRID), len(RHO_GRID)))
    for i, (mat, ll, lv, real, ov) in enumerate(base):
        for j, pi in enumerate(PI_GRID):
            for k, rho in enumerate(RHO_GRID):
                x1, xo = _brier_conjunto(_ajustar_matriz(mat, ll, lv, pi, rho), real, ov)
                B[i, j, k] = x1 + xo

    media = B.mean(axis=0)
    jstar, kstar = np.unravel_index(int(np.argmin(media)), media.shape)
    pi_opt, rho_opt = PI_GRID[jstar], RHO_GRID[kstar]

    base_j, base_1, base_ou, base_pov = score(0.0, 0.0)
    cur_j, cur_1, cur_ou, cur_pov = score(config.PI_ZIP_SELECCIONES, config.RHO_DIXON_COLES)
    opt_j, opt_1, opt_ou, opt_pov = score(pi_opt, rho_opt)
    over_real = float(np.mean([b[4] for b in base]))  # tasa de Over 2.5 real en la muestra

    # LOO (fuera de muestra): para cada partido, el (pi,rho) optimo de los OTROS, evaluado en el.
    col = B.sum(axis=0)
    loo = float(np.mean([B[i].flatten()[int(np.argmin((col - B[i]).flatten()))] for i in range(n)]))

    print("=" * 66)
    print(f"  TUNEO DE PRIORS (ZIP pi x Dixon-Coles rho)  |  n={n} partidos StatsBomb")
    print(f"  ({'solo 2024' if solo_2024 else '2024 + AFCON23'}, period-correct; w={w} produccion)")
    print(f"  Objetivo: Brier CONJUNTO (1X2 3-way + O/U 2.5), ambos en escala 0-2")
    print("=" * 66)

    print(f"\n  Grid del Brier conjunto (filas pi, columnas rho):")
    print("        " + "".join(f"{r:>8.2f}" for r in RHO_GRID))
    for j, pi in enumerate(PI_GRID):
        celdas = "".join(
            (f"{media[j,k]:>8.4f}" if (j, k) != (jstar, kstar) else f"{media[j,k]:>7.4f}*")
            for k in range(len(RHO_GRID)))
        print(f"  {pi:>5.3f} {celdas}")
    print("  (* = optimo)")

    print(f"\n  {'config':<26}{'joint':>9}{'1X2':>9}{'O/U':>9}{'P(over)':>9}")
    print(f"  {'baseline (0, 0)':<26}{base_j:>9.4f}{base_1:>9.4f}{base_ou:>9.4f}{base_pov*100:>8.0f}%")
    print(f"  {'actual (%.3f, %.2f)' % (config.PI_ZIP_SELECCIONES, config.RHO_DIXON_COLES):<26}"
          f"{cur_j:>9.4f}{cur_1:>9.4f}{cur_ou:>9.4f}{cur_pov*100:>8.0f}%")
    print(f"  {'OPTIMO (%.3f, %.2f)' % (pi_opt, rho_opt):<26}"
          f"{opt_j:>9.4f}{opt_1:>9.4f}{opt_ou:>9.4f}{opt_pov*100:>8.0f}%")
    print(f"  {'(over REAL en la muestra)':<26}{'':>27}{over_real*100:>8.0f}%")

    red = (1 - opt_j / base_j) * 100 if base_j > 0 else 0
    print(f"\n  Reduccion del Brier conjunto (optimo vs baseline): {red:+.2f}%")
    print(f"  LOO (optimo fuera de muestra): {loo:.4f}  -> "
          f"{'GENERALIZA (mejor que baseline)' if loo < base_j else 'NO generaliza (ruido)'}")
    en_borde = pi_opt in (PI_GRID[0], PI_GRID[-1]) or rho_opt in (RHO_GRID[0], RHO_GRID[-1])
    if en_borde:
        print(f"  [!] El optimo cae en el BORDE del grid -> el verdadero optimo puede estar mas alla.")
    print(f"\n  ALPHA_ANCLA_PINNACLE: NO tuneable aca (StatsBomb sin cuotas). Se mantiene 0.35;")
    print(f"  tunear forward con predicciones_log (pin_*/cierre_*) cuando haya muestra de WC.")
    print(f"\n  GANADORES -> PI_ZIP_SELECCIONES={pi_opt}  RHO_DIXON_COLES={rho_opt}")
    print(f"  (n={n}: muestra seria. pi/rho son CALIBRACION 1-D c/u -> bajo riesgo de overfit.)")


def main() -> None:
    if "--tune-priors" in sys.argv:
        tunear_priors(solo_2024="--solo-2024" in sys.argv)
    elif "--mercados" in sys.argv:
        escanear_mercados(solo_2024="--solo-2024" in sys.argv)
    else:
        tunear(solo_2024="--solo-2024" in sys.argv)


if __name__ == "__main__":
    main()
