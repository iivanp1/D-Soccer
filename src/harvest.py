"""Validacion historica: cosecha partidos de selecciones ya jugados (con sus alineaciones
reales) de API-Football y mide cuanto le pega el Motor Mundialista al resultado real.

Es la PRIMERA medicion honesta de si el modelo de selecciones tiene edge.

Decisiones clave (rigor):
  - RATINGS PERIODO-CORRECTOS: para validar partidos de 2024, usamos solo la forma de
    clubes de la temporada 2023-24 (season 2324 en jugadores.csv). Asi NO hay fuga de
    informacion del futuro (la temporada 24-25 paso DESPUES de los torneos de 2024).
  - CACHE LOCAL: cada respuesta de la API se guarda en data/raw/api_cache/ -> re-correr
    NO vuelve a gastar cuota (plan gratis: 100 requests/dia).
  - El plan gratis solo da 2022-2024, asi que validamos con internacionales de 2024
    (Eurocopa, Copa America, amistosos, etc.).

Uso (necesita API_FOOTBALL_KEY):
    python -m src.harvest 4 2024 8     # liga 4 (Eurocopa), 2024, primeros 8 partidos
    python -m src.harvest 9 2024 12    # liga 9 (Copa America)
    python -m src.harvest 10 2024 30   # liga 10 (Amistosos)
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from src import config
from src.backtest import brier_score, resultado_real
from src.jugadores_model import JugadoresModel

CACHE = config.RAIZ / "data" / "raw" / "api_cache"
SEASON_RATINGS = 2324            # forma de clubes a usar (periodo-correcto para 2024)
BENCH = (0.40, 0.27, 0.33)       # benchmark naive (referencia)


# --------------------------------------------------------------------------- #
def _cache_json(nombre: str, fetch_fn):
    """Devuelve el JSON cacheado; si no existe, lo baja con fetch_fn y lo guarda."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{nombre}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    data = fetch_fn()
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _fixtures(liga: int, season: int):
    from src.fixtures import _api_get
    return _cache_json(f"fixtures_{liga}_{season}",
                       lambda: _api_get("fixtures", {"league": liga, "season": season, "status": "FT"}))


def _lineups(fixture_id: int):
    from src.fixtures import _api_get
    return _cache_json(f"lineups_{fixture_id}",
                       lambda: _api_get("fixtures/lineups", {"fixture": fixture_id}))


def _modelo_periodo_correcto():
    """Motor entrenado SOLO con la forma de clubes 2023-24 (sin fuga para 2024)."""
    df = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    df = df[df["season"] == SEASON_RATINGS]
    jm = JugadoresModel().entrenar_jugadores(df)
    # Recalibramos atk_ref/def_ref en ESTA base de ratings; base_real (media del Mundial)
    # la reutilizamos de calibracion.json para no re-scrapear.
    cal = json.loads((config.DATA_PROC / "calibracion.json").read_text(encoding="utf-8"))
    jm.calibrar(cal["base_real"], compresion=cal["compresion"])
    return jm, df


# --------------------------------------------------------------------------- #
def validar(liga: int, season: int, limite: int) -> None:
    from src.fixtures import _codigo_nacion, armar_xi

    jm, dfj = _modelo_periodo_correcto()
    fixtures = _fixtures(liga, season)
    print(f"Liga {liga}, {season}: {len(fixtures)} partidos jugados. Validando hasta {limite}.\n")

    filas, saltados = [], 0
    for f in fixtures:
        if len(filas) >= limite:
            break
        gl, gv = f["goals"]["home"], f["goals"]["away"]
        if gl is None or gv is None:
            continue
        nl, nv = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
        cl, cv = _codigo_nacion(nl), _codigo_nacion(nv)
        if not cl or not cv:
            saltados += 1
            continue

        # Alineaciones reales -> XI emparejado (contra la base de ratings 2324)
        xi_l = xi_v = None
        for eq in _lineups(f["fixture"]["id"]):
            cod = _codigo_nacion(eq["team"]["name"])
            nombres = [p["player"]["name"] for p in eq.get("startXI", [])]
            r = armar_xi(nombres, cod, dfj)
            if cod == cl:
                xi_l = r["xi_real"]
            elif cod == cv:
                xi_v = r["xi_real"]

        pred = jm.predecir_partido_mundial(xi_l, xi_v, cl, cv)
        real = resultado_real(gl, gv)
        bm = brier_score(pred["prob_local"], pred["prob_empate"], pred["prob_visitante"], real)
        bb = brier_score(*BENCH, real)
        filas.append({"local": nl, "visitante": nv, "gl": gl, "gv": gv, "real": real,
                      "pL": pred["prob_local"], "pE": pred["prob_empate"], "pV": pred["prob_visitante"],
                      "brier_modelo": bm, "brier_bench": bb})
        print(f"  {nl} {gl}-{gv} {nv} ({real}) | "
              f"modelo L/E/V {pred['prob_local']*100:.0f}/{pred['prob_empate']*100:.0f}/"
              f"{pred['prob_visitante']*100:.0f} | brier {bm:.3f}")

    if not filas:
        print("No se valido ningun partido (¿faltan mapeos de nacion?).")
        return

    df = pd.DataFrame(filas)
    pred_argmax = df[["pL", "pE", "pV"]].values.argmax(axis=1)
    real_idx = df["real"].map({"H": 0, "D": 1, "A": 2}).values
    acierto = (pred_argmax == real_idx).mean()

    print(f"\n{'='*52}")
    print(f"  VALIDACION  {liga}/{season}  ({len(df)} partidos, {saltados} saltados)")
    print(f"{'='*52}")
    print(f"  Brier modelo    : {df['brier_modelo'].mean():.4f}")
    print(f"  Brier benchmark : {df['brier_bench'].mean():.4f}")
    mejora = (df['brier_bench'].mean() - df['brier_modelo'].mean()) / df['brier_bench'].mean() * 100
    print(f"  -> {'el modelo APORTA' if mejora>0 else 'el modelo NO supera al benchmark'} ({mejora:+.1f}%)")
    print(f"  Acierto del favorito del modelo: {acierto*100:.0f}%")
    print(f"\n  (Muestra chica: leer como senal preliminar, no veredicto. Cachear y ampliar.)")


def main() -> None:
    liga = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    limite = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    validar(liga, season, limite)


if __name__ == "__main__":
    main()
