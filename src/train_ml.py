"""Motor de Machine Learning (LightGBM) para predecir tasas de gol por partido.

ENFOQUE HONESTO (ver discusion de diseno):
  El spec original queria features de jugadores por partido, pero NO tenemos las
  alineaciones de cada partido de club. Asi que entrenamos sobre partidos.csv (datos
  por equipo, que SI tenemos) con features de FORMA RECIENTE construidas sin fuga de
  futuro: promedios moviles de los ultimos N partidos ANTES de cada encuentro.

  El modelo predice la TASA esperada de goles de cada bando (objetivo 'poisson' de
  LightGBM). Esas tasas alimentan luego al Simulador de Montecarlo.

REGLA DE ORO: el ML no reemplaza nada por decreto. Se mide contra Dixon-Coles con el
mismo Brier Score y el mismo split. Entra al sistema SOLO si gana el backtest.

Uso:
    python -m src.train_ml
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src import config
from src.backtest import brier_score, frecuencias_base, resultado_real
from src.dixon_coles import DixonColes

VENTANA = 6            # partidos de forma reciente para los promedios moviles
TEMPORADA_TEST = "2425"
# Estadisticas (desde la optica de cada equipo) que promediamos como forma reciente
STATS = ["gf", "gc", "sf", "stf", "ff", "cf", "cards"]


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROC / "partidos.csv", parse_dates=["fecha"])
    df["temporada"] = df["temporada"].astype(str)
    return df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)


def _largo(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo-partido, con sus stats desde su propia optica."""
    def cards(pref):
        return df[f"amarillas_{pref}"].fillna(0) + df[f"rojas_{pref}"].fillna(0)

    home = pd.DataFrame({
        "idx": df.index, "fecha": df["fecha"], "equipo": df["local"], "es_local": True,
        "gf": df["goles_local"], "gc": df["goles_visitante"],
        "sf": df["tiros_local"], "stf": df["tiros_arco_local"],
        "ff": df["faltas_local"], "cf": df["corners_local"], "cards": cards("local"),
    })
    away = pd.DataFrame({
        "idx": df.index, "fecha": df["fecha"], "equipo": df["visitante"], "es_local": False,
        "gf": df["goles_visitante"], "gc": df["goles_local"],
        "sf": df["tiros_visitante"], "stf": df["tiros_arco_visitante"],
        "ff": df["faltas_visitante"], "cf": df["corners_visitante"], "cards": cards("visitante"),
    })
    return pd.concat([home, away], ignore_index=True)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de features de forma reciente (sin fuga: solo partidos previos)."""
    largo = _largo(df).sort_values(["equipo", "fecha"])

    roll_cols = []
    for s in STATS:
        col = f"{s}_roll"
        # shift(1) excluye el partido actual -> solo historia previa (cero fuga)
        largo[col] = (largo.groupby("equipo")[s]
                      .transform(lambda x: x.shift(1).rolling(VENTANA, min_periods=2).mean()))
        roll_cols.append(col)

    feat_h = largo[largo["es_local"]].set_index("idx")[roll_cols].add_prefix("h_")
    feat_a = largo[~largo["es_local"]].set_index("idx")[roll_cols].add_prefix("a_")

    out = df.join(feat_h).join(feat_a)
    out["liga"] = out["liga"].astype("category")
    feat_cols = list(feat_h.columns) + list(feat_a.columns) + ["liga"]

    # Tiramos partidos sin suficiente historia (primeros de cada equipo)
    out = out.dropna(subset=[c for c in feat_cols if c != "liga"])
    return out, feat_cols


def poisson_1x2(lam_l: float, lam_v: float, maxg: int = 10) -> tuple[float, float, float]:
    g = np.arange(maxg + 1)
    m = np.outer(poisson.pmf(g, lam_l), poisson.pmf(g, lam_v))
    m /= m.sum()
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def main() -> None:
    try:
        import lightgbm as lgb
    except ImportError:
        print("Falta lightgbm. Instala con:  python -m pip install lightgbm")
        return

    df = cargar_datos()
    datos, feat_cols = construir_features(df)

    train = datos[datos["temporada"] != TEMPORADA_TEST]
    test = datos[datos["temporada"] == TEMPORADA_TEST]
    print(f"Train: {len(train)} partidos (pre {TEMPORADA_TEST})  |  Test: {len(test)} partidos\n")

    # --- LightGBM: dos regresores Poisson (goles local y visitante) ---
    params = dict(objective="poisson", n_estimators=400, learning_rate=0.04,
                  num_leaves=31, min_child_samples=50, subsample=0.8,
                  colsample_bytree=0.8, random_state=42, verbose=-1)
    Xtr = train[feat_cols]
    m_local = lgb.LGBMRegressor(**params).fit(Xtr, train["goles_local"])
    m_visit = lgb.LGBMRegressor(**params).fit(Xtr, train["goles_visitante"])

    Xte = test[feat_cols]
    lam_l = m_local.predict(Xte)
    lam_v = m_visit.predict(Xte)

    # --- Dixon-Coles entrenado en el MISMO train, para comparacion justa ---
    dc = DixonColes().entrenar(train)
    pb_h, pb_d, pb_a = frecuencias_base(train)

    bs_ml, bs_dc, bs_bench, n = 0.0, 0.0, 0.0, 0
    for (lh, lv), (_, p) in zip(zip(lam_l, lam_v), test.iterrows()):
        real = resultado_real(p["goles_local"], p["goles_visitante"])
        try:
            d = dc.predecir(p["local"], p["visitante"])
        except KeyError:
            continue  # equipo no visto: lo saltamos para los TRES por igual
        ph, pd_, pa = poisson_1x2(lh, lv)
        bs_ml += brier_score(ph, pd_, pa, real)
        bs_dc += brier_score(d["prob_local"], d["prob_empate"], d["prob_visitante"], real)
        bs_bench += brier_score(pb_h, pb_d, pb_a, real)
        n += 1

    bs_ml, bs_dc, bs_bench = bs_ml / n, bs_dc / n, bs_bench / n
    print("=" * 56)
    print(f"  HEAD-TO-HEAD (Brier Score, {n} partidos, menor = mejor)")
    print("=" * 56)
    print(f"  LightGBM (forma reciente) : {bs_ml:.4f}")
    print(f"  Dixon-Coles               : {bs_dc:.4f}")
    print(f"  Benchmark base            : {bs_bench:.4f}")
    print()
    if bs_ml < bs_dc:
        print(f"  -> El ML GANA a Dixon-Coles por {(bs_dc-bs_ml)/bs_dc*100:.1f}%. Merece entrar.")
    else:
        print(f"  -> Dixon-Coles aguanta: el ML NO mejora ({(bs_ml-bs_dc)/bs_dc*100:+.1f}%).")
        print("     Conclusion honesta: para 1X2 con estos datos, el modelo estadistico")
        print("     bien especificado iguala o supera al ML. Lo esperable en la literatura.")

    # --- Que features mira el ML (interpretabilidad) ---
    imp = (pd.Series(m_local.feature_importances_, index=feat_cols)
           .sort_values(ascending=False).head(8))
    print("\nTop features para predecir goles del local:")
    for f, v in imp.items():
        print(f"  {f:<14} {int(v)}")


if __name__ == "__main__":
    main()
