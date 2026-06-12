"""Auto-validacion del Motor Mundialista: registra predicciones y las compara con los
resultados REALES (bajados de API-Football). Es el unico camino para saber si el modelo
de selecciones tiene edge o si la sub-diferenciacion lo hace inservible.

Flujo:
  1. Antes del partido:  registrar  -> corre el motor y guarda la prediccion en el log.
  2. Despues del partido: actualizar -> baja el resultado final y calcula el Brier.
  3. Cuando quieras:      reporte    -> metricas agregadas (Brier modelo vs benchmark, acierto).

Uso:
    python -m src.validacion registrar 1540358   # loguea la prediccion de un fixture
    python -m src.validacion actualizar           # completa resultados de partidos jugados
    python -m src.validacion reporte               # metricas acumuladas

Necesita API_FOOTBALL_KEY (variable de entorno) para registrar y actualizar.
"""

from __future__ import annotations

import sys

import pandas as pd

from src import config
from src.backtest import brier_score, resultado_real

LOG = config.DATA_PROC / "predicciones_log.csv"
COLUMNAS = [
    "fixture_id", "fecha", "local", "visitante", "cod_l", "cod_v", "arbitro",
    "prob_local", "prob_empate", "prob_visitante", "goles_esp_l", "goles_esp_v", "over_2_5",
    "cuota_l", "cuota_e", "cuota_v",  # mejores cuotas 1X2 del mercado al registrar
    "gl_real", "gv_real", "resultado_real", "brier_modelo", "brier_bench", "brier_mercado",
]
# Benchmark naive para internacionales (referencia, no sofisticado). Ligero sesgo al local.
BENCH = (0.40, 0.27, 0.33)


def _cargar() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=COLUMNAS)


def _guardar(df: pd.DataFrame) -> None:
    df.to_csv(LOG, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
def registrar(fixture_id: int) -> None:
    """Corre el motor para un fixture y guarda la prediccion (sin resultado todavia)."""
    from src.fixtures import correr_partido_auto

    log = _cargar()
    if (log["fixture_id"] == fixture_id).any():
        print(f"El fixture {fixture_id} ya esta registrado. (Usa 'actualizar' para el resultado.)")
        return

    info = correr_partido_auto(fixture_id)  # corre, imprime y devuelve la prediccion
    if info is None:
        return
    r = info["res"]
    fila = {
        "fixture_id": fixture_id, "fecha": info["fecha"],
        "local": info["local"], "visitante": info["visitante"],
        "cod_l": info["cod_l"], "cod_v": info["cod_v"], "arbitro": info["arbitro"] or "",
        "prob_local": round(r["prob_local"], 4), "prob_empate": round(r["prob_empate"], 4),
        "prob_visitante": round(r["prob_visitante"], 4),
        "goles_esp_l": round(r["goles_esp"][0], 3), "goles_esp_v": round(r["goles_esp"][1], 3),
        "over_2_5": round(r["over_2_5_goles"], 4),
        "gl_real": "", "gv_real": "", "resultado_real": "",
        "brier_modelo": "", "brier_bench": "", "brier_mercado": "",
    }
    # Capturar las mejores cuotas 1X2 del mercado al momento de registrar
    mkt = None
    try:
        from src.valor import cuotas_mercado
        mkt = cuotas_mercado(fixture_id)
        fila["cuota_l"] = round(mkt["Home"]["mejor"], 2) if mkt and mkt.get("Home") else ""
        fila["cuota_e"] = round(mkt["Draw"]["mejor"], 2) if mkt and mkt.get("Draw") else ""
        fila["cuota_v"] = round(mkt["Away"]["mejor"], 2) if mkt and mkt.get("Away") else ""
    except Exception:
        fila["cuota_l"] = fila["cuota_e"] = fila["cuota_v"] = ""

    log = pd.concat([log, pd.DataFrame([fila])], ignore_index=True)
    _guardar(log)
    print(f"\n-> Prediccion registrada en {LOG.name} (fixture {fixture_id}).")
    return {"info": info, "cuotas": mkt}  # para que autorun pueda notificar a Telegram


def actualizar_resultados() -> None:
    """Baja el resultado final de los partidos jugados y calcula el Brier de cada uno."""
    from src.fixtures import _api_get

    log = _cargar()
    # Columnas de texto/resultado: forzar a object para poder asignarles strings ('H'/'D'/'A')
    # aunque pandas las haya leido como float (cuando estaban vacias en el CSV).
    for c in ("resultado_real", "arbitro", "gl_real", "gv_real"):
        if c in log.columns:
            log[c] = log[c].astype("object")
    pendientes = log[log["resultado_real"].isna() | (log["resultado_real"].astype(str) == "")]
    if pendientes.empty:
        print("No hay predicciones pendientes de resultado.")
        return

    actualizados = 0
    for idx, fila in pendientes.iterrows():
        info = _api_get("fixtures", {"id": int(fila["fixture_id"])})
        if not info:
            continue
        f = info[0]
        estado = f["fixture"]["status"]["short"]
        if estado not in ("FT", "AET", "PEN"):
            continue  # todavia no termino
        gl, gv = f["goals"]["home"], f["goals"]["away"]
        real = resultado_real(gl, gv)
        log.at[idx, "gl_real"] = gl
        log.at[idx, "gv_real"] = gv
        log.at[idx, "resultado_real"] = real
        log.at[idx, "brier_modelo"] = round(brier_score(
            fila["prob_local"], fila["prob_empate"], fila["prob_visitante"], real), 4)
        log.at[idx, "brier_bench"] = round(brier_score(*BENCH, real), 4)
        # Brier del MERCADO: de-margina las cuotas 1X2 guardadas (la vara que importa)
        try:
            cl, ce, cv = float(fila["cuota_l"]), float(fila["cuota_e"]), float(fila["cuota_v"])
            if cl > 0 and ce > 0 and cv > 0:  # 'nan > 0' es False -> saltea filas sin cuota
                imp = [1 / cl, 1 / ce, 1 / cv]; s = sum(imp)
                log.at[idx, "brier_mercado"] = round(
                    brier_score(imp[0] / s, imp[1] / s, imp[2] / s, real), 4)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        actualizados += 1
        print(f"  {fila['local']} {gl}-{gv} {fila['visitante']}  ({real})")

    _guardar(log)
    print(f"\nActualizados {actualizados} resultados.")


def reporte() -> None:
    """Metricas agregadas sobre las predicciones que ya tienen resultado."""
    log = _cargar()
    hechos = log[log["resultado_real"].isin(["H", "D", "A"])].copy()
    n = len(hechos)
    if n == 0:
        print("Aun no hay partidos con resultado. Corre 'actualizar' despues de que se jueguen.")
        return

    # Acierto: el resultado mas probable del modelo, ¿ocurrio?
    pred_argmax = hechos[["prob_local", "prob_empate", "prob_visitante"]].values.argmax(axis=1)
    real_idx = hechos["resultado_real"].map({"H": 0, "D": 1, "A": 2}).values
    acierto = (pred_argmax == real_idx).mean()

    bs_mod = hechos["brier_modelo"].mean()
    bs_ben = hechos["brier_bench"].mean()

    print("=" * 50)
    print(f"  VALIDACION del Motor Mundialista ({n} partidos)")
    print("=" * 50)
    print(f"  Brier modelo    : {bs_mod:.4f}")
    print(f"  Brier benchmark : {bs_ben:.4f}")
    veredicto = "el modelo aporta" if bs_mod < bs_ben else "el modelo NO supera al benchmark"
    print(f"  -> {veredicto}")
    print(f"  Acierto del favorito del modelo: {acierto*100:.0f}% ({n} partidos)")

    # La comparacion que IMPORTA: modelo vs MERCADO (sobre los partidos con cuota guardada)
    con_mkt = hechos[hechos["brier_mercado"].notna()]
    if len(con_mkt):
        bm_mod = con_mkt["brier_modelo"].mean()
        bm_mkt = con_mkt["brier_mercado"].mean()
        print(f"\n  --- vs MERCADO ({len(con_mkt)} partidos con cuota) ---")
        print(f"  Brier modelo : {bm_mod:.4f}")
        print(f"  Brier mercado: {bm_mkt:.4f}")
        if bm_mod < bm_mkt:
            print(f"  -> el modelo LE GANA al mercado (edge!). Confirmar con mas muestra.")
        else:
            print(f"  -> el mercado es mejor (lo esperable). El edge esta en mercados 2rios.")
    print(f"\n  (Muestra chica: con <15-20 partidos esto es ruido. Seguir acumulando.)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m src.validacion [registrar <id> | actualizar | reporte]")
        return
    cmd = sys.argv[1]
    if cmd == "registrar" and len(sys.argv) >= 3:
        registrar(int(sys.argv[2]))
    elif cmd == "actualizar":
        actualizar_resultados()
    elif cmd == "reporte":
        reporte()
    else:
        print("Comando no reconocido. Uso: registrar <id> | actualizar | reporte")


if __name__ == "__main__":
    main()
