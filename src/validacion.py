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

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.backtest import brier_score, resultado_real

# FUENTE DE VERDAD: el log canonico vive SOLO en el server. Para que un run LOCAL (dev) no
# contamine ese archivo con filas divergentes (timing/cuotas/codigo distintos -> duplicados
# al juntar), por defecto se escribe a un sandbox '-dev'. El server se declara escritor
# canonico con D_SOCCER_CANONICAL=1 en su .env. Asi local es SEGURO por defecto.
CANONICO = "predicciones_log.csv"
DEV = "predicciones_log_dev.csv"
COLUMNAS = [
    "fixture_id", "fecha", "local", "visitante", "cod_l", "cod_v", "arbitro",
    "prob_local", "prob_empate", "prob_visitante", "goles_esp_l", "goles_esp_v", "over_2_5",
    "cuota_l", "cuota_e", "cuota_v",  # mejores cuotas 1X2 del mercado al registrar
    "gl_real", "gv_real", "resultado_real", "brier_modelo", "brier_bench", "brier_mercado",
    "registrado_en", "fuente",  # metadata: permite consolidar logs sin duplicar (clave fixture_id)
]
# Benchmark naive para internacionales (referencia, no sofisticado). Ligero sesgo al local.
BENCH = (0.40, 0.27, 0.33)


def _es_canonico() -> bool:
    """True solo donde el server lo declara (D_SOCCER_CANONICAL=1 en .env)."""
    config.cargar_env()
    return os.environ.get("D_SOCCER_CANONICAL", "").strip().lower() in ("1", "true", "yes", "si")


def _log_path() -> Path:
    return config.DATA_PROC / (CANONICO if _es_canonico() else DEV)


def _cargar(path: Path | None = None) -> pd.DataFrame:
    p = path or _log_path()
    if p.exists():
        # reindex asegura el esquema actual aunque el CSV sea viejo (sin columnas nuevas).
        return pd.read_csv(p).reindex(columns=COLUMNAS)
    return pd.DataFrame(columns=COLUMNAS)


def _guardar(df: pd.DataFrame, path: Path | None = None) -> None:
    df.to_csv(path or _log_path(), index=False, encoding="utf-8")


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
        "registrado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuente": "canonico" if _es_canonico() else "dev",
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
    destino = _log_path().name
    print(f"\n-> Prediccion registrada en {destino} (fixture {fixture_id}) [fuente: {fila['fuente']}].")
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


def consolidar(archivos: list[str], salida: str = "predicciones_consolidado.csv") -> None:
    """Une varios logs en UNO SOLO sin duplicar (clave: fixture_id).

    Resuelve el problema de tener el log canonico del server y respaldos/sandbox -dev por
    separado: al juntarlos, un mismo partido aparece varias veces y en conflicto. Aca, ante
    duplicados del mismo fixture_id, nos quedamos con la MEJOR fila en este orden:
      1) la 'canonico' (el server) le gana a la 'dev' (registro local ad-hoc),
      2) la registrada mas temprano (pre-match > post-FT),
      3) la que ya tiene resultado cargado (no perder el dato).
    Los archivos se resuelven relativos a data/processed/ (o ruta absoluta). Default: junta
    el canonico + el -dev locales.
    """
    frames = []
    for a in archivos:
        p = Path(a)
        if not p.is_absolute():
            p = config.DATA_PROC / a
        if not p.exists():
            print(f"  (no existe, se saltea: {p.name})")
            continue
        frames.append(pd.read_csv(p).reindex(columns=COLUMNAS))
        print(f"  + {p.name}: {len(frames[-1])} filas")
    if not frames:
        print("No hay archivos validos para consolidar.")
        return

    todo = pd.concat(frames, ignore_index=True)
    # Claves de prioridad (menor = se prefiere). Legacy sin 'fuente' -> tratado como 'dev'.
    todo["_f"] = (todo["fuente"].fillna("dev") != "canonico").astype(int)
    todo["_r"] = todo["resultado_real"].isna().astype(int)
    todo = todo.sort_values(["_f", "_r", "registrado_en"], na_position="last")
    out = (todo.drop_duplicates("fixture_id", keep="first")
               .drop(columns=["_f", "_r"])[COLUMNAS]
               .sort_values("fecha"))
    sp = config.DATA_PROC / salida
    _guardar(out, sp)
    print(f"\nConsolidado: {len(todo)} filas -> {len(out)} unicas por fixture_id -> {sp.name}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m src.validacion [registrar <id> | actualizar | reporte | "
              "consolidar [logs...]]")
        return
    cmd = sys.argv[1]
    if cmd == "registrar" and len(sys.argv) >= 3:
        registrar(int(sys.argv[2]))
    elif cmd == "actualizar":
        actualizar_resultados()
    elif cmd == "reporte":
        reporte()
    elif cmd == "consolidar":
        consolidar(sys.argv[2:] or [CANONICO, DEV])
    else:
        print("Comando no reconocido. Uso: registrar <id> | actualizar | reporte | "
              "consolidar [logs...]")


if __name__ == "__main__":
    main()
