"""Punto de entrada DESATENDIDO para el servidor (cron). Acumula datos de validacion solo.

Hace dos cosas:
  1. actualizar : baja el resultado real de los partidos ya logueados que terminaron.
  2. registrar  : registra los partidos que arrancan en ~20-45 min (ya con alineacion
                  confirmada), predice, loguea y manda el reporte completo a Telegram.

Asi el predicciones_log.csv crece dia a dia = tu dataset de validacion del Mundial, sin
tocar nada. Solo usa la API-Football + los datos commiteados (NO scrapea, NO necesita Chrome).
Los fixtures del dia se cachean -> las corridas frecuentes casi no gastan cuota.

AUDIT TRAIL: cada corrida loguea CADA partido de World Cup que la API devuelve, con la
decision tomada (REGISTRADO / fuera-ventana / sin-mapeo / ya-registrado / error). Si un
partido no se notifico, hacer grep "AUDIT" autorun.log te dice exactamente por que.

Uso (en cron, SEPARADOS por frecuencia):
    python -m src.autorun registrar    # cada ~15 min (para caer en la ventana de cada partido)
    python -m src.autorun actualizar   # cada hora (resultados)
    python -m src.autorun              # ambos (util a mano)

Necesita API_FOOTBALL_KEY (env o .env). OJO cuota: plan gratis = 100 req/dia; con fixtures
cacheados y registro solo cerca del kickoff, alcanza de sobra para el Mundial.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timedelta, timezone

from src import config
from src.fixtures import _api_get, _codigo_nacion
from src.validacion import actualizar_resultados, registrar

# Filtro de competicion: todas las variantes conocidas del nombre en API-Football.
# Agregar variantes aqui si la API cambia el nombre en algun matchday.
COMPETICIONES_WC = {
    "world cup", "fifa world cup", "copa mundial", "copa mundo",
    "wc 2026", "world cup 2026", "fifa world cup 2026",
}

# Ventana de registro EN MINUTOS antes del kickoff: registramos cuando faltan entre MIN y MAX
# minutos, asi la ALINEACION CONFIRMADA ya esta disponible (la API la trae ~60-75 min antes).
# Requiere cron FRECUENTE (cada ~15 min) para caer en la ventana de cada partido.
MIN_ANTES = 20.0
MAX_ANTES = 45.0
CACHE = config.RAIZ / "data" / "raw" / "api_cache"
LOG_FILE = config.RAIZ / "autorun.log"


def _setup_logging() -> None:
    """Configura logging rotativo (5MB x3) mas consola. Idempotente (se llama desde main)."""
    root = logging.getLogger("dsoccer")
    if root.handlers:
        return  # ya configurado
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    # Rotar en 5MB, guardar 3 backups -> maximo ~15MB de logs
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)


def _es_wc(nombre_liga: str) -> bool:
    """True si el nombre de liga es alguna variante conocida de World Cup."""
    n = nombre_liga.lower()
    return any(pat in n for pat in COMPETICIONES_WC)


def _fixtures_dia(fecha: str) -> list:
    """Fixtures de un dia, CACHEADOS en disco. No cambian (kickoff fijo), asi que se bajan
    una vez por dia y las corridas frecuentes del cron NO gastan cuota de la API."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"dia_{fecha}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    data = _api_get("fixtures", {"date": fecha})
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def registrar_proximos() -> None:
    """Registra los partidos que arrancan en la ventana [MIN_ANTES, MAX_ANTES] minutos.

    AUDIT: loguea CADA partido de World Cup con la decision tomada. Si un partido no se
    notifico, 'grep AUDIT autorun.log' te dice exactamente por que.
    """
    log = logging.getLogger("dsoccer.autorun")
    ahora = datetime.now(timezone.utc)
    fechas = [(ahora + timedelta(days=d)).strftime("%Y-%m-%d") for d in (0, 1)]

    ids_en_ventana = []
    for fecha in fechas:
        todos = _fixtures_dia(fecha)
        wc = [f for f in todos if _es_wc(f["league"]["name"])]
        log.info("DIA %s: %d fixtures totales, %d World Cup (liga coincide con patron)",
                 fecha, len(todos), len(wc))

        for f in todos:
            liga = f["league"]["name"]
            if not _es_wc(liga):
                continue  # no WC -> silencio (no spammear el log con 50 ligas de clubes)

            nl, nv = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
            fid = f["fixture"]["id"]

            try:
                ko = datetime.fromisoformat(f["fixture"]["date"])
            except (ValueError, KeyError):
                log.warning("AUDIT fid=%s %s vs %s | liga='%s' | fecha invalida='%s' | "
                            "DECISION=fecha-invalida",
                            fid, nl, nv, liga, f["fixture"].get("date", ""))
                continue

            min_faltan = (ko - ahora).total_seconds() / 60.0
            cl = _codigo_nacion(nl)
            cv = _codigo_nacion(nv)

            if not cl or not cv:
                falta = nl if not cl else nv
                log.warning("AUDIT fid=%s %s vs %s | liga='%s' | min=%.0f | "
                            "DECISION=sin-mapeo '%s'",
                            fid, nl, nv, liga, min_faltan, falta)
                print(f"[autorun] OJO: {nl} vs {nv} en ventana pero '{falta}' SIN MAPEAR "
                      f"-> no se registra. Agregar a fixtures.PAIS_API_A_CODIGO.")
                continue

            if not (MIN_ANTES <= min_faltan <= MAX_ANTES):
                log.info("AUDIT fid=%s %s vs %s | liga='%s' | min=%.0f | "
                         "DECISION=fuera-ventana [%g-%g]",
                         fid, nl, nv, liga, min_faltan, MIN_ANTES, MAX_ANTES)
                continue

            log.info("AUDIT fid=%s %s vs %s | liga='%s' | min=%.0f | DECISION=EN-VENTANA",
                     fid, nl, nv, liga, min_faltan)
            ids_en_ventana.append(fid)

    print(f"[autorun] {ahora:%Y-%m-%d %H:%M} UTC | {len(ids_en_ventana)} partidos en ventana "
          f"({MIN_ANTES:.0f}-{MAX_ANTES:.0f} min al inicio)")
    log.info("RESUMEN: %d partidos en ventana para registrar", len(ids_en_ventana))

    for fid in ids_en_ventana:
        try:
            datos = registrar(fid)  # se auto-saltea si ya estaba logueado (devuelve None)
            if datos:  # se registro por primera vez -> reporte completo a Telegram
                from src.telegram_alert import enviar_reporte_partido
                enviar_reporte_partido(datos["info"], datos["cuotas"])
                log.info("REGISTRADO fid=%d -> Telegram enviado", fid)
            else:
                log.info("SKIP fid=%d (ya registrado anteriormente)", fid)
        except Exception as e:
            log.error("ERROR fid=%d: %s: %s", fid, type(e).__name__, e)
            print(f"[autorun] error con fixture {fid}: {type(e).__name__}: {e}")


def main() -> None:
    from src import config
    config.cargar_env()  # carga TELEGRAM_TOKEN/CHAT_ID (y API key si esta en .env)
    _setup_logging()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if cmd in ("todo", "actualizar"):
        try:
            actualizar_resultados()
        except Exception as e:
            logging.getLogger("dsoccer.autorun").error("error en actualizar: %s: %s",
                                                       type(e).__name__, e)
            print(f"[autorun] error en actualizar: {type(e).__name__}: {e}")
    if cmd in ("todo", "registrar"):
        registrar_proximos()


if __name__ == "__main__":
    main()
