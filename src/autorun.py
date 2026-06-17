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
# Ventana de polling de alineaciones para props: arranca antes que la de registro para
# atrapar el XI apenas sale, pero con TECHO en 90 min (la API casi nunca publica antes).
# Pollear a 120/105 min solo quemaba los MAX_INTENTOS en el vacio y dejaba el fixture en
# ERROR justo cuando el XI iba a aparecer. Con techo 90 + MAX_INTENTOS=6 los intentos caen
# en ~90/75/60/45/30/20 min y cubren la franja real de publicacion.
PROPS_MIN_ANTES = 20.0
PROPS_MAX_ANTES = 90.0
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


def registrar_props() -> None:
    """Ventana 20-90 min: detecta WC matches, intenta confirmar XI y calcula Player Props.

    Mas ancha que la ventana de registro (20-45 min) para capturar los lineups en cuanto
    salen (~40-75 min antes), pero con techo 90 min para no quemar intentos antes de que
    exista el XI. Budget: max MAX_INTENTOS API calls por fixture, cero si ya CONFIRMED.
    Solo envia el alerta de props UNA VEZ por fixture (estado PROPS_SENT en disco).
    """
    from src.props_lineups import get_estado, poll_y_cachear, marcar_props_sent
    from src.props_data import cargar as cargar_tiros
    from src.props_model import calcular_props_partido, top_props
    from src.telegram_alert import enviar_reporte_props

    log = logging.getLogger("dsoccer.props")
    ahora = datetime.now(timezone.utc)
    fechas = [(ahora + timedelta(days=d)).strftime("%Y-%m-%d") for d in (0, 1)]

    datos_tiros = cargar_tiros()
    if not datos_tiros:
        log.warning("props: tiros_intl.json no encontrado -> corre python -m src.props_data")
        return

    import pandas as pd
    dfj = pd.read_csv(config.DATA_PROC / "jugadores.csv")

    for fecha in fechas:
        for f in _fixtures_dia(fecha):
            if not _es_wc(f["league"]["name"]):
                continue
            try:
                ko = datetime.fromisoformat(f["fixture"]["date"])
            except (ValueError, KeyError):
                continue

            min_faltan = (ko - ahora).total_seconds() / 60.0
            if not (PROPS_MIN_ANTES <= min_faltan <= PROPS_MAX_ANTES):
                continue

            fid = f["fixture"]["id"]
            nom_l, nom_v = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
            cod_l, cod_v = _codigo_nacion(nom_l), _codigo_nacion(nom_v)
            if not cod_l or not cod_v:
                continue

            est = get_estado(fid)
            if est["status"] in ("props_sent", "error"):
                log.info("props fid=%d %s vs %s: status=%s -> skip",
                         fid, nom_l, nom_v, est["status"])
                continue

            log.info("props fid=%d %s vs %s | min=%.0f | status=%s -> poll",
                     fid, nom_l, nom_v, min_faltan, est["status"])

            nuevo_status = poll_y_cachear(fid, nom_l, nom_v, cod_l, cod_v)
            if nuevo_status != "confirmed":
                continue

            # XI confirmado -> calcular props y enviar
            try:
                from src.props_lineups import get_lineups_confirmados
                lineups = get_lineups_confirmados(fid)
                if not lineups:
                    continue

                # Necesitamos lambda de goles: corremos el engine completo
                from src.validacion import registrar
                datos_reg = registrar(fid)  # registra si no estaba, devuelve None si ya estaba
                if datos_reg is None:
                    # Ya estaba registrado: buscar info del log
                    log.info("props fid=%d: ya registrado, calculando props sin re-registrar", fid)
                    from src.mundial_engine import correr
                    res_engine = correr(cod_l, cod_v, lineups["xi_l"], lineups["xi_v"],
                                        f["fixture"].get("referee"), n_sims=5000)
                    info_props = {
                        "xi_l": lineups["xi_l"], "xi_v": lineups["xi_v"],
                        "cod_l": cod_l, "cod_v": cod_v,
                        "local": nom_l, "visitante": nom_v,
                        "fecha": f["fixture"]["date"],
                        "res": res_engine,
                    }
                else:
                    info_props = datos_reg["info"]
                    info_props["xi_l"] = lineups["xi_l"]
                    info_props["xi_v"] = lineups["xi_v"]

                props = calcular_props_partido(info_props, dfj, datos_tiros)
                top = top_props(props)
                if top:
                    enviar_reporte_props(info_props, props, top)
                    marcar_props_sent(fid)
                    log.info("props fid=%d: alerta enviada (%d jugadores en top)",
                             fid, len(top))
                else:
                    log.info("props fid=%d: sin jugadores con lambda suficiente -> no se envia", fid)
                    marcar_props_sent(fid)  # igual marcamos para no re-intentar

            except Exception as e:
                log.error("props fid=%d: error calculando props: %s: %s",
                          fid, type(e).__name__, e)


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
        registrar_props()      # props primero: ventana mas amplia (20-120 min)
        registrar_proximos()   # reporte completo: ventana 20-45 min


if __name__ == "__main__":
    main()
