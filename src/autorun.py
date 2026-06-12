"""Punto de entrada DESATENDIDO para el servidor (cron). Acumula datos de validacion solo.

Hace dos cosas:
  1. actualizar : baja el resultado real de los partidos ya logueados que terminaron.
  2. registrar  : registra los partidos que arrancan en ~20-45 min (ya con alineacion
                  confirmada), predice, loguea y manda el reporte completo a Telegram.

Asi el predicciones_log.csv crece dia a dia = tu dataset de validacion del Mundial, sin
tocar nada. Solo usa la API-Football + los datos commiteados (NO scrapea, NO necesita Chrome).
Los fixtures del dia se cachean -> las corridas frecuentes casi no gastan cuota.

Uso (en cron, SEPARADOS por frecuencia):
    python -m src.autorun registrar    # cada ~15 min (para caer en la ventana de cada partido)
    python -m src.autorun actualizar   # cada hora (resultados)
    python -m src.autorun              # ambos (util a mano)

Necesita API_FOOTBALL_KEY (env o .env). OJO cuota: plan gratis = 100 req/dia; con fixtures
cacheados y registro solo cerca del kickoff, alcanza de sobra para el Mundial.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

from src import config
from src.fixtures import _api_get, _codigo_nacion
from src.validacion import actualizar_resultados, registrar

# Filtro de competicion por nombre de liga en API-Football (None = todos los internacionales).
COMPETICION = "World Cup"
# Ventana de registro EN MINUTOS antes del kickoff: registramos cuando faltan entre MIN y MAX
# minutos, asi la ALINEACION CONFIRMADA ya esta disponible (la API la trae ~60-75 min antes).
# Requiere cron FRECUENTE (cada ~15 min) para caer en la ventana de cada partido.
MIN_ANTES = 20.0
MAX_ANTES = 45.0
CACHE = config.RAIZ / "data" / "raw" / "api_cache"


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

    Asi se captura la ALINEACION CONFIRMADA (no el XI por calidad). Pensado para cron cada
    ~15 min: cada partido cae en la ventana y se registra una sola vez, cerca del inicio.
    """
    ahora = datetime.now(timezone.utc)
    fechas = [(ahora + timedelta(days=d)).strftime("%Y-%m-%d") for d in (0, 1)]

    ids = []
    for fecha in fechas:
        for f in _fixtures_dia(fecha):
            if COMPETICION.lower() not in f["league"]["name"].lower():
                continue
            try:
                ko = datetime.fromisoformat(f["fixture"]["date"])
            except (ValueError, KeyError):
                continue
            min_faltan = (ko - ahora).total_seconds() / 60.0
            if MIN_ANTES <= min_faltan <= MAX_ANTES:
                cl = _codigo_nacion(f["teams"]["home"]["name"])
                cv = _codigo_nacion(f["teams"]["away"]["name"])
                if cl and cv:  # solo si sabemos mapear ambas selecciones
                    ids.append(f["fixture"]["id"])

    print(f"[autorun] {ahora:%Y-%m-%d %H:%M} UTC | {len(ids)} partidos en ventana "
          f"({MIN_ANTES:.0f}-{MAX_ANTES:.0f} min al inicio)")
    for fid in ids:
        try:
            datos = registrar(fid)  # se auto-saltea si ya estaba logueado (devuelve None)
            if datos:  # se registro por primera vez -> reporte completo a Telegram
                from src.telegram_alert import enviar_reporte_partido
                enviar_reporte_partido(datos["info"], datos["cuotas"])
        except Exception as e:
            print(f"[autorun] error con fixture {fid}: {type(e).__name__}: {e}")


def main() -> None:
    from src import config
    config.cargar_env()  # carga TELEGRAM_TOKEN/CHAT_ID (y API key si esta en .env)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if cmd in ("todo", "actualizar"):
        try:
            actualizar_resultados()
        except Exception as e:
            print(f"[autorun] error en actualizar: {type(e).__name__}: {e}")
    if cmd in ("todo", "registrar"):
        registrar_proximos()


if __name__ == "__main__":
    main()
