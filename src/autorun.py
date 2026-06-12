"""Punto de entrada DESATENDIDO para el servidor (cron). Acumula datos de validacion solo.

Hace dos cosas:
  1. actualizar : baja el resultado real de los partidos ya logueados que terminaron.
  2. registrar  : busca los partidos que arrancan dentro de las proximas HORAS_ANTES horas
                  (asi las alineaciones suelen estar disponibles) y registra la prediccion.

Asi el predicciones_log.csv crece dia a dia = tu dataset de validacion del Mundial, sin
tocar nada. Solo usa la API-Football + los datos commiteados (NO scrapea, NO necesita Chrome).

Uso:
    python -m src.autorun              # actualizar + registrar (lo que corre el cron)
    python -m src.autorun actualizar
    python -m src.autorun registrar

Necesita API_FOOTBALL_KEY (variable de entorno). Pensado para cron cada 1-2 horas.
OJO cuota: plan gratis = 100 requests/dia; con cron cada 2h alcanza de sobra para el Mundial.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from src.fixtures import _api_get, _codigo_nacion
from src.validacion import actualizar_resultados, registrar

# Filtro de competicion por nombre de liga en API-Football (None = todos los internacionales).
COMPETICION = "World Cup"
# Registrar partidos que arrancan dentro de esta ventana (horas). ~1-2h antes -> hay alineacion.
HORAS_ANTES = 2.0


def registrar_proximos(horas: float = HORAS_ANTES) -> None:
    """Registra la prediccion de los partidos que arrancan dentro de `horas`."""
    ahora = datetime.now(timezone.utc)
    fechas = [(ahora + timedelta(days=d)).strftime("%Y-%m-%d") for d in (0, 1)]

    ids = []
    for fecha in fechas:
        for f in _api_get("fixtures", {"date": fecha}):
            liga = f["league"]["name"]
            if COMPETICION and COMPETICION.lower() not in liga.lower():
                continue
            try:
                ko = datetime.fromisoformat(f["fixture"]["date"])
            except (ValueError, KeyError):
                continue
            faltan = (ko - ahora).total_seconds() / 3600.0
            if 0 < faltan <= horas:
                cl = _codigo_nacion(f["teams"]["home"]["name"])
                cv = _codigo_nacion(f["teams"]["away"]["name"])
                if cl and cv:  # solo si sabemos mapear ambas selecciones
                    ids.append(f["fixture"]["id"])

    print(f"[autorun] {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC | "
          f"{len(ids)} partidos por registrar (proximas {horas}h)")
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
