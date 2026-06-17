"""State machine de alineaciones para Player Props.

Gestiona el polling de lineups ANTES de la ventana de registro (20-90 min previos
al KO) y los cachea en disco para que el resto del pipeline los reutilice sin llamadas
a la API adicionales.

Estados por fixture:
  UNKNOWN    -> primera vez que lo vemos
  PENDING    -> lo intentamos, la API aun no tiene lineups
  CONFIRMED  -> XI confirmados, cacheados en disco
  PROPS_SENT -> alerta de props ya enviada (no volver a mandar)
  ERROR      -> fallo 3+ veces seguidas -> abandonamos polling

Budget de API: maximo MAX_INTENTOS intentos por fixture (1 call/intento). Una vez
CONFIRMED, cero llamadas adicionales: el resto del pipeline lee del disco.

Uso:
    from src.props_lineups import poll_y_cachear, get_lineups_confirmados
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.fixtures import _codigo_nacion, alineaciones, armar_xi

CACHE = config.RAIZ / "data" / "raw" / "api_cache"
# Max API calls por fixture antes de marcar ERROR. Con la ventana de polling acotada a
# 20-90 min (autorun.PROPS_MAX_ANTES) y cron cada ~15 min, hacen falta ~5-6 intentos para
# cubrir toda la franja en que la API publica el XI. Con 3-4 partidos/dia son <30 calls/dia,
# muy por debajo del limite del plan gratis (100/dia).
MAX_INTENTOS = 6

logger = logging.getLogger("dsoccer.props_lineups")


def _cache_path(fixture_id: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"props_state_{fixture_id}.json"


def get_estado(fixture_id: int) -> dict:
    """Carga el estado del fixture desde disco. Devuelve estado vacío si no existe."""
    p = _cache_path(fixture_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"status": "unknown", "poll_count": 0, "xi_l": None, "xi_v": None,
            "nom_l": "", "nom_v": "", "cod_l": "", "cod_v": ""}


def _set_estado(fixture_id: int, estado: dict) -> None:
    estado["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _cache_path(fixture_id).write_text(
        json.dumps(estado, ensure_ascii=False), encoding="utf-8")


def get_lineups_confirmados(fixture_id: int) -> dict | None:
    """Devuelve {xi_l, xi_v, nom_l, nom_v, cod_l, cod_v} si CONFIRMED/PROPS_SENT, None si no."""
    est = get_estado(fixture_id)
    if est["status"] in ("confirmed", "props_sent") and est["xi_l"] and est["xi_v"]:
        return {k: est[k] for k in ("xi_l", "xi_v", "nom_l", "nom_v", "cod_l", "cod_v")}
    return None


def marcar_props_sent(fixture_id: int) -> None:
    est = get_estado(fixture_id)
    est["status"] = "props_sent"
    _set_estado(fixture_id, est)


def poll_y_cachear(fixture_id: int, nom_l: str, nom_v: str,
                   cod_l: str, cod_v: str) -> str:
    """Intenta obtener el XI de la API y actualiza el estado en disco.

    Devuelve el nuevo estado: 'confirmed', 'pending' o 'error'.
    Usa 1 API call si el estado era unknown/pending (max MAX_INTENTOS veces).
    Si el estado ya es confirmed/props_sent/error, NO llama a la API y devuelve
    el estado actual.
    """
    import pandas as pd
    est = get_estado(fixture_id)
    est["nom_l"], est["nom_v"] = nom_l, nom_v
    est["cod_l"], est["cod_v"] = cod_l, cod_v

    if est["status"] in ("confirmed", "props_sent"):
        return est["status"]
    if est["status"] == "error":
        return "error"

    if est["poll_count"] >= MAX_INTENTOS:
        est["status"] = "error"
        _set_estado(fixture_id, est)
        logger.warning("props fid=%d: max intentos (%d) alcanzados -> ERROR", fixture_id, MAX_INTENTOS)
        return "error"

    # Un intento de lineup (1 API call)
    est["poll_count"] = est.get("poll_count", 0) + 1
    try:
        lineups = alineaciones(fixture_id)  # llama a _api_get("fixtures/lineups", ...)
    except Exception as e:
        logger.warning("props fid=%d: error al pedir lineups intento %d: %s",
                       fixture_id, est["poll_count"], e)
        est["status"] = "pending"
        _set_estado(fixture_id, est)
        return "pending"

    if not lineups:
        est["status"] = "pending"
        _set_estado(fixture_id, est)
        logger.info("props fid=%d: lineups no disponibles aun (intento %d/%d)",
                    fixture_id, est["poll_count"], MAX_INTENTOS)
        return "pending"

    # alineaciones() ya devuelve {nombre_equipo: [nombres_jugadores]} (no la respuesta
    # cruda de la API). Mapeamos cada equipo a su codigo y armamos el XI con armar_xi.
    dfj = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    xi_l, xi_v = [], []
    for nom_api, nombres in lineups.items():
        cod = _codigo_nacion(nom_api)
        r = armar_xi(nombres, cod or "", dfj)
        if cod == cod_l:
            xi_l = r["xi_real"]
        elif cod == cod_v:
            xi_v = r["xi_real"]

    # Si la API aun no publico el XI de AMBOS equipos, seguimos en pending y reintentamos.
    if not xi_l or not xi_v:
        est["status"] = "pending"
        _set_estado(fixture_id, est)
        logger.info("props fid=%d: XI incompleto (%d local, %d visit) -> pending",
                    fixture_id, len(xi_l), len(xi_v))
        return "pending"

    est["xi_l"] = xi_l
    est["xi_v"] = xi_v
    est["status"] = "confirmed"
    _set_estado(fixture_id, est)
    logger.info("props fid=%d: XI CONFIRMADOS (%d local, %d visit)",
                fixture_id, len(xi_l), len(xi_v))
    return "confirmed"
