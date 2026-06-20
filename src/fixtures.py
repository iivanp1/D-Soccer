"""Automatizacion de inputs: trae partidos, arbitro y alineaciones de API-Football
y los empareja con nuestro jugadores.csv para alimentar el motor sin tipear nada.

Dos piezas:
  1. CRUCE DE NOMBRES (rapidfuzz) -- la parte dificil, TESTEABLE sin API. Empareja los
     nombres que devuelve la API (ej. "Cristiano Ronaldo dos Santos Aveiro") con los de
     nuestro dataset FBref (ej. "Cristiano Ronaldo"). Lo que no matchea -> lo imputa el
     motor con rating sombra (no se pierde).
  2. CLIENTE API-Football (pendiente de verificar contra una respuesta real). Lee la key
     de la variable de entorno API_FOOTBALL_KEY (NUNCA hardcodear ni commitear la key).

Setup de la key (gratis):
  1. Crear cuenta en https://www.api-football.com/  (o RapidAPI).
  2. Copiar la API key del dashboard.
  3. Setearla como variable de entorno:
       Windows (PowerShell):  $env:API_FOOTBALL_KEY = "tu_key"
       Linux/Mac:             export API_FOOTBALL_KEY="tu_key"

Uso:
    python -m src.fixtures                 # sin key: corre el self-test del cruce de nombres
    python -m src.fixtures --dia 2026-06-11  # con key: lista los partidos de ese dia
"""

from __future__ import annotations

import logging
import os
import sys
import time

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from src import config
from src.jugadores_model import _norm

logger = logging.getLogger("dsoccer.fixtures")

API_BASE = "https://v3.football.api-sports.io"
UMBRAL_MATCH = 82  # score minimo (0-100) para aceptar un emparejamiento de nombres


# =========================================================================== #
#  1. CRUCE DE NOMBRES  (testeable sin API)
# =========================================================================== #
def _candidatos_por_nacion(jugadores: pd.DataFrame, nacion: str) -> dict[str, str]:
    """{nombre_normalizado: nombre_real} de los jugadores de una nacion en el dataset."""
    sub = jugadores[jugadores["nacion"] == nacion]
    return {_norm(n): n for n in sub["player"].dropna().unique()}


def emparejar_nombre(nombre_api: str, candidatos: dict[str, str],
                     umbral: int = UMBRAL_MATCH) -> tuple[str | None, float]:
    """Empareja un nombre de la API con el mejor candidato del dataset (fuzzy).

    Devuelve (nombre_real, score) o (None, score) si ningun candidato supera el umbral.
    """
    if not candidatos:
        return None, 0.0
    m = process.extractOne(_norm(nombre_api), list(candidatos.keys()), scorer=fuzz.WRatio)
    if m and m[1] >= umbral:
        return candidatos[m[0]], m[1]
    return None, (m[1] if m else 0.0)


def armar_xi(nombres_api: list[str], nacion: str, jugadores: pd.DataFrame) -> dict:
    """Convierte la lista de nombres de la API en los nombres reales del dataset.

    Los que no matchean quedan fuera (el motor los completa con rating sombra de la nacion).
    """
    candidatos = _candidatos_por_nacion(jugadores, nacion)
    encontrados, sin_match = [], []
    for nombre in nombres_api:
        real, score = emparejar_nombre(nombre, candidatos)
        (encontrados if real else sin_match).append((nombre, real, round(score, 1)))
    return {
        "nacion": nacion,
        "xi_real": [real for _, real, _ in encontrados],
        "matcheados": encontrados,
        "sin_match": sin_match,  # estos los imputa el motor (sombra)
    }


def _self_test_matching() -> None:
    """Prueba el cruce de nombres OFFLINE con nombres tipo-API (sin tocar la API)."""
    jugadores = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    casos = {
        "POR": ["Cristiano Ronaldo dos Santos Aveiro", "Bruno Miguel Borges Fernandes",
                "Rafael Leao", "Bernardo Mota Veiga de Carvalho e Silva", "Vitinha",
                "Ruben Dias", "Joao Cancelo", "Diogo Jota"],
        "NGA": ["Victor Osimhen", "Ademola Lookman", "Alex Iwobi", "Wilfred Ndidi",
                "Victor Boniface", "Samuel Chukwueze"],
    }
    print("=== SELF-TEST cruce de nombres (API -> dataset FBref) ===\n")
    for nacion, nombres in casos.items():
        r = armar_xi(nombres, nacion, jugadores)
        print(f"[{nacion}]  {len(r['xi_real'])}/{len(nombres)} emparejados")
        for nombre_api, real, score in r["matcheados"]:
            print(f"   OK  {nombre_api:<42} -> {real}  ({score})")
        for nombre_api, _, score in r["sin_match"]:
            print(f"   --  {nombre_api:<42} -> (sin match, score {score}) -> sombra")
        print()


# =========================================================================== #
#  2. CLIENTE API-Football  (pendiente de verificar contra respuesta real)
# =========================================================================== #
def _api_get(path: str, params: dict, reintentos: int = 3) -> list:
    """Cliente API-Football con reintentos, manejo de rate-limit y degradacion graceful.

    Ante errores de red o 429, reintenta con backoff exponencial. Si falla los 3 intentos,
    loguea el error y devuelve [] (NO lanza excepcion) para no tumbar el cron.
    Llama al caller con RuntimeError SOLO si falta la API key (error de config, no de red).
    """
    key = os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("Falta API_FOOTBALL_KEY. Ver instrucciones en el encabezado del archivo.")
    for intento in range(reintentos):
        try:
            resp = requests.get(
                f"{API_BASE}/{path}", headers={"x-apisports-key": key},
                params=params, timeout=20,
            )
            if resp.status_code == 429:
                espera = 30 * (intento + 1)
                logger.warning("rate-limit 429 en /%s intento %d/%d, esperando %ds",
                               path, intento + 1, reintentos, espera)
                time.sleep(espera)
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("errors"):
                logger.error("API-Football errores en /%s %s: %s", path, params, data["errors"])
                return []
            return data.get("response", [])
        except requests.Timeout:
            logger.warning("timeout en /%s intento %d/%d", path, intento + 1, reintentos)
        except requests.RequestException as e:
            logger.warning("error de red en /%s intento %d/%d: %s", path, intento + 1, reintentos, e)
        if intento < reintentos - 1:
            time.sleep(5 * (intento + 1))
    logger.error("DROP /%s %s tras %d intentos -> devolviendo []", path, params, reintentos)
    return []


def partidos_del_dia(fecha: str) -> list[dict]:
    """Fixtures de una fecha (YYYY-MM-DD). Trae equipos, arbitro y venue.

    NOTA: la forma exacta del JSON hay que confirmarla con una respuesta real; los campos
    de abajo siguen el esquema documentado de API-Football v3.
    """
    out = []
    for f in _api_get("fixtures", {"date": fecha}):
        out.append({
            "fixture_id": f["fixture"]["id"],
            "fecha": f["fixture"]["date"],
            "arbitro": f["fixture"].get("referee"),
            "venue": (f["fixture"].get("venue") or {}).get("name"),
            "local": f["teams"]["home"]["name"],
            "visitante": f["teams"]["away"]["name"],
        })
    return out


def alineaciones(fixture_id: int) -> dict[str, list[str]]:
    """Alineaciones titulares (startXI) de un fixture. Disponibles ~1h antes del partido."""
    res = {}
    for equipo in _api_get("fixtures/lineups", {"fixture": fixture_id}):
        nombre_equipo = equipo["team"]["name"]
        res[nombre_equipo] = [p["player"]["name"] for p in equipo.get("startXI", [])]
    return res


# Nombre de seleccion (como lo da API-Football, en ingles) -> codigo FBref (el que usa
# el motor). Ampliable; si falta una nacion, se avisa. Claves normalizadas para robustez.
PAIS_API_A_CODIGO = {
    "portugal": "POR", "nigeria": "NGA", "mexico": "MEX", "south africa": "RSA",
    "south korea": "KOR", "korea republic": "KOR", "czech republic": "CZE", "czechia": "CZE",
    "canada": "CAN", "bosnia and herzegovina": "BIH", "usa": "USA", "united states": "USA",
    "paraguay": "PAR", "qatar": "QAT", "switzerland": "SUI", "brazil": "BRA",
    "morocco": "MAR", "haiti": "HAI", "scotland": "SCO", "argentina": "ARG", "france": "FRA",
    "england": "ENG", "spain": "ESP", "germany": "GER", "italy": "ITA", "netherlands": "NED",
    "belgium": "BEL", "croatia": "CRO", "uruguay": "URU", "colombia": "COL", "japan": "JPN",
    "senegal": "SEN", "ghana": "GHA", "egypt": "EGY", "cameroon": "CMR", "ecuador": "ECU",
    # Resto de UEFA (Eurocopa) para validacion historica
    "hungary": "HUN", "albania": "ALB", "slovenia": "SVN", "slovakia": "SVK",
    "romania": "ROU", "georgia": "GEO", "serbia": "SRB", "denmark": "DEN",
    "poland": "POL", "austria": "AUT", "ukraine": "UKR", "turkey": "TUR",
    "turkiye": "TUR", "wales": "WAL", "sweden": "SWE", "norway": "NOR", "slovakia ": "SVK",
    # Resto de CONMEBOL / CONCACAF (Copa America)
    "peru": "PER", "chile": "CHI", "venezuela": "VEN", "bolivia": "BOL",
    "panama": "PAN", "costa rica": "CRC",
    # --- Mundial 2026: resto de selecciones de TODAS las confederaciones ---
    # AFC
    "australia": "AUS", "saudi arabia": "KSA", "iran": "IRN", "ir iran": "IRN",
    "uzbekistan": "UZB", "jordan": "JOR", "iraq": "IRQ", "united arab emirates": "UAE",
    # CAF
    "ivory coast": "CIV", "cote d'ivoire": "CIV", "tunisia": "TUN", "algeria": "ALG",
    "cape verde islands": "CPV", "cape verde": "CPV", "dr congo": "COD", "congo dr": "COD",
    "mali": "MLI",
    # CONCACAF
    "jamaica": "JAM", "honduras": "HON", "curacao": "CUW",
    # OFC
    "new zealand": "NZL",
}


def _codigo_nacion(nombre_api: str) -> str | None:
    # La API usa '&' (ej. "Bosnia & Herzegovina"); nuestro mapeo usa "and". Normalizamos.
    n = _norm(nombre_api).replace(" & ", " and ").replace("&", "and")
    return PAIS_API_A_CODIGO.get(n)


def correr_partido_auto(fixture_id: int, n_sims: int = 10000,
                        anclar: bool = True, ancla: tuple[float, float] | None = None) -> None:
    """Pipeline 100% automatico: baja fixture + arbitro + alineaciones de API-Football,
    empareja los nombres con el dataset y se lo pasa al motor. Sin tipear nada.

    Si las alineaciones aun no estan (faltan >1h), cae al XI por calidad (seleccion_probable).

    Ancla de Pinnacle: si anclar=True (default) y no se pasa una 'ancla' precomputada,
    baja las cuotas sharp y deduce el lambda implicito del mercado para interpolar los
    goles del modelo (ver src/valor.py). Si Pinnacle no cotiza el partido -> modelo puro.
    """
    from src.mundial_engine import correr

    # 1. Fixture: equipos + arbitro
    info = _api_get("fixtures", {"id": fixture_id})
    if not info:
        print(f"No encontre el fixture {fixture_id}.")
        return
    f = info[0]
    nom_l, nom_v = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
    arbitro = f["fixture"].get("referee")
    cod_l, cod_v = _codigo_nacion(nom_l), _codigo_nacion(nom_v)
    if not cod_l or not cod_v:
        falta = nom_l if not cod_l else nom_v
        print(f"Falta mapear '{falta}' en PAIS_API_A_CODIGO (src/fixtures.py).")
        return
    print(f"Partido: {nom_l} ({cod_l}) vs {nom_v} ({cod_v})  | arbitro: {arbitro or 'sin asignar'}")

    # 2. Alineaciones (si ya estan); emparejar nombres -> XI real
    jugadores = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    lineups = alineaciones(fixture_id)
    xi_l = xi_v = None
    if lineups:
        for nombre_equipo, nombres in lineups.items():
            cod = _codigo_nacion(nombre_equipo)
            r = armar_xi(nombres, cod, jugadores)
            print(f"  {nombre_equipo}: {len(r['xi_real'])}/{len(nombres)} emparejados "
                  f"({len(r['sin_match'])} -> sombra)")
            if cod == cod_l:
                xi_l = r["xi_real"]
            elif cod == cod_v:
                xi_v = r["xi_real"]
    else:
        print("  (alineaciones aun no disponibles -> uso XI por calidad)")

    # 2b. Ancla de Pinnacle: deduce el lambda implicito del mercado (si no vino precomputada).
    # Guardamos el mercado bajado para que el caller (validacion.registrar) lo REUSE sin
    # volver a pegarle a /odds (ahorra cuota; plan gratis = 100 req/dia).
    mercado_ancla = None
    if ancla is None and anclar:
        try:
            from src.valor import cuotas_mercado, lambda_pinnacle
            mercado_ancla = cuotas_mercado(fixture_id)
            ancla = lambda_pinnacle(mercado_ancla)
        except Exception as e:
            logger.warning("ancla Pinnacle no disponible para %s: %s", fixture_id, e)
            ancla = None
    if ancla:
        print(f"  Ancla Pinnacle: lam {ancla[0]:.2f}-{ancla[1]:.2f}")
    elif anclar:
        print("  (Pinnacle no cotiza este fixture -> sin ancla, modelo puro)")

    # 3. Al motor (correr completa con sombra los que falten)
    print()
    res = correr(cod_l, cod_v, xi_l, xi_v, arbitro, n_sims, ancla_pinnacle=ancla)
    return {
        "fixture_id": fixture_id, "fecha": f["fixture"]["date"],
        "local": nom_l, "visitante": nom_v, "cod_l": cod_l, "cod_v": cod_v,
        "arbitro": arbitro, "res": res, "cuotas": mercado_ancla,
    }


def main() -> None:
    if "--dia" in sys.argv:
        fecha = sys.argv[sys.argv.index("--dia") + 1]
        for p in partidos_del_dia(fecha):
            print(f"{p['fecha']}  {p['local']} vs {p['visitante']}  "
                  f"| arbitro: {p['arbitro']}  | {p['venue']}  (id {p['fixture_id']})")
    elif "--partido" in sys.argv:
        fixture_id = int(sys.argv[sys.argv.index("--partido") + 1])
        correr_partido_auto(fixture_id)
    else:
        # Sin args: corre el self-test del cruce de nombres (no necesita key).
        _self_test_matching()


if __name__ == "__main__":
    main()
