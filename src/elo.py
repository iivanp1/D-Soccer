"""Ratings Elo de selecciones (World Football Elo, eloratings.net).

El Motor Mundialista bottom-up SUB-DIFERENCIA (suma de jugadores comprime la brecha real).
El Elo de selecciones, en cambio, encodea decadas de resultados y diferencia bien
(ej. Mexico 1875 vs Sudafrica 1517 -> ~89% Mexico). Lo usamos como COLUMNA VERTEBRAL
de la fuerza de equipo, y el modelo de jugadores queda como AJUSTE por alineacion.

Fuente: eloratings.net publica dos TSV gratis:
  - en.teams.tsv : codigo -> nombre de pais
  - World.tsv    : ranking actual (overall / ofensivo / defensivo por seleccion)

Mapeamos el nombre de pais -> nuestro codigo FBref (reusando el de fixtures.py).

Uso:
    from src.elo import cargar_elo
    elo = cargar_elo()          # {'MEX': {'overall':1875,'off':..,'def':..}, ...}
    python -m src.elo           # demo: top selecciones + prob implicita de un cruce
    python -m src.elo --refrescar
"""

from __future__ import annotations

import json
import sys

import requests

from src import config
from src.fixtures import PAIS_API_A_CODIGO
from src.jugadores_model import _norm

ELO_BASE = "https://www.eloratings.net"
CACHE = config.DATA_PROC / "elo.json"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def descargar_elo() -> dict:
    """Baja y parsea los TSV de eloratings.net -> {codigo_FBref: {overall, off, def}}."""
    teams = requests.get(f"{ELO_BASE}/en.teams.tsv", timeout=20, headers=_HEADERS).text
    code2name = {}
    for linea in teams.split("\n"):
        p = linea.split("\t")
        if len(p) >= 2:
            code2name[p[0]] = p[1]

    world = requests.get(f"{ELO_BASE}/World.tsv", timeout=20, headers=_HEADERS).text
    out = {}
    for linea in world.split("\n"):
        p = linea.split("\t")
        if len(p) < 8 or p[2] not in code2name:
            continue
        cod = PAIS_API_A_CODIGO.get(_norm(code2name[p[2]]))  # nombre pais -> codigo FBref
        if not cod:
            continue
        try:
            out[cod] = {"overall": float(p[3]), "off": float(p[5]), "def": float(p[7])}
        except ValueError:
            continue
    return out


def cargar_elo(refrescar: bool = False) -> dict:
    """Elo cacheado en data/processed/elo.json; lo refresca si se pide o no existe."""
    if CACHE.exists() and not refrescar:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    data = descargar_elo()
    CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def prob_elo(cod_local: str, cod_visit: str, elo: dict | None = None,
             ventaja_local: float = 0.0) -> float:
    """Probabilidad (sin empate) del local segun la formula Elo estandar."""
    elo = elo or cargar_elo()
    dr = elo[cod_local]["overall"] - elo[cod_visit]["overall"] + ventaja_local
    return 1.0 / (1.0 + 10 ** (-dr / 400))


if __name__ == "__main__":
    refrescar = "--refrescar" in sys.argv
    elo = cargar_elo(refrescar=refrescar)
    print(f"Selecciones con Elo mapeado a nuestros codigos: {len(elo)}")
    top = sorted(elo.items(), key=lambda kv: -kv[1]["overall"])[:10]
    print("\nTop 10:")
    for cod, v in top:
        print(f"  {cod}  overall {v['overall']:.0f}  (atk {v['off']:.0f} / def {v['def']:.0f})")
    for a, b in [("MEX", "RSA"), ("POR", "NGA")]:
        if a in elo and b in elo:
            print(f"\n  {a} vs {b}: P({a}) ~ {prob_elo(a, b, elo)*100:.0f}% "
                  f"(modelo bottom-up daba mucho menos)")
