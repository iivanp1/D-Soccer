"""Cosechador de datos historicos de selecciones desde StatsBomb Open Data.

POR QUE: el log de validacion tiene muy pocos partidos reales (no alcanza para tunear w ni
validar el Motor Mundialista) y la API-Football gratis tope en 100 llamadas/dia. StatsBomb
Open Data es GRATIS, sin API key y SIN limite de cuota (son archivos JSON en GitHub raw), y
trae torneos de selecciones MODERNOS con alineaciones reales, resultados y eventos por jugador
-- incluido xG REAL (shot.statsbomb_xg), el dato que a FBref le falta.

QUE HACE (y que NO):
  - Descarga SELECTIVAMENTE (no clona el repo entero, son varios GB) solo los JSON necesarios:
    competitions -> matches -> lineups -> events, cacheados en data/raw/statsbomb/.
  - Filtra torneos internacionales masculinos modernos (>= --desde, default 2018).
  - Por cada partido: equipos, marcador, XI titular real y stats agregadas por jugador
    (pases, remates, goles, xG, recuperaciones, intercepciones).
  - Guarda todo en SQLite (data/processed/dsoccer_historico.db), idempotente por match_id
    (re-correr NO duplica ni re-descarga). Export opcional a historico_statsbomb.csv.
  - Modulo AISLADO: NO importa ni modifica el modelo ni validacion.py. Solo construye el
    almacen de datos. Conectarlo al modelo es un paso futuro y separado.

ATRIBUCION: datos de StatsBomb Open Data (https://github.com/statsbomb/open-data), gratis
bajo su user agreement con atribucion a StatsBomb.

USO:
    python -m src.ingesta_historica                       # modernos masculinos 2018+, a SQLite
    python -m src.ingesta_historica --desde 2024 --limite 5   # prueba chica
    python -m src.ingesta_historica --export-csv          # ademas exporta el CSV plano
    python -m src.ingesta_historica --rehacer             # re-procesa aunque ya esten (no skip)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone

import requests
from unidecode import unidecode

from src import config

SB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = config.DATA_RAW / "statsbomb"
DB = config.DATA_PROC / "dsoccer_historico.db"
CSV = config.DATA_PROC / "historico_statsbomb.csv"

_SESION = requests.Session()
_SESION.headers.update({"User-Agent": "D-Soccer/1.0 (ingesta historica StatsBomb open-data)"})


def _normalizar(nombre: str) -> str:
    """Identica a jugadores_model._norm (sin acentos, minusculas) -> mismo espacio de nombres
    que jugadores.csv para poder joinear a futuro. Replicada aca para no acoplar al modelo."""
    return unidecode(str(nombre)).strip().lower()


# --------------------------------------------------------------------------- #
#  Descarga con cache (idempotente: re-correr no re-descarga)
# --------------------------------------------------------------------------- #
def _get_json(rel_path: str, reintentos: int = 3) -> object:
    p = CACHE / rel_path
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    err = None
    for intento in range(reintentos):
        try:
            r = _SESION.get(f"{SB_RAW}/{rel_path}", timeout=30)
            r.raise_for_status()
            datos = r.json()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.3)  # educado con GitHub raw (solo tras descarga real)
            return datos
        except requests.RequestException as e:
            err = e
            time.sleep(1.5 * (intento + 1))  # backoff
    raise RuntimeError(f"No pude bajar {rel_path}: {err}")


# --------------------------------------------------------------------------- #
#  Parsers (campos verificados contra el repo real)
# --------------------------------------------------------------------------- #
def _competiciones(desde_anio: int, genero: str) -> list[dict]:
    """Torneos internacionales del genero pedido con season_name >= desde_anio."""
    out = []
    for c in _get_json("competitions.json"):
        if (c.get("competition_international") and c.get("competition_gender") == genero):
            try:
                anio = int(str(c["season_name"])[:4])
            except (ValueError, KeyError):
                continue
            if anio >= desde_anio:
                out.append(c)
    # dedup por (competition_id, season_id) preservando orden
    visto, unicos = set(), []
    for c in out:
        k = (c["competition_id"], c["season_id"])
        if k not in visto:
            visto.add(k)
            unicos.append(c)
    return unicos


def _partidos(comp_id: int, season_id: int) -> list[dict]:
    return _get_json(f"matches/{comp_id}/{season_id}.json")


def _starting_xi(events: list) -> tuple[dict, set]:
    """Devuelve ({equipo: [(player_id, player, posicion)]}, set de player_id titulares)."""
    xis, titulares = {}, set()
    for ev in events:
        if ev.get("type", {}).get("name") != "Starting XI":
            continue
        equipo = ev.get("team", {}).get("name")
        once = []
        for j in ev.get("tactics", {}).get("lineup", []):
            pid = j.get("player", {}).get("id")
            once.append((pid, j.get("player", {}).get("name"), j.get("position", {}).get("name")))
            if pid is not None:
                titulares.add(pid)
        xis[equipo] = once
    return xis, titulares


def _squad(lineups: list) -> tuple[list, dict]:
    """De lineups/{match}.json: lista (equipo, player_id, player, posicion) de TODOS los
    jugadores convocados (titulares + suplentes) y un mapa {player_id: equipo}."""
    plantel, jug_equipo = [], {}
    for eq in lineups:
        equipo = eq.get("team_name")
        for j in eq.get("lineup", []):
            pid = j.get("player_id")
            nombre = j.get("player_name")
            posiciones = j.get("positions") or []
            pos = posiciones[0].get("position") if posiciones else None
            plantel.append((equipo, pid, nombre, pos))
            if pid is not None:
                jug_equipo[pid] = equipo
    return plantel, jug_equipo


def _stats_jugador(events: list) -> dict:
    """Agrega por player_id en UNA pasada: pases/completados, remates, goles, xG,
    recuperaciones, intercepciones. Nombre del jugador tomado del evento."""
    acc: dict[int, dict] = {}

    def fila(pid, nombre):
        f = acc.get(pid)
        if f is None:
            f = {"player": nombre, "pases": 0, "pases_completados": 0, "remates": 0,
                 "goles": 0, "xg": 0.0, "recuperaciones": 0, "intercepciones": 0}
            acc[pid] = f
        return f

    for ev in events:
        pid = ev.get("player", {}).get("id")
        if pid is None:
            continue
        tipo = ev.get("type", {}).get("name")
        f = fila(pid, ev.get("player", {}).get("name"))
        if tipo == "Pass":
            f["pases"] += 1
            if ev.get("pass", {}).get("outcome") is None:  # SB marca outcome solo si fallo
                f["pases_completados"] += 1
        elif tipo == "Shot":
            shot = ev.get("shot", {})
            f["remates"] += 1
            f["xg"] += float(shot.get("statsbomb_xg") or 0.0)
            if shot.get("outcome", {}).get("name") == "Goal":
                f["goles"] += 1
        elif tipo == "Ball Recovery":
            f["recuperaciones"] += 1
        elif tipo == "Interception":
            f["intercepciones"] += 1
    return acc


# --------------------------------------------------------------------------- #
#  SQLite (idempotente por PRIMARY KEY)
# --------------------------------------------------------------------------- #
def _crear_tablas(con: sqlite3.Connection) -> None:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS partidos (
        match_id INTEGER PRIMARY KEY, comp_id INTEGER, competicion TEXT, season TEXT,
        fecha TEXT, etapa TEXT, equipo_local TEXT, equipo_visitante TEXT,
        goles_local INTEGER, goles_visitante INTEGER, resultado TEXT, ingestado_en TEXT);
    CREATE TABLE IF NOT EXISTS alineaciones (
        match_id INTEGER, equipo TEXT, es_local INTEGER, player_id INTEGER,
        player TEXT, player_norm TEXT, posicion TEXT, es_titular INTEGER,
        PRIMARY KEY (match_id, player_id));
    CREATE TABLE IF NOT EXISTS jugador_partido_stats (
        match_id INTEGER, player_id INTEGER, player TEXT, player_norm TEXT, equipo TEXT,
        pases INTEGER, pases_completados INTEGER, remates INTEGER, goles INTEGER,
        xg REAL, recuperaciones INTEGER, intercepciones INTEGER,
        PRIMARY KEY (match_id, player_id));
    """)
    con.commit()


def _partido_existe(con: sqlite3.Connection, match_id: int) -> bool:
    return con.execute("SELECT 1 FROM partidos WHERE match_id=?", (match_id,)).fetchone() is not None


def _resultado(gl, gv) -> str:
    if gl is None or gv is None:
        return ""
    return "H" if gl > gv else ("A" if gv > gl else "D")


def _procesar_partido(con: sqlite3.Connection, m: dict, comp: dict) -> None:
    mid = m["match_id"]
    local = m["home_team"]["home_team_name"]
    visit = m["away_team"]["away_team_name"]
    gl, gv = m.get("home_score"), m.get("away_score")

    lineups = _get_json(f"lineups/{mid}.json")
    events = _get_json(f"events/{mid}.json")
    plantel, jug_equipo = _squad(lineups)
    _, titulares = _starting_xi(events)
    stats = _stats_jugador(events)

    con.execute(
        "INSERT OR REPLACE INTO partidos VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, comp["competition_id"], comp["competition_name"], str(comp["season_name"]),
         m.get("match_date"), m.get("competition_stage", {}).get("name"),
         local, visit, gl, gv, _resultado(gl, gv),
         datetime.now(timezone.utc).isoformat(timespec="seconds")))

    for equipo, pid, nombre, pos in plantel:
        if pid is None:
            continue
        con.execute(
            "INSERT OR REPLACE INTO alineaciones VALUES (?,?,?,?,?,?,?,?)",
            (mid, equipo, 1 if equipo == local else 0, pid, nombre, _normalizar(nombre or ""),
             pos, 1 if pid in titulares else 0))

    for pid, f in stats.items():
        con.execute(
            "INSERT OR REPLACE INTO jugador_partido_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, pid, f["player"], _normalizar(f["player"] or ""), jug_equipo.get(pid),
             f["pases"], f["pases_completados"], f["remates"], f["goles"],
             round(f["xg"], 4), f["recuperaciones"], f["intercepciones"]))
    con.commit()


# --------------------------------------------------------------------------- #
#  Orquestador + export
# --------------------------------------------------------------------------- #
def ingestar(desde: int = 2018, genero: str = "male", limite: int | None = None,
             rehacer: bool = False, export_csv: bool = False) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    _crear_tablas(con)

    comps = _competiciones(desde, genero)
    print(f"[ingesta] {len(comps)} torneos {genero} >= {desde}:")
    for c in comps:
        print(f"   - {c['competition_name']} {c['season_name']} "
              f"(comp {c['competition_id']}/season {c['season_id']})")

    nuevos = saltados = errores = 0
    for c in comps:
        partidos = _partidos(c["competition_id"], c["season_id"])
        print(f"\n[{c['competition_name']} {c['season_name']}] {len(partidos)} partidos")
        for m in partidos:
            if limite is not None and nuevos >= limite:
                print(f"\n[ingesta] limite {limite} alcanzado.")
                break
            mid = m["match_id"]
            if not rehacer and _partido_existe(con, mid):
                saltados += 1
                continue
            try:
                _procesar_partido(con, m, c)
                nuevos += 1
                print(f"   OK {m['home_team']['home_team_name']} {m.get('home_score')}-"
                      f"{m.get('away_score')} {m['away_team']['away_team_name']}  (id {mid})")
            except Exception as e:
                errores += 1
                print(f"   ERROR match {mid}: {type(e).__name__}: {e}")
        if limite is not None and nuevos >= limite:
            break

    total = con.execute("SELECT count(*) FROM partidos").fetchone()[0]
    con.close()
    print(f"\n[ingesta] nuevos={nuevos} saltados(ya estaban)={saltados} errores={errores} "
          f"| total en DB={total}")
    if export_csv:
        exportar_csv()


def exportar_csv() -> None:
    """Vista denormalizada (una fila por jugador-partido) -> historico_statsbomb.csv."""
    import csv as _csv
    con = sqlite3.connect(DB)
    filas = con.execute("""
        SELECT s.match_id, p.competicion, p.season, p.fecha, p.etapa,
               p.equipo_local, p.equipo_visitante, p.goles_local, p.goles_visitante, p.resultado,
               s.equipo, s.player, s.player_norm, s.pases, s.pases_completados, s.remates,
               s.goles, s.xg, s.recuperaciones, s.intercepciones
        FROM jugador_partido_stats s JOIN partidos p ON p.match_id = s.match_id
        ORDER BY p.fecha, s.match_id""").fetchall()
    cols = [d[0] for d in con.execute("SELECT * FROM jugador_partido_stats LIMIT 0").description]
    con.close()
    encabezado = ["match_id", "competicion", "season", "fecha", "etapa", "equipo_local",
                  "equipo_visitante", "goles_local", "goles_visitante", "resultado",
                  "equipo", "player", "player_norm", "pases", "pases_completados", "remates",
                  "goles", "xg", "recuperaciones", "intercepciones"]
    with open(CSV, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(encabezado)
        w.writerows(filas)
    print(f"[ingesta] export CSV: {len(filas)} filas jugador-partido -> {CSV.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingesta historica de selecciones (StatsBomb Open Data).")
    ap.add_argument("--desde", type=int, default=2018, help="anio minimo de temporada (default 2018)")
    ap.add_argument("--limite", type=int, default=None, help="cortar tras N partidos nuevos (prueba)")
    ap.add_argument("--genero", default="male", choices=["male", "female"])
    ap.add_argument("--rehacer", action="store_true", help="re-procesar aunque ya esten en la DB")
    ap.add_argument("--export-csv", action="store_true", help="exportar tambien el CSV plano")
    args = ap.parse_args()
    ingestar(args.desde, args.genero, args.limite, args.rehacer, args.export_csv)


if __name__ == "__main__":
    main()
