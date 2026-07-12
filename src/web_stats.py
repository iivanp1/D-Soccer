"""Generador de la PAGINA WEB estatica de D-Soccer: historiales + promedios consultables.

Produce docs/data.json + docs/index.html (autonomo, sin dependencias) desde los datos que
YA tiene el proyecto:
  - EQUIPOS (selecciones): equipo_partido_stats + partidos (StatsBomb: WC22, AFCON23,
    Copa America 24, Euro 24) -> ultimos 7: faltas hechas/recibidas, corners a favor/
    en contra, tarjetas, marcador.
  - EQUIPOS (clubes): partidos.csv (football-data, 7 ligas x 3 temporadas) -> idem.
  - JUGADORES: promedios por 90 de jugadores.csv (tiros, tiros al arco, faltas com/rec)
    + tabla de ULTIMOS 5 partidos internacionales (rival, remates, goles, xG) de
    jugador_partido_stats.

Publicacion (elegir una):
  - GitHub Pages: commitear docs/ y activar Pages en el repo (rama main, carpeta /docs).
  - Server propio:  python -m http.server 8123 --directory docs   (via Tailscale).

El cron del server puede regenerar esto tras cada ingesta (los datos se actualizan solos).

Uso:
    python -m src.web_stats            # genera docs/data.json + docs/index.html
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from src import config

DOCS = config.RAIZ / "docs"
DB = config.DATA_PROC / "dsoccer_historico.db"
N_EQUIPO = 7   # ultimos N partidos por equipo
N_JUGADOR = 5  # ultimos N partidos por jugador
MIN_90S = 3.0  # minimo de 90s en la temporada para publicar promedios de un jugador


# --------------------------------------------------------------------------- #
#  Equipos: selecciones (StatsBomb) y clubes (football-data)
# --------------------------------------------------------------------------- #
def equipos_selecciones() -> dict:
    if not DB.exists():
        return {}
    con = sqlite3.connect(DB)
    q = """SELECT p.match_id, p.fecha, p.competicion, p.equipo_local, p.equipo_visitante,
                  p.goles_local, p.goles_visitante, e.equipo, e.faltas, e.tarjetas, e.corners
           FROM equipo_partido_stats e JOIN partidos p ON p.match_id = e.match_id"""
    df = pd.read_sql_query(q, con)
    con.close()
    # stats del RIVAL en el mismo partido -> faltas recibidas / corners en contra
    riv = df[["match_id", "equipo", "faltas", "corners"]].rename(
        columns={"equipo": "rival_eq", "faltas": "faltas_rec", "corners": "corners_rec"})
    df = df.merge(riv, on="match_id")
    df = df[df["equipo"] != df["rival_eq"]]

    out = {}
    for eq, g in df.groupby("equipo"):
        g = g.sort_values("fecha", ascending=False).head(N_EQUIPO)
        partidos = []
        for _, r in g.iterrows():
            local = r["equipo"] == r["equipo_local"]
            gf, gc = (r["goles_local"], r["goles_visitante"]) if local else (r["goles_visitante"], r["goles_local"])
            partidos.append({
                "fecha": str(r["fecha"])[:10], "rival": r["rival_eq"],
                "torneo": r["competicion"], "gf": int(gf), "gc": int(gc),
                "faltas": _i(r["faltas"]), "faltas_rec": _i(r["faltas_rec"]),
                "corners": _i(r["corners"]), "corners_rec": _i(r["corners_rec"]),
                "tarjetas": _i(r["tarjetas"]),
            })
        out[eq] = {"tipo": "seleccion", "partidos": partidos, "prom": _promedios(partidos)}
    return out


def equipos_clubes() -> dict:
    ruta = config.DATA_PROC / "partidos.csv"
    if not ruta.exists():
        return {}
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    out = {}
    equipos = pd.unique(pd.concat([df["local"], df["visitante"]]))
    for eq in equipos:
        g = df[(df["local"] == eq) | (df["visitante"] == eq)].sort_values(
            "fecha", ascending=False).head(N_EQUIPO)
        partidos = []
        for _, r in g.iterrows():
            local = r["local"] == eq
            lado, rlado = ("local", "visitante") if local else ("visitante", "local")
            partidos.append({
                "fecha": str(r["fecha"])[:10], "rival": r[rlado],
                "torneo": config.LIGAS.get(r["liga"], r["liga"]),
                "gf": _i(r[f"goles_{lado}"]), "gc": _i(r[f"goles_{rlado}"]),
                "faltas": _i(r[f"faltas_{lado}"]), "faltas_rec": _i(r[f"faltas_{rlado}"]),
                "corners": _i(r[f"corners_{lado}"]), "corners_rec": _i(r[f"corners_{rlado}"]),
                "tarjetas": _i(r[f"amarillas_{lado}"]) + 2 * _i(r[f"rojas_{lado}"]),
            })
        out[eq] = {"tipo": "club", "partidos": partidos, "prom": _promedios(partidos)}
    return out


def _i(x) -> int:
    try:
        return int(x) if pd.notna(x) else 0
    except (TypeError, ValueError):
        return 0


def _s(x) -> str:
    """String saneado: NaN/None -> '' (json.dumps serializa NaN y rompe JSON.parse en el browser)."""
    return str(x) if pd.notna(x) else ""


def _promedios(partidos: list[dict]) -> dict:
    n = len(partidos) or 1
    keys = ("gf", "gc", "faltas", "faltas_rec", "corners", "corners_rec", "tarjetas")
    return {k: round(sum(p[k] for p in partidos) / n, 2) for k in keys}


# --------------------------------------------------------------------------- #
#  Jugadores: promedios de club + ultimos 5 internacionales
# --------------------------------------------------------------------------- #
def jugadores() -> dict:
    out = {}
    # 1. Promedios por 90 de la temporada de club mas reciente (jugadores.csv)
    ruta = config.DATA_PROC / "jugadores.csv"
    if ruta.exists():
        df = pd.read_csv(ruta)
        df = df[df["noventas"] >= MIN_90S]
        df = df.sort_values("season", ascending=False).drop_duplicates("player", keep="first")
        for _, r in df.iterrows():
            n90 = float(r["noventas"])
            out[r["player"]] = {
                "equipo": _s(r.get("team")), "liga": _s(r.get("league")),
                "nacion": _s(r.get("nacion")), "pos": _s(r.get("posicion")),
                "prom90": {
                    "tiros": _p90(r.get("tiros"), n90), "tiros_arco": _p90(r.get("tiros_arco"), n90),
                    "goles": _p90(r.get("goles"), n90), "asist": _p90(r.get("asistencias"), n90),
                    "faltas_com": _p90(r.get("faltas_com"), n90), "faltas_rec": _p90(r.get("faltas_rec"), n90),
                },
                "ultimos": [],
            }
    # 2. Ultimos 5 internacionales por jugador (StatsBomb)
    if DB.exists():
        con = sqlite3.connect(DB)
        q = """SELECT j.player, j.equipo, j.remates, j.goles, j.xg, j.pases,
                      p.fecha, p.equipo_local, p.equipo_visitante, p.competicion
               FROM jugador_partido_stats j JOIN partidos p ON p.match_id = j.match_id"""
        dj = pd.read_sql_query(q, con)
        con.close()
        dj["rival"] = dj.apply(
            lambda r: r["equipo_visitante"] if r["equipo"] == r["equipo_local"] else r["equipo_local"], axis=1)
        for pl, g in dj.groupby("player"):
            g = g.sort_values("fecha", ascending=False).head(N_JUGADOR)
            ult = [{"fecha": str(r["fecha"])[:10], "rival": r["rival"], "torneo": r["competicion"],
                    "remates": _i(r["remates"]), "goles": _i(r["goles"]),
                    "xg": round(float(r["xg"] or 0), 2), "pases": _i(r["pases"])}
                   for _, r in g.iterrows()]
            if pl in out:
                out[pl]["ultimos"] = ult
            else:  # jugador internacional sin temporada de club en el dataset
                out[pl] = {"equipo": "", "liga": "", "nacion": "", "pos": "",
                           "prom90": {}, "ultimos": ult}
    return out


def _p90(x, n90: float) -> float | None:
    try:
        return round(float(x) / n90, 2) if pd.notna(x) and n90 > 0 else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
#  Salida: data.json + index.html
# --------------------------------------------------------------------------- #
HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D-Soccer | Historiales y promedios</title>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#e2e8f0;margin:0}
 header{background:#1e293b;padding:14px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 h1{font-size:18px;margin:0;color:#38bdf8}
 input{background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:8px 12px;border-radius:8px;min-width:260px}
 .tabs button{background:#334155;border:0;color:#cbd5e1;padding:8px 14px;border-radius:8px;cursor:pointer;margin-right:6px}
 .tabs button.on{background:#38bdf8;color:#0f172a;font-weight:600}
 main{padding:16px 20px;max-width:1050px;margin:auto}
 table{border-collapse:collapse;width:100%;margin:10px 0 22px;font-size:13.5px}
 th,td{border-bottom:1px solid #24324a;padding:6px 8px;text-align:center}
 th{color:#94a3b8;font-weight:600;background:#16233b}
 td:first-child,th:first-child{text-align:left}
 .card{background:#16233b;border-radius:12px;padding:14px 16px;margin:12px 0}
 .muted{color:#64748b;font-size:12px}
 .prom{color:#38bdf8;font-weight:600}
 .hit{cursor:pointer;padding:6px 10px;border-bottom:1px solid #24324a}
 .hit:hover{background:#1e293b}
</style></head><body>
<header><h1>D-Soccer</h1>
 <div class="tabs"><button id="tE" class="on">Equipos</button><button id="tJ">Jugadores</button></div>
 <input id="q" placeholder="Buscar equipo o jugador..." autocomplete="off">
 <span class="muted" id="gen"></span></header>
<main><div id="hits"></div><div id="out"></div></main>
<script>
let D=null, tab='E';
fetch('data.json').then(r=>r.json()).then(d=>{D=d;document.getElementById('gen').textContent='actualizado: '+d.generado;});
const q=document.getElementById('q'),hits=document.getElementById('hits'),out=document.getElementById('out');
document.getElementById('tE').onclick=()=>setTab('E');document.getElementById('tJ').onclick=()=>setTab('J');
function setTab(t){tab=t;document.getElementById('tE').classList.toggle('on',t=='E');
 document.getElementById('tJ').classList.toggle('on',t=='J');q.value='';hits.innerHTML='';out.innerHTML='';}
q.oninput=()=>{if(!D)return;const s=q.value.toLowerCase();hits.innerHTML='';out.innerHTML='';if(s.length<2)return;
 const src=tab=='E'?D.equipos:D.jugadores;
 Object.keys(src).filter(k=>k.toLowerCase().includes(s)).slice(0,12).forEach(k=>{
  const d=document.createElement('div');d.className='hit';d.textContent=k+(tab=='E'?' ('+src[k].tipo+')':'');
  d.onclick=()=>{hits.innerHTML='';q.value=k;tab=='E'?verEquipo(k):verJugador(k);};hits.appendChild(d);});};
function verEquipo(k){const e=D.equipos[k],p=e.prom;
 let h=`<div class="card"><h2>${k} <span class="muted">(${e.tipo})</span></h2>
 <div>Promedios ultimos ${e.partidos.length}: <span class="prom">${p.gf}-${p.gc} goles | ${p.faltas} faltas (${p.faltas_rec} rec.) | ${p.corners} corners (${p.corners_rec} contra) | ${p.tarjetas} tarj.</span></div>
 <table><tr><th>fecha</th><th>rival</th><th>res</th><th>faltas</th><th>f.rec</th><th>corners</th><th>c.contra</th><th>tarj</th><th>torneo</th></tr>`;
 e.partidos.forEach(x=>{h+=`<tr><td>${x.fecha}</td><td>${x.rival}</td><td>${x.gf}-${x.gc}</td><td>${x.faltas}</td><td>${x.faltas_rec}</td><td>${x.corners}</td><td>${x.corners_rec}</td><td>${x.tarjetas}</td><td class="muted">${x.torneo}</td></tr>`});
 out.innerHTML=h+'</table></div>';}
function verJugador(k){const j=D.jugadores[k],pr=j.prom90||{};
 let h=`<div class="card"><h2>${k}</h2><div class="muted">${j.equipo} | ${j.liga} | ${j.nacion} | ${j.pos}</div>`;
 if(pr.tiros!=null)h+=`<div>Promedios club (por 90): <span class="prom">${pr.tiros} tiros | ${pr.tiros_arco} al arco | ${pr.goles} goles | ${pr.asist} asist | ${pr.faltas_com} faltas com. | ${pr.faltas_rec} rec.</span></div>`;
 if(j.ultimos.length){h+=`<h3>Ultimos ${j.ultimos.length} internacionales</h3>
 <table><tr><th>fecha</th><th>rival</th><th>remates</th><th>goles</th><th>xG</th><th>pases</th><th>torneo</th></tr>`;
 j.ultimos.forEach(x=>{h+=`<tr><td>${x.fecha}</td><td>${x.rival}</td><td>${x.remates}</td><td>${x.goles}</td><td>${x.xg}</td><td>${x.pases}</td><td class="muted">${x.torneo}</td></tr>`});h+='</table>';}
 else h+='<div class="muted">Sin partidos internacionales en la base (torneos 2022-2024).</div>';
 out.innerHTML=h+'</div>';}
</script></body></html>
"""


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    data = {
        "generado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "equipos": {**equipos_clubes(), **equipos_selecciones()},
        "jugadores": jugadores(),
    }
    # allow_nan=False: si un NaN se cuela, mejor explotar aca que romper JSON.parse en el browser.
    (DOCS / "data.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False),
                                    encoding="utf-8")
    (DOCS / "index.html").write_text(HTML, encoding="utf-8")
    kb = (DOCS / "data.json").stat().st_size / 1024
    print(f"[web_stats] docs/data.json: {len(data['equipos'])} equipos, "
          f"{len(data['jugadores'])} jugadores ({kb:,.0f} KB)")
    print(f"[web_stats] docs/index.html generado.")
    print("Ver local:  python -m http.server 8123 --directory docs   ->  http://localhost:8123")


if __name__ == "__main__":
    main()
