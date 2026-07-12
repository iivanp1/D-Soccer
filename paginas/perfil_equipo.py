"""Pagina PERFIL DE EQUIPO: KPIs de temporada + lineas de apuestas con rachas visuales.

El core es el renderizado de badges HTML (via st.markdown unsafe_allow_html): por cada
linea clasica (+4.5 corners, +3.5 tiros a puerta) se pintan los ultimos 5 partidos como
bloques redondeados -- verde tenue si el valor SUPERO la linea, rojo tenue si no --
con el formato estricto  [Valor] (vs [RIV]).  La std al lado del nombre mide volatilidad
(std baja = linea mas confiable para apostar).

Corre dentro del router app.py (st.navigation).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config                                        # noqa: E402
from src.estadisticas_detalladas import (                     # noqa: E402
    calcular_consistencia, obtener_racha_equipo, obtener_valor_portero)

DB = config.DATA_PROC / "dsoccer_clubes.db"

# Lineas clasicas a evaluar: (titulo, campo en la racha, linea)
LINEAS = [
    ("Córners", "corners", 4.5),
    ("Tiros a puerta", "tiros_puerta", 3.5),
]

# --- Estilos de los badges (spec estricto del usuario) --- #
BADGE_OK = ("background-color:#064e3b;color:#6ee7b7;border-radius:4px;padding:4px 8px;"
            "margin-right:4px;display:inline-block;font-weight:600;font-size:.85rem;")
BADGE_NO = ("background-color:#7f1d1d;color:#fca5a5;border-radius:4px;padding:4px 8px;"
            "margin-right:4px;display:inline-block;font-weight:600;font-size:.85rem;")
BADGE_SD = ("background-color:#1f2937;color:#6b7280;border-radius:4px;padding:4px 8px;"
            "margin-right:4px;display:inline-block;font-weight:600;font-size:.85rem;")
TARJETA = ("background-color:#131c2b;border-radius:12px;padding:16px 20px;"
           "margin-bottom:14px;")


@st.cache_data(ttl=3600)
def cargar_equipos() -> pd.DataFrame:
    """Todos los equipos de las 4 ligas: id, nombre, liga (de partidos_xg)."""
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT DISTINCT id_equipo_local AS id, nombre_local AS nombre, liga FROM partidos_xg "
        "UNION SELECT DISTINCT id_equipo_visitante, nombre_visitante, liga FROM partidos_xg", con)
    con.close()
    return df.sort_values("nombre").reset_index(drop=True)


@st.cache_data(ttl=3600)
def kpis_equipo(equipo_id: int) -> dict | None:
    """Fila de equipos_temporada (temporada mas reciente) para los KPIs."""
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    r = con.execute(
        "SELECT * FROM equipos_temporada WHERE id_equipo = ? "
        "ORDER BY temporada DESC LIMIT 1", (equipo_id,)).fetchone()
    con.close()
    return dict(r) if r else None


def _abrev(nombre: str) -> str:
    """Abreviatura de 3 letras del rival: 'Real Madrid' -> RMA, 'Barcelona' -> BAR."""
    palabras = [p for p in nombre.replace(".", " ").split() if len(p) > 1]
    if len(palabras) >= 2:
        return (palabras[0][0] + palabras[-1][:2]).upper()
    return nombre[:3].upper()


def _badges_linea(racha: list[dict], campo: str, linea: float) -> tuple[str, int, int]:
    """HTML de los badges de una linea + (aciertos, evaluables). Mas reciente primero."""
    html, ok, n = [], 0, 0
    for p in racha:
        riv = _abrev(p["rival"])
        v = p.get(campo)
        if v is None:  # partido sin cruce football-data -> badge gris (caso manejado)
            html.append(f'<span style="{BADGE_SD}">— (vs {riv})</span>')
            continue
        n += 1
        supero = v > linea
        ok += supero
        estilo = BADGE_OK if supero else BADGE_NO
        html.append(f'<span style="{estilo}">{int(v)} (vs {riv})</span>')
    return "".join(html), ok, n


def _std_txt(cons: dict, campo: str) -> str:
    """'σ L 1.1 · V 2.6' desde calcular_consistencia (None -> s/d)."""
    partes = []
    for cond, tag in (("local", "L"), ("visitante", "V")):
        c = cons.get(cond) or {}
        m = c.get(campo)
        partes.append(f"{tag} {m['std']:.1f}" if m and m.get("std") is not None else f"{tag} s/d")
    return "σ " + " · ".join(partes)


# ----------------------------------------------------------------------------- #
#  1. Selector de equipo
# ----------------------------------------------------------------------------- #
equipos = cargar_equipos()
LIGA_CORTA = {"ESP-La Liga": "La Liga", "ENG-Premier League": "Premier",
              "GER-Bundesliga": "Bundesliga", "ITA-Serie A": "Serie A"}
opciones = [f"{r.nombre} · {LIGA_CORTA.get(r.liga, r.liga)}" for r in equipos.itertuples()]
idx_default = next((i for i, o in enumerate(opciones) if o.startswith("Barcelona ")), 0)
sel = st.selectbox("Equipo", opciones, index=idx_default, label_visibility="collapsed")
fila = equipos.iloc[opciones.index(sel)]
equipo_id, nombre = int(fila["id"]), fila["nombre"]

st.markdown(f"# {nombre}")

# ----------------------------------------------------------------------------- #
#  2. KPIs de temporada
# ----------------------------------------------------------------------------- #
k = kpis_equipo(equipo_id)
gk = obtener_valor_portero(equipo_id)

if k is None:
    st.warning("Este equipo no tiene temporada cargada en la base todavía.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Puntos", int(k["puntos"]), delta=f"{k['dif_pts']:+.1f} vs xPts", delta_color="off")
c2.metric("xG a favor", f"{k['xg_favor']:.1f}", delta=f"{k['dif_goles']:+.1f} goles vs xG",
          delta_color="off")
c3.metric("xG en contra", f"{k['xg_contra']:.1f}")
c4.metric("Goles evitados (portero)",
          f"{gk['goles_evitados']:+.1f}" if gk else "s/d",
          delta=gk["portero"] if gk else None, delta_color="off")
st.caption(f"Temporada {k['temporada']} · {k['pj']} partidos · GF {int(k['gf'])} — GC {int(k['gc'])}"
           + (f" · 🧤 {gk['lectura']}" if gk else ""))

# ----------------------------------------------------------------------------- #
#  3. Lineas de apuestas: badges de los ultimos 5 (el core)
# ----------------------------------------------------------------------------- #
st.markdown("## Líneas de apuestas — últimos 5 partidos")
racha = obtener_racha_equipo(equipo_id, limite=5)
cons = calcular_consistencia(equipo_id)

if not racha:
    st.info("Sin racha disponible para este equipo.")
    st.stop()

for titulo, campo, linea in LINEAS:
    badges, ok, n = _badges_linea(racha, campo, linea)
    hit = f"{ok}/{n}" if n else "s/d"
    st.markdown(f"""
<div style="{TARJETA}">
  <div style="margin-bottom:10px;">
    <span style="font-weight:800;font-size:1.05rem;">+{linea} {titulo}</span>
    <span style="color:#6b7280;font-size:.8rem;margin-left:10px;">{_std_txt(cons, campo)}</span>
    <span style="color:#38bdf8;font-size:.8rem;margin-left:10px;font-weight:600;">supera {hit}</span>
  </div>
  {badges}
</div>
""", unsafe_allow_html=True)

st.caption("Más reciente primero. Verde = superó la línea, rojo = no. σ por condición "
           "(L local / V visitante) sobre los últimos 10: menor σ = línea más predecible. "
           "Gris = sin cruce de datos para ese partido.")
