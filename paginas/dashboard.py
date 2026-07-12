"""Pagina PARTIDOS: Montecarlo por partido + scatter de valor xPts vs Puntos.

Corre dentro del router app.py (st.navigation). El set_page_config y el CSS global
viven en app.py; aca solo el contenido de la pagina.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config                                        # noqa: E402
from src.montecarlo import SimuladorMontecarlo, construir_parametros  # noqa: E402
from src.estadisticas_detalladas import (                     # noqa: E402
    obtener_racha_equipo, obtener_valor_portero)

DB = config.DATA_PROC / "dsoccer_clubes.db"
LIGAS = {"España": "ESP-La Liga", "Inglaterra": "ENG-Premier League",
         "Alemania": "GER-Bundesliga", "Italia": "ITA-Serie A"}
N_FORMA = 15      # partidos recientes por lado para las fuerzas xG
N_SIMS = 10_000


# ----------------------------------------------------------------------------- #
#  Datos (cacheados)
# ----------------------------------------------------------------------------- #
@st.cache_data(ttl=3600)
def cargar_partidos(liga: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM partidos_xg WHERE liga = ? ORDER BY fecha", con, params=(liga,))
    con.close()
    return df


@st.cache_data(ttl=3600)
def cargar_valor(liga: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM equipos_temporada WHERE liga = ? "
        "AND temporada = (SELECT MAX(temporada) FROM partidos_xg WHERE liga = ?)",
        con, params=(liga, liga))
    con.close()
    return df


def fuerzas_xg(df: pd.DataFrame, equipo_id: int, es_local: bool, n: int = N_FORMA):
    """(xg_favor_medio, xg_contra_medio) del equipo en esa CONDICION, ultimos n partidos."""
    if es_local:
        sub = df[df["id_equipo_local"] == equipo_id].tail(n)
        return sub["xg_local"].mean(), sub["xg_visitante"].mean()
    sub = df[df["id_equipo_visitante"] == equipo_id].tail(n)
    return sub["xg_visitante"].mean(), sub["xg_local"].mean()


def lambdas_desde_xg(df: pd.DataFrame, id_local: int, id_visit: int) -> tuple[float, float] | None:
    """lambda esperado de cada lado combinando ataque xG propio y defensa xG rival,
    normalizado por la media de la liga en esa condicion (metodo clasico de ratios)."""
    mu_l, mu_v = df["xg_local"].mean(), df["xg_visitante"].mean()
    atk_l, def_l = fuerzas_xg(df, id_local, es_local=True)
    atk_v, def_v = fuerzas_xg(df, id_visit, es_local=False)
    if any(pd.isna(x) for x in (atk_l, def_l, atk_v, def_v)):
        return None  # equipo sin historial suficiente (recien ascendido, etc.)
    lam_l = (atk_l / mu_l) * (def_v / mu_l) * mu_l   # ataque local x debilidad visitante
    lam_v = (atk_v / mu_v) * (def_l / mu_v) * mu_v
    return float(np.clip(lam_l, 0.15, 4.5)), float(np.clip(lam_v, 0.15, 4.5))


@st.cache_data(ttl=3600, show_spinner=False)
def simular(liga: str, id_local: int, id_visit: int) -> dict | None:
    df = cargar_partidos(liga)
    lams = lambdas_desde_xg(df, id_local, id_visit)
    if lams is None:
        return None
    params = construir_parametros(goles_local=lams[0], goles_visitante=lams[1])
    res = SimuladorMontecarlo().simular_partido(params, n_simulaciones=N_SIMS)
    res["lams"] = lams
    return res


# ----------------------------------------------------------------------------- #
#  Sidebar: liga + fecha / simulador
# ----------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚽ D-Soccer")
    liga_nombre = st.selectbox("Liga", list(LIGAS))
    liga = LIGAS[liga_nombre]
    df = cargar_partidos(liga)

    equipos = (pd.concat([df[["id_equipo_local", "nombre_local"]]
                          .rename(columns={"id_equipo_local": "id", "nombre_local": "nombre"}),
                          df[["id_equipo_visitante", "nombre_visitante"]]
                          .rename(columns={"id_equipo_visitante": "id", "nombre_visitante": "nombre"})])
               .drop_duplicates("id").sort_values("nombre"))
    id_de = dict(zip(equipos["nombre"], equipos["id"]))

    modo = st.radio("Modo", ["Partidos por fecha", "Simulador libre"],
                    label_visibility="collapsed")

    if modo == "Partidos por fecha":
        fechas = sorted(df["fecha"].unique(), reverse=True)
        fecha_sel = st.date_input("Fecha (jornada)", value=pd.Timestamp(fechas[0]),
                                  min_value=pd.Timestamp(fechas[-1]),
                                  max_value=pd.Timestamp(fechas[0]))
        # ventana +-3 dias = la jornada alrededor de la fecha elegida
        f0 = pd.Timestamp(fecha_sel)
        cerca = df[(pd.to_datetime(df["fecha"]) >= f0 - pd.Timedelta(days=3)) &
                   (pd.to_datetime(df["fecha"]) <= f0 + pd.Timedelta(days=3))]
        if cerca.empty:
            st.info("Sin partidos en esa ventana.")
            st.stop()
        etiquetas = [f"{r.fecha} · {r.nombre_local} vs {r.nombre_visitante}"
                     for r in cerca.itertuples()]
        sel = st.selectbox("Partido", etiquetas)
        r = cerca.iloc[etiquetas.index(sel)]
        id_local, id_visit = int(r["id_equipo_local"]), int(r["id_equipo_visitante"])
        nom_local, nom_visit = r["nombre_local"], r["nombre_visitante"]
        partido_jugado = r if pd.notna(r["goles_local"]) else None
    else:
        nom_local = st.selectbox("Local", list(id_de), index=0)
        nom_visit = st.selectbox("Visitante", list(id_de), index=1)
        id_local, id_visit = id_de[nom_local], id_de[nom_visit]
        partido_jugado = None

    st.caption(f"{len(df)} partidos con xG · {liga_nombre} · "
               "cuando arranque la 26/27 los nuevos partidos entran solos (cron semanal)")


# ----------------------------------------------------------------------------- #
#  Panel principal: Montecarlo del partido
# ----------------------------------------------------------------------------- #
st.markdown(f"# {nom_local} — {nom_visit}")

if id_local == id_visit:
    st.warning("Elegí dos equipos distintos.")
    st.stop()

res = simular(liga, id_local, id_visit)
if res is None:
    st.warning("Alguno de los equipos no tiene historial xG suficiente en la base "
               "(recién ascendido). Se completará cuando junte partidos.")
    st.stop()

lam_l, lam_v = res["lams"]
c1, c2, c3 = st.columns(3)
c1.metric(f"Victoria {nom_local}", f"{res['prob_local']*100:.1f}%")
c2.metric("Empate", f"{res['prob_empate']*100:.1f}%")
c3.metric(f"Victoria {nom_visit}", f"{res['prob_visitante']*100:.1f}%")

marcador, p_marc = res["marcadores_top"][0]
st.caption(
    f"Montecarlo {N_SIMS:,} sims · xG esperado {lam_l:.2f} — {lam_v:.2f} · "
    f"marcador más probable {marcador[0]}-{marcador[1]} ({p_marc*100:.0f}%) · "
    f"Over 2.5: {res['over_2_5_goles']*100:.0f}% · Ambos anotan: {res['btts']*100:.0f}% · "
    f"cuotas justas {1/max(res['prob_local'],1e-9):.2f} / {1/max(res['prob_empate'],1e-9):.2f} / "
    f"{1/max(res['prob_visitante'],1e-9):.2f}"
)
if partido_jugado is not None:
    st.caption(f"✓ Este partido ya se jugó: terminó "
               f"**{int(partido_jugado['goles_local'])}-{int(partido_jugado['goles_visitante'])}** "
               f"(xG real {partido_jugado['xg_local']:.2f} — {partido_jugado['xg_visitante']:.2f}).")

# --- Racha y portero (capa estadisticas_detalladas) --- #
with st.expander("Racha reciente y porteros"):
    ca, cb = st.columns(2)
    for col, eid, nombre in ((ca, id_local, nom_local), (cb, id_visit, nom_visit)):
        with col:
            st.markdown(f"**{nombre}**")
            racha = obtener_racha_equipo(eid, limite=5)
            if racha:
                t = pd.DataFrame(racha)[["fecha", "rival", "condicion", "goles",
                                         "goles_rival", "xG", "tiros_puerta", "corners"]]
                st.dataframe(t, hide_index=True, height=222)
            gk = obtener_valor_portero(eid)
            if gk:
                st.caption(f"🧤 {gk['portero']}: {gk['goles_evitados']:+.1f} goles evitados "
                           f"(save {gk['save_pct']:.0f}% vs liga {gk['save_pct_liga']:.0f}%) — "
                           f"{gk['lectura']}")


# ----------------------------------------------------------------------------- #
#  Grafico de valor: xPts vs Puntos reales
# ----------------------------------------------------------------------------- #
st.markdown("## Suerte vs mérito — xPts contra puntos reales")
val = cargar_valor(liga)

lo = float(min(val["xpts"].min(), val["puntos"].min())) - 3
hi = float(max(val["xpts"].max(), val["puntos"].max())) + 3
fig = go.Figure()
fig.add_trace(go.Scatter(  # diagonal de referencia: puntos = xPts
    x=[lo, hi], y=[lo, hi], mode="lines",
    line=dict(color="#3b4657", width=1, dash="dot"), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(
    x=val["xpts"], y=val["puntos"], mode="markers+text",
    text=val["equipo"], textposition="top center",
    textfont=dict(size=10, color="#8b95a7"),
    marker=dict(size=9, color=val["puntos"] - val["xpts"],
                colorscale=[[0, "#4ade80"], [0.5, "#8b95a7"], [1, "#f87171"]],
                line=dict(width=0)),
    customdata=np.stack([val["dif_pts"]], axis=-1),
    hovertemplate="<b>%{text}</b><br>xPts %{x:.1f} · Puntos %{y}<br>"
                  "dif %{customdata[0]:+.1f}<extra></extra>",
    showlegend=False))
fig.add_annotation(x=lo + 4, y=hi - 3, text="con suerte ↑", showarrow=False,
                   font=dict(color="#f87171", size=12))
fig.add_annotation(x=hi - 4, y=lo + 3, text="mala suerte ↓", showarrow=False,
                   font=dict(color="#4ade80", size=12))
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=560, margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(title="xPts (puntos esperados por xG)", gridcolor="#1b2331", zeroline=False),
    yaxis=dict(title="Puntos reales", gridcolor="#1b2331", zeroline=False),
    font=dict(family="Inter, sans-serif"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Verde (bajo la diagonal): sacó menos puntos de los que su xG merecía — candidatos a "
           "mejorar (buscar valor a favor). Rojo (arriba): sobre-rendimiento, cuidado al "
           "apostarles como favoritos. La regresión a la media no es garantía: es tendencia.")
