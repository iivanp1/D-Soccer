"""D-Soccer Web -- router principal (Streamlit multipagina via st.navigation).

Paginas:
  - paginas/dashboard.py      Partidos: Montecarlo por partido + scatter xPts vs Puntos.
  - paginas/perfil_equipo.py  Perfil de equipo: KPIs + lineas de apuestas con badges.

El set_page_config y el CSS global viven ACA (el entrypoint corre antes que la pagina
en cada rerun, asi el estilo aplica a todas). Las paginas no deben llamar set_page_config.

Correr:
    python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="D-Soccer", page_icon="⚽", layout="wide")

# --- Estetica global: minimalista, oscuro, sans-serif, sin ruido --- #
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  header { background: transparent !important; }
  header [data-testid="stToolbar"] { visibility: hidden; }
  .block-container { padding-top: 2rem; max-width: 1150px; }
  [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; }
  [data-testid="stMetricLabel"] { color: #8b95a7; letter-spacing: .06em;
                                   text-transform: uppercase; font-size: .75rem; }
  h1 { font-weight: 800; letter-spacing: -.02em; }
  .stCaption, .caption { color: #66707f; }
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("paginas/dashboard.py", title="Partidos", icon="⚽", default=True),
    st.Page("paginas/perfil_equipo.py", title="Perfil de equipo", icon="📊"),
])
pg.run()
