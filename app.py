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
  [data-testid="stSidebar"] { display: none; }
  .block-container { padding-top: 1rem; max-width: 1150px; }
  [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; }
  [data-testid="stMetricLabel"] { color: #8b95a7; letter-spacing: .06em;
                                   text-transform: uppercase; font-size: .75rem; }
  h1 { font-weight: 800; letter-spacing: -.02em; }
  .stCaption, .caption { color: #66707f; }

  /* --- Navbar (prototipo) --- */
  .navbar-brand {
    font-size: 1.3rem; font-weight: 800; letter-spacing: -.02em;
    color: #38bdf8; padding: .4rem 0 1rem 0;
  }
  div[data-testid="stPageLink"] {
    background: transparent; border-radius: 8px;
  }
  div[data-testid="stPageLink"] a {
    display: flex; align-items: center; width: 100%;
    padding: 8px 14px; border-radius: 8px;
    text-decoration: none;
  }
  div[data-testid="stPageLink"] a:hover {
    background: #1e293b;
  }
  div[data-testid="stPageLink"] p {
    font-weight: 600; font-size: .95rem;
  }
</style>
""", unsafe_allow_html=True)

paginas = [
    st.Page("paginas/dashboard.py", title="Partidos", icon="⚽", default=True),
    st.Page("paginas/perfil_equipo.py", title="Perfil de equipo", icon="📊"),
]

# --- Navbar horizontal (prototipo, a mejorar despues con frontend propio) --- #
nav_col_brand, nav_col1, nav_col2, nav_col_spacer = st.columns([2, 1, 1.4, 4])
with nav_col_brand:
    st.markdown('<div class="navbar-brand">⚽ D-Soccer</div>', unsafe_allow_html=True)
with nav_col1:
    st.page_link(paginas[0], label="Partidos", icon="⚽")
with nav_col2:
    st.page_link(paginas[1], label="Perfil de equipo", icon="📊")

st.divider()

pg = st.navigation(paginas, position="hidden")
pg.run()
