"""Dixon-Coles entrenado sobre xG (Understat) en vez de goles reales.

POR QUE FUNCIONA SIN REESCRIBIR EL MODELO: la log-verosimilitud de DixonColes ya omite
los factoriales (son constantes respecto de los parametros), asi que el termino
    -lambda + k*log(lambda)
es una QUASI-verosimilitud Poisson valida tambien con k CONTINUO. Entrenar con
goles_local=xG_local es matematicamente legitimo: los ratings de ataque/defensa pasan a
medir la CREACION/CONCESION de peligro (menos ruido de finiquito que los goles).

EL UNICO AJUSTE REAL (rho): la correccion de marcadores bajos tau() compara k==0 y k==1
EXACTOS; con xG continuo (0.83, 1.47...) nunca activa, la verosimilitud queda plana en
rho y el optimizador lo deja en el valor inicial. Por eso el 2do paso: con los ratings
xG YA fijos, se re-estima rho por maxima verosimilitud sobre los MARCADORES ENTEROS
reales (1-D, minimize_scalar). Cada parametro se estima con el dato que lo identifica:
ratings <- xG, correlacion de marcadores bajos <- goles enteros.

Uso:
    from src.dixon_coles_xg import DixonColesXG
    m = DixonColesXG().entrenar_xg(df_xg, df_goles)   # df_xg con xG en goles_local/visitante
    m.predecir("Real Madrid", "Barcelona")            # misma interfaz que DixonColes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.dixon_coles import DixonColes, _tau


class DixonColesXG(DixonColes):
    """DixonColes con ratings entrenados sobre xG y rho re-estimado sobre goles reales."""

    def entrenar_xg(self, df_xg: pd.DataFrame, df_goles: pd.DataFrame | None = None) -> "DixonColesXG":
        """df_xg: fecha, local, visitante, goles_local=xG_local, goles_visitante=xG_visit.

        df_goles (opcional pero recomendado): mismos equipos con MARCADORES ENTEROS
        reales, para identificar rho. Sin el, rho queda en el default clasico -0.10.
        """
        self.entrenar(df_xg)          # quasi-Poisson sobre xG -> ratings + gamma
        if df_goles is not None and len(df_goles):
            self._reestimar_rho(df_goles)
        else:
            self.rho = -0.10          # prior clasico de clubes si no hay goles para estimar
        return self

    def _reestimar_rho(self, df: pd.DataFrame) -> None:
        """MV 1-D de rho sobre marcadores enteros reales, con los ratings xG fijos."""
        conocidos = df["local"].isin(self.ataque) & df["visitante"].isin(self.ataque)
        d = df[conocidos]
        if d.empty:
            self.rho = -0.10
            return
        atk = d["local"].map(self.ataque).to_numpy()
        dfn = d["visitante"].map(self.defensa).to_numpy()
        atk_v = d["visitante"].map(self.ataque).to_numpy()
        dfn_l = d["local"].map(self.defensa).to_numpy()
        lam = np.exp(atk - dfn + self.gamma)
        mu = np.exp(atk_v - dfn_l)
        gl = d["goles_local"].to_numpy()
        gv = d["goles_visitante"].to_numpy()
        pesos = self._pesos_tiempo(d["fecha"])

        def _neg_ll(rho: float) -> float:
            t = _tau(gl, gv, lam, mu, rho)
            return -float(np.sum(pesos * np.log(np.clip(t, 1e-10, None))))

        r = minimize_scalar(_neg_ll, bounds=(-0.35, 0.35), method="bounded")
        self.rho = float(r.x)
