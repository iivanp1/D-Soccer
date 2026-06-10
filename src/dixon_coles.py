"""Modelo Dixon-Coles para predecir goles en futbol.

La idea (Dixon & Coles, 1997):
  - Cada equipo tiene una fuerza de ATAQUE y una de DEFENSA.
  - Jugar de local da una ventaja fija (gamma).
  - Los goles del local siguen un Poisson con media:
        lambda = exp(ataque_local - defensa_visitante + gamma)
    y los del visitante:
        mu     = exp(ataque_visitante - defensa_local)
  - Correccion 'rho' para marcadores bajos (0-0, 1-0, 0-1, 1-1), donde el
    Poisson puro se equivoca: en la realidad esos resultados estan correlacionados.
  - Decaimiento temporal: los partidos viejos pesan menos que los recientes
    (un equipo de hace 2 anios no dice tanto como el de hace 2 meses).

Entrenamiento por maxima verosimilitud con scipy.

Uso:
    from src.dixon_coles import DixonColes
    modelo = DixonColes().entrenar(df)
    modelo.predecir("Real Madrid", "Barcelona")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Correccion Dixon-Coles para la dependencia en marcadores bajos."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


@dataclass
class DixonColes:
    xi: float = 0.0018  # tasa de decaimiento temporal (por dia). 0 = sin decaimiento.
    max_goles: int = 10  # tope para las matrices de probabilidad de marcador

    equipos: list[str] = field(default_factory=list)
    ataque: dict[str, float] = field(default_factory=dict)
    defensa: dict[str, float] = field(default_factory=dict)
    gamma: float = 0.0  # ventaja de local
    rho: float = 0.0    # correccion marcadores bajos
    _entrenado: bool = False

    # ------------------------------------------------------------------ #
    def _pesos_tiempo(self, fechas: pd.Series) -> np.ndarray:
        """Peso exponencial: exp(-xi * dias_desde_el_partido)."""
        if self.xi <= 0:
            return np.ones(len(fechas))
        ref = fechas.max()
        dias = (ref - fechas).dt.days.to_numpy()
        return np.exp(-self.xi * dias)

    def _neg_log_verosimilitud(self, params, idx_l, idx_v, gl, gv, pesos, n) -> float:
        ataque = params[:n]
        defensa = params[n:2 * n]
        gamma, rho = params[2 * n], params[2 * n + 1]

        # Identificabilidad: centramos el ataque en 0 (sino ataque/defensa/gamma
        # se pueden desplazar libremente sin cambiar el modelo).
        ataque = ataque - ataque.mean()

        lam = np.exp(ataque[idx_l] - defensa[idx_v] + gamma)
        mu = np.exp(ataque[idx_v] - defensa[idx_l])

        t = _tau(gl, gv, lam, mu, rho)
        # log-verosimilitud por partido (ignorando factoriales, son constantes)
        ll = np.log(np.clip(t, 1e-10, None)) \
            + (-lam + gl * np.log(lam)) \
            + (-mu + gv * np.log(mu))
        return -np.sum(pesos * ll)

    # ------------------------------------------------------------------ #
    def entrenar(self, df: pd.DataFrame) -> "DixonColes":
        """df necesita columnas: fecha, local, visitante, goles_local, goles_visitante."""
        self.equipos = sorted(pd.concat([df["local"], df["visitante"]]).unique())
        n = len(self.equipos)
        idx = {e: i for i, e in enumerate(self.equipos)}

        idx_l = df["local"].map(idx).to_numpy()
        idx_v = df["visitante"].map(idx).to_numpy()
        gl = df["goles_local"].to_numpy()
        gv = df["goles_visitante"].to_numpy()
        pesos = self._pesos_tiempo(df["fecha"])

        # Parametros iniciales: ataque/defensa en 0, gamma=0.25 (~ventaja local tipica), rho=-0.1
        p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.1]])

        res = minimize(
            self._neg_log_verosimilitud,
            p0,
            args=(idx_l, idx_v, gl, gv, pesos, n),
            method="L-BFGS-B",
            options={"maxiter": 200, "disp": False},
        )

        ataque = res.x[:n] - res.x[:n].mean()
        defensa = res.x[n:2 * n]
        self.ataque = dict(zip(self.equipos, ataque))
        self.defensa = dict(zip(self.equipos, defensa))
        self.gamma = float(res.x[2 * n])
        self.rho = float(res.x[2 * n + 1])
        self._entrenado = True
        return self

    # ------------------------------------------------------------------ #
    def matriz_marcador(self, local: str, visitante: str) -> np.ndarray:
        """Matriz P[i,j] = prob de que termine i goles local, j goles visitante."""
        if not self._entrenado:
            raise RuntimeError("El modelo no esta entrenado. Llama a .entrenar(df) primero.")
        for eq in (local, visitante):
            if eq not in self.ataque:
                raise KeyError(f"Equipo desconocido: '{eq}'. No estaba en los datos de entrenamiento.")

        lam = np.exp(self.ataque[local] - self.defensa[visitante] + self.gamma)
        mu = np.exp(self.ataque[visitante] - self.defensa[local])

        g = np.arange(self.max_goles + 1)
        p_local = poisson.pmf(g, lam)
        p_visit = poisson.pmf(g, mu)
        matriz = np.outer(p_local, p_visit)

        # Aplicar correccion Dixon-Coles a las 4 celdas bajas
        matriz[0, 0] *= 1.0 - lam * mu * self.rho
        matriz[0, 1] *= 1.0 + lam * self.rho
        matriz[1, 0] *= 1.0 + mu * self.rho
        matriz[1, 1] *= 1.0 - self.rho

        return matriz / matriz.sum()  # renormalizar

    def predecir(self, local: str, visitante: str) -> dict:
        """Devuelve probabilidades 1X2, marcador mas probable y goles esperados."""
        m = self.matriz_marcador(local, visitante)
        p_local = float(np.tril(m, -1).sum())   # local mete mas
        p_empate = float(np.trace(m))
        p_visit = float(np.triu(m, 1).sum())

        i, j = np.unravel_index(m.argmax(), m.shape)
        g = np.arange(self.max_goles + 1)
        return {
            "local": local,
            "visitante": visitante,
            "prob_local": p_local,
            "prob_empate": p_empate,
            "prob_visitante": p_visit,
            "marcador_probable": (int(i), int(j)),
            "goles_esp_local": float((m.sum(axis=1) * g).sum()),
            "goles_esp_visitante": float((m.sum(axis=0) * g).sum()),
            "prob_over_2_5": float(sum(m[a, b] for a in g for b in g if a + b > 2.5)),
        }

    def ranking(self, top: int = 10) -> pd.DataFrame:
        """Ranking de equipos por fuerza neta.

        Ojo con el signo: en este modelo lambda = exp(ataque_local - defensa_visita),
        asi que un valor de 'defensa' ALTO significa que el rival mete menos = buena
        defensa. Por eso la fuerza neta es ataque + defensa (los dos altos = equipo top).
        """
        filas = [
            {"equipo": e, "ataque": self.ataque[e], "defensa": self.defensa[e],
             "fuerza": self.ataque[e] + self.defensa[e]}
            for e in self.equipos
        ]
        return (pd.DataFrame(filas)
                .sort_values("fuerza", ascending=False)
                .head(top)
                .reset_index(drop=True))
