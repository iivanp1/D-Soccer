"""Modelo de metricas secundarias: tiros, corners, faltas y tarjetas.

A diferencia de los goles (que usan Dixon-Coles por su correlacion rara en
marcadores bajos), estas variables son conteos "limpios" y se modelan con Poisson
estandar via tasas relativas al promedio de liga.

Estructura:
  - VOLUMEN DE JUEGO (tiros, corners): cada equipo tiene un factor de generacion
    (cuanto produce vs la media) y uno de concesion (cuanto permite al rival).
        esperado_local = media_liga * generacion[local] * concesion[visitante]
  - INTENSIDAD (faltas, tarjetas):
        faltas: factor de cometer del equipo A x factor de provocar del equipo B.
        tarjetas: faltas_esperadas x (tarjetas-por-falta de la liga, por localia)
                  x factor_arbitro.

EL INDICE DEL ARBITRO (la parte original)
-----------------------------------------
Para cada arbitro calculamos su ratio tarjetas-por-falta y lo comparamos con el
promedio de la liga -> factor centrado en 1.0 (1.0 = arbitro promedio; 1.3 = saca
30% mas tarjetas por falta; 0.8 = mas permisivo).

Problema: un arbitro con pocos partidos tiene un ratio ruidoso (4 partidos con
muchas tarjetas no lo hacen "estricto"). Solucion: SHRINKAGE (regresion a la media).
Mezclamos su ratio con el promedio de liga, pesando por cuantos partidos dirigio:

    factor = (n_partidos * factor_crudo + K * 1.0) / (n_partidos + K)

Con pocos partidos el factor tira hacia 1.0 (prudente); con muchos, confiamos en el
arbitro. K controla cuanta evidencia exigimos antes de creerle (por defecto 10).
Un arbitro nuevo (0 partidos) cae automaticamente en factor = 1.0 (promedio de liga).

Uso:
    from src.estilos_model import EstilosModel
    modelo = EstilosModel().entrenar(df)
    modelo.predecir_metricas("Real Madrid", "Barcelona", "A Marciniak")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _pesos_tiempo(fechas: pd.Series, xi: float) -> np.ndarray:
    """Peso exponencial por antiguedad: exp(-xi * dias)."""
    if xi <= 0:
        return np.ones(len(fechas))
    ref = fechas.max()
    dias = (ref - fechas).dt.days.to_numpy()
    return np.exp(-xi * dias)


def _wmean(valores: pd.Series, pesos: np.ndarray) -> float:
    """Media ponderada que ignora NaN (datos faltantes en ligas/temporadas viejas)."""
    v = valores.to_numpy(dtype=float)
    mask = ~np.isnan(v)
    if not mask.any():
        return np.nan
    return float(np.sum(pesos[mask] * v[mask]) / np.sum(pesos[mask]))


def _wmean_por_grupo(df: pd.DataFrame, grupo: str, col: str) -> dict[str, float]:
    """Media ponderada de `col` por cada valor de `grupo`, ignorando NaN."""
    sub = pd.DataFrame({
        "g": df[grupo].to_numpy(),
        "w": np.where(df[col].notna(), df["peso"], 0.0),
        "wx": (df["peso"] * df[col]).fillna(0.0).to_numpy(),
    })
    agg = sub.groupby("g").agg(w=("w", "sum"), wx=("wx", "sum"))
    return (agg["wx"] / agg["w"]).to_dict()


@dataclass
class EstilosModel:
    xi: float = 0.0018      # decaimiento temporal (igual que Dixon-Coles)
    k_arbitro: int = 10     # fuerza del shrinkage del arbitro, en "partidos equivalentes"

    # --- aprendido en entrenar() ---
    liga: dict = field(default_factory=dict)            # constantes globales (fallback)
    liga_cph: dict = field(default_factory=dict)        # tarjetas-por-falta por cada liga
    factores: dict = field(default_factory=dict)        # factores por equipo
    arbitros: dict = field(default_factory=dict)        # factor de rigurosidad por arbitro
    arbitros_n: dict = field(default_factory=dict)      # partidos dirigidos por arbitro
    _entrenado: bool = False

    # ------------------------------------------------------------------ #
    def _formato_largo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte cada partido en 2 filas (una por equipo) con sus metricas."""
        peso = _pesos_tiempo(df["fecha"], self.xi)

        local = pd.DataFrame({
            "equipo": df["local"].to_numpy(),
            "liga": df["liga"].to_numpy(),
            "es_local": True,
            "peso": peso,
            "tiros_favor": df["tiros_local"], "tiros_contra": df["tiros_visitante"],
            "corners_favor": df["corners_local"], "corners_contra": df["corners_visitante"],
            "faltas_com": df["faltas_local"], "faltas_rec": df["faltas_visitante"],
            "tarjetas": df["amarillas_local"].fillna(0) + df["rojas_local"].fillna(0),
        })
        visita = pd.DataFrame({
            "equipo": df["visitante"].to_numpy(),
            "liga": df["liga"].to_numpy(),
            "es_local": False,
            "peso": peso,
            "tiros_favor": df["tiros_visitante"], "tiros_contra": df["tiros_local"],
            "corners_favor": df["corners_visitante"], "corners_contra": df["corners_local"],
            "faltas_com": df["faltas_visitante"], "faltas_rec": df["faltas_local"],
            "tarjetas": df["amarillas_visitante"].fillna(0) + df["rojas_visitante"].fillna(0),
        })
        return pd.concat([local, visita], ignore_index=True)

    # ------------------------------------------------------------------ #
    def entrenar(self, df: pd.DataFrame) -> "EstilosModel":
        largo = self._formato_largo(df)

        # --- Promedios de liga (base de todos los factores) ---
        self.liga["tiros"] = _wmean(largo["tiros_favor"], largo["peso"].to_numpy())
        self.liga["corners"] = _wmean(largo["corners_favor"], largo["peso"].to_numpy())
        self.liga["faltas"] = _wmean(largo["faltas_com"], largo["peso"].to_numpy())

        # --- Factores por equipo (tasa del equipo / tasa de liga) ---
        def factor(col, base):
            medias = _wmean_por_grupo(largo, "equipo", col)
            return {e: v / base for e, v in medias.items()}

        self.factores = {
            "gen_tiros":  factor("tiros_favor", self.liga["tiros"]),
            "con_tiros":  factor("tiros_contra", self.liga["tiros"]),
            "gen_corners": factor("corners_favor", self.liga["corners"]),
            "con_corners": factor("corners_contra", self.liga["corners"]),
            "gen_faltas": factor("faltas_com", self.liga["faltas"]),     # propension a cometer
            "draw_faltas": factor("faltas_rec", self.liga["faltas"]),    # propension a provocar
        }

        # --- Tarjetas por falta (cards-per-foul), separado por localia ---
        # Mejora #3: el visitante recibe mas tarjetas; mantenemos el split local/visita.
        def cph(sub: pd.DataFrame) -> float:
            w = sub["peso"].to_numpy()
            cards = (w * sub["tarjetas"]).sum()
            fouls = (w * sub["faltas_com"].fillna(0)).sum()
            return cards / fouls if fouls > 0 else np.nan

        self.liga["cph_local"] = cph(largo[largo["es_local"]])
        self.liga["cph_visita"] = cph(largo[~largo["es_local"]])
        self.liga["cph_global"] = cph(largo)

        # Tarjetas-por-falta de CADA liga por separado (aisla el efecto liga del
        # efecto arbitro). La Premier es mas estricta que otras ligas; eso ahora lo
        # captura el baseline de la liga, no el factor del arbitro.
        self.liga_cph = {}
        for lg, sub in largo.groupby("liga"):
            self.liga_cph[lg] = {
                "local": cph(sub[sub["es_local"]]),
                "visita": cph(sub[~sub["es_local"]]),
                "global": cph(sub),
            }

        # --- Indice del arbitro: observado/esperado vs SU liga + shrinkage ---
        self._entrenar_arbitros(df)

        self._entrenado = True
        return self

    def _entrenar_arbitros(self, df: pd.DataFrame) -> None:
        """Factor del arbitro por el metodo OBSERVADO / ESPERADO.

        Para cada partido que dirigio, calculamos las tarjetas que esperaria un
        arbitro PROMEDIO DE ESA LIGA (faltas del partido x tarjetas-por-falta de la
        liga). El factor del arbitro es: tarjetas reales / tarjetas esperadas.
        Asi, un arbitro que pita en varias ligas se compara contra el promedio
        ponderado de SUS ligas automaticamente (cada partido trae su propio baseline).
        """
        d = df[df["arbitro"].notna() & (df["arbitro"].astype(str).str.strip() != "")].copy()
        if d.empty:
            return
        d["peso"] = _pesos_tiempo(d["fecha"], self.xi)
        d["cards"] = (d["amarillas_local"].fillna(0) + d["rojas_local"].fillna(0)
                      + d["amarillas_visitante"].fillna(0) + d["rojas_visitante"].fillna(0))
        d["fouls"] = d["faltas_local"].fillna(0) + d["faltas_visitante"].fillna(0)

        # Baseline de tarjetas-por-falta de la liga de CADA partido.
        base_global = self.liga["cph_global"]
        d["cph_liga"] = d["liga"].map(
            lambda lg: self.liga_cph.get(lg, {}).get("global", base_global))

        # Tarjetas esperadas si el arbitro fuera promedio de su liga.
        d["esperadas"] = d["fouls"] * d["cph_liga"]
        d["w_real"] = d["peso"] * d["cards"]
        d["w_esp"] = d["peso"] * d["esperadas"]

        agg = d.groupby("arbitro").agg(
            w_real=("w_real", "sum"),
            w_esp=("w_esp", "sum"),
            n=("arbitro", "size"),
        )
        for arb, fila in agg.iterrows():
            if fila["w_esp"] <= 0:
                continue
            # Observado/esperado: >1 = mas estricto que el promedio de SU liga.
            factor_crudo = fila["w_real"] / fila["w_esp"]
            n = fila["n"]
            # Shrinkage hacia 1.0 (= promedio de su liga) segun cantidad de partidos.
            factor = (n * factor_crudo + self.k_arbitro * 1.0) / (n + self.k_arbitro)
            self.arbitros[arb] = float(factor)
            self.arbitros_n[arb] = int(n)

    # ------------------------------------------------------------------ #
    def _f(self, nombre: str, equipo: str) -> float:
        """Factor de un equipo; 1.0 (promedio de liga) si es desconocido."""
        return self.factores.get(nombre, {}).get(equipo, 1.0)

    def predecir_metricas(self, local: str, visitante: str,
                          arbitro: str | None = None, liga: str | None = None) -> dict:
        if not self._entrenado:
            raise RuntimeError("El modelo no esta entrenado. Llama a .entrenar(df) primero.")

        # --- Volumen de juego ---
        tiros_local = self.liga["tiros"] * self._f("gen_tiros", local) * self._f("con_tiros", visitante)
        tiros_vis = self.liga["tiros"] * self._f("gen_tiros", visitante) * self._f("con_tiros", local)
        corners_local = self.liga["corners"] * self._f("gen_corners", local) * self._f("con_corners", visitante)
        corners_vis = self.liga["corners"] * self._f("gen_corners", visitante) * self._f("con_corners", local)

        # --- Faltas (cometer x provocar) ---
        faltas_local = self.liga["faltas"] * self._f("gen_faltas", local) * self._f("draw_faltas", visitante)
        faltas_vis = self.liga["faltas"] * self._f("gen_faltas", visitante) * self._f("draw_faltas", local)

        # --- Tarjetas: faltas x (tarjetas/falta de SU liga, por localia) x factor arbitro ---
        # El baseline lo pone la liga del partido; el factor del arbitro es su desviacion
        # relativa a esa liga. Arbitro o liga desconocidos -> 1.0 / promedio global.
        cph = self.liga_cph.get(liga) if liga else None
        cph_local = cph["local"] if cph else self.liga["cph_local"]
        cph_visita = cph["visita"] if cph else self.liga["cph_visita"]
        f_arb = self.arbitros.get(arbitro, 1.0) if arbitro else 1.0
        tarjetas_local = faltas_local * cph_local * f_arb
        tarjetas_vis = faltas_vis * cph_visita * f_arb

        return {
            "tiros_local": tiros_local,
            "tiros_vis": tiros_vis,
            "corners_local": corners_local,
            "corners_vis": corners_vis,
            "faltas_local": faltas_local,
            "faltas_vis": faltas_vis,
            "faltas_total": faltas_local + faltas_vis,
            "tarjetas_local": tarjetas_local,
            "tarjetas_vis": tarjetas_vis,
            "tarjetas_total": tarjetas_local + tarjetas_vis,
            "factor_arbitro": f_arb,
        }

    # ------------------------------------------------------------------ #
    def ranking_arbitros(self, top: int = 15, min_partidos: int = 15) -> pd.DataFrame:
        """Arbitros mas estrictos (factor > 1) y mas permisivos, con suficiente muestra."""
        filas = [
            {"arbitro": a, "partidos": self.arbitros_n[a], "factor": f}
            for a, f in self.arbitros.items()
            if self.arbitros_n[a] >= min_partidos
        ]
        return (pd.DataFrame(filas)
                .sort_values("factor", ascending=False)
                .head(top)
                .reset_index(drop=True))


if __name__ == "__main__":
    # Demo rapida
    import sys
    from src import config

    df = pd.read_csv(config.DATA_PROC / "partidos.csv", parse_dates=["fecha"])
    modelo = EstilosModel().entrenar(df)

    print(f"Promedios de liga -> tiros: {modelo.liga['tiros']:.1f}  "
          f"corners: {modelo.liga['corners']:.1f}  faltas: {modelo.liga['faltas']:.1f}")
    print(f"Tarjetas por falta -> local: {modelo.liga['cph_local']:.3f}  "
          f"visita: {modelo.liga['cph_visita']:.3f}  "
          f"(el visitante recibe {'mas' if modelo.liga['cph_visita'] > modelo.liga['cph_local'] else 'menos'})\n")

    print("Arbitros mas estrictos (tarjetas/falta vs promedio de liga):")
    from tabulate import tabulate
    print(tabulate(modelo.ranking_arbitros(10), headers="keys", floatfmt=".3f", showindex=False))

    local = sys.argv[1] if len(sys.argv) > 1 else modelo.factores["gen_tiros"] and "Real Madrid"
    visitante = sys.argv[2] if len(sys.argv) > 2 else "Barcelona"
    arbitro = sys.argv[3] if len(sys.argv) > 3 else None
    m = modelo.predecir_metricas(local, visitante, arbitro)
    print(f"\n=== {local} vs {visitante} (arbitro: {arbitro or 'promedio'}) ===")
    print(f"  Tiros:    {m['tiros_local']:.1f} - {m['tiros_vis']:.1f}")
    print(f"  Corners:  {m['corners_local']:.1f} - {m['corners_vis']:.1f}")
    print(f"  Faltas:   {m['faltas_total']:.1f} total")
    print(f"  Tarjetas: {m['tarjetas_total']:.1f} total  (factor arbitro: {m['factor_arbitro']:.2f})")
