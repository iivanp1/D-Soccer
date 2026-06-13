"""Motor Mundialista: fuerza de una seleccion agregando a sus jugadores (bottom-up).

La idea (la vision original del proyecto): una seleccion no se mide por los pocos
duelos entre selecciones, sino por la FORMA ACTUAL de sus jugadores en sus clubes.

Pipeline:
  1. Rating individual de cada jugador (perfil ofensivo -> alpha; defensivo -> beta),
     ajustado por la calidad de su liga y suavizado por minutos jugados.
  2. Agregacion ponderada CONVEXA: las estrellas pesan mas que los suplentes
     (un promedio simple diluiria a un Messi entre 10 jugadores del monton).
  3. Mapeo a goles esperados -> distribucion de Poisson -> probabilidades del partido.

ADAPTACIONES HONESTAS al dato real (FBref da stats por TEMPORADA, no por partido):
  - El decaimiento temporal xi se aplica por RECENCIA DE TEMPORADA, no por dia.
  - Shrinkage por minutos: ratings de jugadores con pocos minutos se suavizan hacia
    el promedio de su posicion (un delantero con 100' y 2 goles no es 1.8 goles/90).

LIMITACIONES (leer antes de creerle a los numeros):
  - SIN VALIDAR: los partidos de selecciones son escasos y no tenemos sus alineaciones
    historicas, asi que este modelo NO paso por backtest como los demas. Es un PRIOR
    razonable, no una verdad calibrada.
  - El perfil DEFENSIVo es el mas debil: intercepciones/tackles miden actividad, no
    necesariamente calidad defensiva.
  - El mapeo rating -> goles NO esta calibrado contra resultados reales. Da rankings
    relativos sensatos (Argentina > un equipo chico) pero las probabilidades absolutas
    son orientativas.
  - Solo cubrimos las 5 grandes ligas: jugadores en MLS, Saudi, ligas locales, etc.
    no aparecen (impacta mas a selecciones con muchos jugadores fuera del Big 5).

Uso:
    from src.jugadores_model import JugadoresModel
    m = JugadoresModel().entrenar_jugadores(df_jugadores)
    xi_arg = m.seleccion_probable("ARG")
    xi_fra = m.seleccion_probable("FRA")
    m.predecir_partido_mundial(xi_arg, xi_fra)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson
from unidecode import unidecode

from src import config

# --- Pesos del rating ofensivo (upgrade tipo-xG: 0.7 produccion de gol + 0.3 asistencias) ---
# NOTA: el xG puro NO esta disponible en FBref via soccerdata (verificado: no expone las
# columnas Expected). Usamos goles SIN PENAL por 90 como "xG realizado". El shrinkage
# bayesiano por minutos (K90) estabiliza las rachas cortas, que era el objetivo del xG.
W_NPG = 0.70      # peso de goles-sin-penal por 90
W_AST_OF = 0.30   # peso de asistencias por 90

# --- Shrinkage por minutos (en "90s equivalentes") ---
K90 = 8.0
# Shrinkage de disciplina (faltas/amarillas/rojas): m = 900 min = 10 partidos completos,
# que en unidades de "90s" es 10. Un jugador con pocos 90s se tracciona hacia el prior
# de su posicion, matando el ruido de rachas cortas.
K_DISC_90 = 10.0

# --- Agregacion convexa: decaimiento geometrico por ranking ---
# weight_k ~ DECAY^k. Con DECAY=0.8 el mejor jugador pesa 1, el 2do 0.8, el 3ro 0.64...
# Es una curva CONVEXA: da mas peso a las estrellas sin anular del todo al resto.
DECAY_RANK = 0.80

# --- Mapeo a goles (NO calibrado, ver limitaciones) ---
BASE_GOLES = 1.35   # goles promedio por equipo en un partido de selecciones
DAMP = 0.70         # atenua la relacion fuerza->goles (rendimientos decrecientes)

# --- Compresion dinamica por brecha de calidad (corrige la sub-diferenciacion) ---
# La compresion 'c' deja de ser fija: en partidos parejos vale c_base (mantiene
# realista un ARG-FRA), pero en partidos asimetricos crece de forma NO-LINEAL,
# estirando la ventaja del favorito hasta su verdadera disparidad. Tuneado a la
# realidad de las casas (grande vs chico ~ 78-82%), no a datos (es un prior).
GAP_AMP = 4.0   # cuanto amplifica la brecha de calidad a la compresion
GAP_POW = 2.0   # no-linealidad: brechas grandes amplifican mucho mas que las chicas
C_MAX = 1.7     # techo de seguridad de la compresion efectiva

# Mapeo Elo->goles AJUSTADO con datos reales (11k internacionales >=2015, ver elo_history.py):
# +400 puntos de Elo = +2.09 goles de ventaja. Reemplaza el multiplicador 2E inicial, que
# sobre-diferenciaba (~3.2 goles/400).
ELO_GOLES_POR_400 = 2.09


def _norm(nombre: str) -> str:
    """Normaliza un nombre para emparejar (sin acentos, minusculas)."""
    return unidecode(str(nombre)).strip().lower()


@dataclass
class JugadoresModel:
    xi: float = 0.0018

    jugadores: pd.DataFrame = field(default_factory=pd.DataFrame)  # indexado por nombre normalizado
    ref_of: float = 1.0
    ref_def: float = 1.0

    # --- Calibracion internacional (None = sin calibrar, usa el mapeo legacy) ---
    base_real: float | None = None   # media real de goles/equipo en el Mundial
    atk_ref: float = 1.0             # ataque agregado promedio de las selecciones
    def_ref: float = 1.0             # defensa agregada promedio de las selecciones
    compresion: float = 0.5          # cuanto se separan los equipos fuertes de la media

    # --- Hibrido con Elo de selecciones (columna vertebral; arregla sub-diferenciacion) ---
    elo: dict = field(default_factory=dict)   # {codigo: {overall, off, def}}
    # Peso del Elo en el hibrido: lam = w*lam_elo + (1-w)*lam_player. 0.65 PROVISIONAL
    # (recentra al mercado con el Elo ya calibrado); el valor final sale del tuneo con datos.
    w_elo: float = 0.65

    _entrenado: bool = False

    # ------------------------------------------------------------------ #
    def entrenar_jugadores(self, df: pd.DataFrame,
                           ajuste_xg: dict | None = None) -> "JugadoresModel":
        d = df[df["minutos"] > 0].copy()

        # Posicion primaria (FW de "FW,MF") y recencia de temporada
        d["pos1"] = d["posicion"].astype(str).str.split(",").str[0].str.split("-").str[0]
        seasons = sorted(d["season"].unique(), reverse=True)
        ago = {s: i for i, s in enumerate(seasons)}  # 0 = mas reciente
        d["w_season"] = np.exp(-self.xi * 365.0 * d["season"].map(ago))
        d["coef"] = d["league"].map(config.LEAGUE_QUALITY).fillna(config.QUALITY_DEFAULT)

        # Sumas ponderadas por recencia para agregar las (posibles) 2 temporadas del jugador
        contables = ["noventas", "goles_sin_pen", "asistencias", "tiros_arco",
                     "intercep", "tackles_ganados", "faltas_com", "amarillas", "rojas"]
        for c in contables:
            d[f"x_{c}"] = d["w_season"] * d[c].fillna(0)
        d["wn"] = d["w_season"] * d["noventas"]
        d["wn_coef"] = d["wn"] * d["coef"]

        agg = d.groupby("player").agg(
            n90=("x_noventas", "sum"),
            npg=("x_goles_sin_pen", "sum"),
            ast=("x_asistencias", "sum"),
            sot=("x_tiros_arco", "sum"),
            intc=("x_intercep", "sum"),
            tkl=("x_tackles_ganados", "sum"),
            fls=("x_faltas_com", "sum"),
            yel=("x_amarillas", "sum"),
            red=("x_rojas", "sum"),
            wn=("wn", "sum"),
            wn_coef=("wn_coef", "sum"),
        )
        agg = agg[agg["n90"] > 0]
        agg["coef_eff"] = agg["wn_coef"] / agg["wn"]

        # Identidad representativa: nacion/posicion de la temporada con mas minutos
        rep = (d.loc[d.groupby("player")["minutos"].idxmax()]
               [["player", "nacion", "pos1"]].set_index("player"))
        agg = agg.join(rep)

        # Tasas por 90 (ya ponderadas por recencia, porque num y den llevan w_season)
        # Perfil ofensivo (tipo-xG): 0.7*npg_90 + 0.3*ast_90, ajustado por calidad de liga.
        npg_90 = agg["npg"] / agg["n90"]   # goles sin penal por 90 (xG realizado)
        ast_90 = agg["ast"] / agg["n90"]   # asistencias por 90
        # Correccion por xG real (enriquecer_xg.py): regresa los goles realizados hacia el nivel
        # que sugiere el xG internacional. Castiga al suertudo, premia al generador. Sin ajuste
        # (None) -> factor 1.0 -> identico al comportamiento original (backward-compatible).
        if ajuste_xg:
            factores = pd.Series([ajuste_xg.get(_norm(p), 1.0) for p in agg.index], index=agg.index)
            npg_90 = npg_90 * factores
        agg["of_raw"] = (W_NPG * npg_90 + W_AST_OF * ast_90) * agg["coef_eff"]
        agg["def_raw"] = ((agg["intc"] + agg["tkl"]) / agg["n90"]) * agg["coef_eff"]

        # Disciplina por 90 con SHRINKAGE por minutos (empirical Bayes).
        # Prior por POSICION: tasa pooled de la posicion (robusta, no la distorsionan los
        # jugadores de pocos minutos). Un jugador se tracciona hacia el prior de su rol con
        # fuerza inversamente proporcional a sus 90s -> elimina outliers de rachas cortas.
        #   tasa_suavizada = (conteo_real + K * prior_posicion) / (n90 + K)
        pos_grp = agg.groupby("pos1")
        n90_pos = pos_grp["n90"].sum()
        for cnt, out in [("fls", "faltas_90"), ("yel", "amarillas_90"), ("red", "rojas_90")]:
            prior_pos = pos_grp[cnt].sum() / n90_pos            # tasa/90 pooled de la posicion
            prior = agg["pos1"].map(prior_pos).fillna(prior_pos.mean())
            agg[out] = (agg[cnt] + K_DISC_90 * prior) / (agg["n90"] + K_DISC_90)
        agg["tarjetas_90"] = agg["amarillas_90"] + agg["rojas_90"]

        # Shrinkage por minutos hacia el promedio de la POSICION (mejora clave)
        base_of = agg.groupby("pos1")["of_raw"].transform("mean")
        base_def = agg.groupby("pos1")["def_raw"].transform("mean")
        agg["ofensivo"] = (agg["n90"] * agg["of_raw"] + K90 * base_of) / (agg["n90"] + K90)
        agg["defensivo"] = (agg["n90"] * agg["def_raw"] + K90 * base_def) / (agg["n90"] + K90)

        # Normalizar a media 1.0 sobre jugadores "regulares" (>= 5 noventas equivalentes)
        reg = agg[agg["n90"] >= 5]
        self.ref_of = float(reg["ofensivo"].mean())
        self.ref_def = float(reg["defensivo"].mean())
        agg["ofensivo"] /= self.ref_of
        agg["defensivo"] /= self.ref_def

        agg["nombre"] = agg.index
        agg.index = [_norm(p) for p in agg.index]
        # Dos jugadores distintos pueden normalizar al mismo nombre; nos quedamos con
        # el de mas minutos para que el indice sea unico (evita choques al emparejar).
        agg = agg.sort_values("n90", ascending=False)
        agg = agg[~agg.index.duplicated(keep="first")]
        self.jugadores = agg[["nombre", "nacion", "pos1", "n90", "ofensivo", "defensivo",
                              "faltas_90", "amarillas_90", "rojas_90", "tarjetas_90"]]
        self._entrenado = True
        return self

    def disciplina_seleccion(self, lista_jugadores: list[str], nacion: str | None = None,
                             completar_a: int = 11) -> dict:
        """Faltas y tarjetas esperadas de una seleccion, sumando la disciplina del XI.

        Cada jugador aporta sus faltas/tarjetas por 90; el equipo es la suma sobre los 11.
        Los jugadores fuera del dataset se rellenan con disciplina sombra (config), igual
        que en construir_fuerza_seleccion, para que cualquier seleccion sume 11.
        """
        idx = [_norm(p) for p in lista_jugadores]
        sub = self.jugadores[self.jugadores.index.isin(idx)]
        faltas = list(sub["faltas_90"].to_numpy())
        tarj = list(sub["tarjetas_90"].to_numpy())
        n_real = len(faltas)

        if nacion is not None and n_real < completar_a:
            n = completar_a - n_real
            faltas += [config.SHADOW_FALTAS_90] * n
            tarj += [config.SHADOW_TARJETAS_90] * n
            return {"faltas": float(np.nansum(faltas)), "tarjetas": float(np.nansum(tarj))}

        if not faltas:
            return {"faltas": 11.0, "tarjetas": 2.0}  # respaldo
        # Sin nacion (legacy): escalamos a un XI completo segun los encontrados
        escala = completar_a / n_real
        return {"faltas": float(np.nansum(faltas)) * escala,
                "tarjetas": float(np.nansum(tarj)) * escala}

    # ------------------------------------------------------------------ #
    def seleccion_probable(self, nacion: str, formacion=(1, 4, 3, 3),
                           min_n90: float = 4.0) -> list[str]:
        """XI probable de una seleccion por CALIDAD dentro de cada posicion.

        Antes elegiamos por minutos, lo que dejaba afuera a estrellas con pocos minutos
        de club (ej. Messi en la MLS). Ahora elegimos a los mejores por posicion
        (arquero por minutos; defensas por rating defensivo; volantes por rating total;
        delanteros por rating ofensivo), exigiendo un piso de minutos para que sean
        titulares reales y no promesas con 200 minutos.
        """
        j = self.jugadores[self.jugadores["nacion"] == nacion].copy()
        if j.empty:
            return []
        regulares = j[j["n90"] >= min_n90]
        if len(regulares) >= sum(formacion):
            j = regulares
        j["calidad"] = j["ofensivo"] + j["defensivo"]

        n_gk, n_df, n_mf, n_fw = formacion
        elegidos = pd.concat([
            j[j["pos1"] == "GK"].nlargest(n_gk, "n90"),
            j[j["pos1"] == "DF"].nlargest(n_df, "defensivo"),
            j[j["pos1"] == "MF"].nlargest(n_mf, "calidad"),
            j[j["pos1"] == "FW"].nlargest(n_fw, "ofensivo"),
        ])
        # Completar hasta 11 con los mejores que queden (por si falta gente en una linea)
        faltan = sum(formacion) - len(elegidos)
        if faltan > 0:
            resto = j[~j.index.isin(elegidos.index)].nlargest(faltan, "calidad")
            elegidos = pd.concat([elegidos, resto])
        return list(elegidos["nombre"])

    @staticmethod
    def _shadow_rating(nacion: str | None) -> tuple[float, float]:
        """Rating sombra (ofensivo, defensivo) en cascada: pais -> confederacion -> default."""
        if nacion and nacion in config.NACION_SHADOW:
            return config.NACION_SHADOW[nacion]
        confed = config.NACION_CONFED.get(nacion)
        return config.CONFED_BASELINE.get(confed, config.CONFED_DEFAULT)

    @staticmethod
    def _convexo(valores: list[float]) -> float:
        """Agregacion convexa: ordena de mayor a menor y pondera con DECAY_RANK^k."""
        v = np.sort(np.array(valores, dtype=float))[::-1]
        w = DECAY_RANK ** np.arange(len(v))
        return float(np.sum(w * v) / np.sum(w))

    def construir_fuerza_seleccion(self, lista_jugadores: list[str], nacion: str | None = None,
                                   completar_a: int = 11) -> dict:
        """Agrega los ratings de una alineacion con pesos convexos (estrellas pesan mas).

        Imputacion por niveles: los jugadores que NO estan en el dataset (ligas locales
        no scrapeadas) se rellenan con el Rating Sombra de su federacion, hasta completar
        11. Asi el motor corre cualquier cruce del planeta, y la curva convexa deja que
        las pocas estrellas en Europa (ej. Almiron, Mitoma) sigan arrastrando el Alpha
        hacia arriba mientras los locales hacen de colchon de fondo.
        """
        idx = [_norm(p) for p in lista_jugadores]
        sub = self.jugadores[self.jugadores.index.isin(idx)]
        of_vals = list(sub["ofensivo"].to_numpy())
        def_vals = list(sub["defensivo"].to_numpy())
        n_real = len(of_vals)

        n_sombra = 0
        if nacion is not None and n_real < completar_a:
            of_b, def_b = self._shadow_rating(nacion)
            n_sombra = completar_a - n_real
            of_vals += [of_b] * n_sombra
            def_vals += [def_b] * n_sombra

        if not of_vals:
            raise ValueError("No hay jugadores en el dataset ni nacion para imputar sombra.")

        return {
            "ataque": self._convexo(of_vals),
            "defensa": self._convexo(def_vals),
            "reales": n_real,
            "sombra": n_sombra,
            "jugadores_pedidos": len(lista_jugadores),
        }

    def predecir_partido_mundial(self, lineup_local: list[str], lineup_visitante: list[str],
                                 nacion_local: str | None = None,
                                 nacion_visitante: str | None = None,
                                 max_goles: int = 8) -> dict:
        if not self._entrenado:
            raise RuntimeError("Llama a .entrenar_jugadores(df) primero.")
        fl = self.construir_fuerza_seleccion(lineup_local, nacion_local)
        fv = self.construir_fuerza_seleccion(lineup_visitante, nacion_visitante)

        if self.base_real is not None:
            # --- 1) Lambda del modelo de JUGADORES (forma actual + alineacion real) ---
            # Centrado en el promedio de selecciones + compresion dinamica por brecha.
            fuerza_l = fl["ataque"] + fl["defensa"]
            fuerza_v = fv["ataque"] + fv["defensa"]
            gap = abs(fuerza_l - fuerza_v) / (self.atk_ref + self.def_ref)
            c = min(self.compresion + GAP_AMP * gap ** GAP_POW, C_MAX)
            lam_l_player = self.base_real * (fl["ataque"] / self.atk_ref) ** c * (self.def_ref / fv["defensa"]) ** c
            lam_v_player = self.base_real * (fv["ataque"] / self.atk_ref) ** c * (self.def_ref / fl["defensa"]) ** c

            # --- 2) HIBRIDO: lam = w*lam_elo + (1-w)*lam_player ---
            # El Elo de selecciones diferencia bien (aplasta la sub-diferenciacion); el
            # modelo de jugadores aporta el ajuste por quien juega de verdad. Si no hay Elo
            # para alguna seleccion, cae limpio al modelo de jugadores puro.
            if self.elo and nacion_local in self.elo and nacion_visitante in self.elo:
                lam_l_elo, lam_v_elo = self._calcular_lambda_elo(
                    self.elo[nacion_local]["overall"], self.elo[nacion_visitante]["overall"], self.base_real)
                w = self.w_elo
                lam_l = w * lam_l_elo + (1 - w) * lam_l_player
                lam_v = w * lam_v_elo + (1 - w) * lam_v_player
                hibrido = {"w": w, "elo": (lam_l_elo, lam_v_elo), "player": (lam_l_player, lam_v_player)}
            else:
                lam_l, lam_v = lam_l_player, lam_v_player
                hibrido = {"w": 0.0, "elo": None, "player": (lam_l_player, lam_v_player)}
        else:
            # Mapeo legacy (sin calibrar): infla los goles con XIs de alto rating.
            lam_l = BASE_GOLES * (fl["ataque"] / fv["defensa"]) ** DAMP
            lam_v = BASE_GOLES * (fv["ataque"] / fl["defensa"]) ** DAMP
            hibrido = {"w": 0.0, "elo": None, "player": (lam_l, lam_v)}

        g = np.arange(max_goles + 1)
        m = np.outer(poisson.pmf(g, lam_l), poisson.pmf(g, lam_v))
        m /= m.sum()

        return {
            "fuerza_local": fl, "fuerza_visitante": fv, "hibrido": hibrido,
            "goles_esp_local": float(lam_l), "goles_esp_visitante": float(lam_v),
            "prob_local": float(np.tril(m, -1).sum()),
            "prob_empate": float(np.trace(m)),
            "prob_visitante": float(np.triu(m, 1).sum()),
            "marcador_probable": tuple(int(x) for x in np.unravel_index(m.argmax(), m.shape)),
            "prob_over_2_5": float(sum(m[a, b] for a in g for b in g if a + b > 2.5)),
        }

    # ------------------------------------------------------------------ #
    #  Calibracion internacional
    # ------------------------------------------------------------------ #
    def referencia_internacional(self, min_jugadores: int = 11) -> tuple[float, float, int]:
        """Ataque y defensa agregados PROMEDIO de las selecciones (la 'seleccion media').

        Recorre todas las naciones con al menos 11 jugadores en el dataset, arma su XI
        probable, y promedia sus fuerzas. Ese promedio es el ancla para centrar la escala.
        """
        nacs = self.jugadores["nacion"].value_counts()
        nacs = nacs[nacs >= min_jugadores].index
        atks, defs = [], []
        for n in nacs:
            xi = self.seleccion_probable(n)
            if len(xi) >= 11:
                f = self.construir_fuerza_seleccion(xi)
                atks.append(f["ataque"])
                defs.append(f["defensa"])
        return float(np.mean(atks)), float(np.mean(defs)), len(atks)

    def calibrar(self, media_goles_real: float, compresion: float = 0.5,
                 min_jugadores: int = 11) -> dict:
        """Fija el nivel (media real del Mundial) y el centrado de escala (selecciones)."""
        self.atk_ref, self.def_ref, n = self.referencia_internacional(min_jugadores)
        self.base_real = float(media_goles_real)
        self.compresion = float(compresion)
        return {"base_real": self.base_real, "atk_ref": self.atk_ref,
                "def_ref": self.def_ref, "compresion": self.compresion, "n_equipos_ref": n}

    def aplicar_calibracion(self, params: dict) -> "JugadoresModel":
        """Carga parametros de calibracion ya calculados (ej. desde calibracion.json)."""
        self.base_real = params["base_real"]
        self.atk_ref = params["atk_ref"]
        self.def_ref = params["def_ref"]
        self.compresion = params["compresion"]
        return self

    def cargar_elo(self, elo: dict, w: float | None = None) -> "JugadoresModel":
        """Inyecta los ratings Elo de selecciones (columna vertebral del hibrido).

        Se recibe el dict ya cargado (lo carga mundial_engine via src.elo) para no acoplar
        este modulo con el scraper y evitar imports circulares.
        """
        self.elo = elo or {}
        if w is not None:
            self.w_elo = float(w)
        return self

    @staticmethod
    def _calcular_lambda_elo(elo_local: float, elo_visitante: float,
                             base_real: float) -> tuple[float, float]:
        """Goles esperados segun el Elo, con el mapeo CALIBRADO a datos reales.

        Ajustado con 11k internacionales (>=2015, elo_history.py): cada 400 puntos de Elo
        valen +2.09 goles de ventaja, con un total de ~2*base_real. Repartimos ese total
        segun la supremacia (lineal en Elo). Es mas fiel que el 2E inicial, que
        sobre-diferenciaba. Clamp a 0.15 para no dar goles negativos en goleadas extremas.
        Partido parejo (mismo Elo) -> base_real cada uno.
        """
        supremacia = (ELO_GOLES_POR_400 / 400.0) * (elo_local - elo_visitante)
        return max(0.15, base_real + supremacia / 2.0), max(0.15, base_real - supremacia / 2.0)

    def ranking_jugadores(self, perfil: str = "ofensivo", top: int = 15,
                          min_n90: float = 10) -> pd.DataFrame:
        col = "ofensivo" if perfil.startswith("of") else "defensivo"
        j = self.jugadores[self.jugadores["n90"] >= min_n90]
        return (j.nlargest(top, col)[["nombre", "nacion", "pos1", "n90", col]]
                .reset_index(drop=True))


if __name__ == "__main__":
    import sys
    from tabulate import tabulate

    df = pd.read_csv(config.DATA_PROC / "jugadores.csv")
    m = JugadoresModel().entrenar_jugadores(df)

    print("Top 10 ofensivos (rating normalizado, 1.0 = jugador promedio):")
    print(tabulate(m.ranking_jugadores("ofensivo", 10), headers="keys",
                   floatfmt=".2f", showindex=False))

    loc = sys.argv[1] if len(sys.argv) > 1 else "ARG"
    vis = sys.argv[2] if len(sys.argv) > 2 else "FRA"
    xi_l, xi_v = m.seleccion_probable(loc), m.seleccion_probable(vis)
    print(f"\nXI probable {loc}: {len(xi_l)} jugadores  |  {vis}: {len(xi_v)} jugadores")

    p = m.predecir_partido_mundial(xi_l, xi_v)
    print(f"\n=== {loc} vs {vis} (Motor Mundialista, PRIOR sin validar) ===")
    print(f"  Fuerza {loc}: ataque {p['fuerza_local']['ataque']:.2f}  defensa {p['fuerza_local']['defensa']:.2f}")
    print(f"  Fuerza {vis}: ataque {p['fuerza_visitante']['ataque']:.2f}  defensa {p['fuerza_visitante']['defensa']:.2f}")
    print(f"  Goles esperados: {p['goles_esp_local']:.2f} - {p['goles_esp_visitante']:.2f}")
    print(f"  Gana {loc}: {p['prob_local']*100:.1f}%  |  Empate: {p['prob_empate']*100:.1f}%  |  "
          f"Gana {vis}: {p['prob_visitante']*100:.1f}%")
    print(f"  Marcador probable: {p['marcador_probable'][0]}-{p['marcador_probable'][1]}  |  "
          f"Over 2.5: {p['prob_over_2_5']*100:.1f}%")
