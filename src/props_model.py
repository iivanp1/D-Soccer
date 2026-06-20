"""Modelo de Player Props: distribucion de tiros entre el XI titular.

Logica:
  1. λ_shots_team = λ_goles_team / CONVERSION_RATE_INTL
  2. usage_i = tiros_90_i / sum(tiros_90_j para j en XI)
  3. λ_shots_i = λ_shots_team × usage_i
  4. P(tiros_i > K) = Poisson(λ_shots_i).sf(K) = 1 - CDF(K)

Jerarquia de fuentes para tiras_90:
  1. Historial internacional (StatsBomb, tiros_intl.json, >= 3 partidos) -- mas relevante
  2. Datos de club (FBref, jugadores.csv) × ESCALA_TIROS_SELECCION -- proxy calibrado
  3. Shadow rate por posicion (config) -- ultimo recurso (no apostamos si es shadow)

Usa Poisson (v1). La diferencia con Binomial Negativa es ~2-3pp para λ 1.5-3.0,
menor que el error de estimacion de λ (30-50%): no vale la complejidad extra todavia.

Uso:
    from src.props_model import calcular_props_partido
    props = calcular_props_partido(info, jugadores_df, datos_tiros)
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd

from src import config
from src.jugadores_model import _norm

# Clamps de seguridad para evitar props absurdas por outliers de λ
LAMBDA_TEAM_MIN = 5.0    # ningun equipo genera menos de 5 tiros/partido
LAMBDA_TEAM_MAX = 22.0   # ningun equipo genera mas de 22 tiros/partido
USAGE_MIN = 0.01         # min 1% del volumen del equipo (GK, DF profundos)
USAGE_MAX = 0.35         # max 35% (nadie monopoliza mas de un tercio)
LAMBDA_JUGADOR_MIN_REPORTAR = 0.8  # debajo de esto: no alertar (poca senal)


# ------------------------------------------------------------------ #
def prob_over_k(lam: float, k: float) -> float:
    """P(X > k) para X ~ Poisson(lam). k puede ser .5, 1.5, 2.5 etc.

    Para k = 1.5: devuelve P(X >= 2) = 1 - P(X=0) - P(X=1).
    Para k = 0.5: devuelve P(X >= 1) = 1 - e^(-lam).
    Numericamente estable para lam en [0.01, 30].
    """
    if lam <= 0:
        return 0.0
    k_ceil = int(k) + 1  # minimo entero > k
    e_neg_lam = math.exp(-lam)
    cumul = 0.0
    lam_pow = 1.0
    fact = 1
    for j in range(k_ceil):
        cumul += e_neg_lam * lam_pow / fact
        if j < k_ceil - 1:
            lam_pow *= lam
            fact *= (j + 1)
    return max(0.0, min(1.0, 1.0 - cumul))


# ------------------------------------------------------------------ #
def _tasa_tiros_jugador(nombre_norm: str, pos1: str, datos_tiros: dict,
                        club_fallback: tuple[float, float] = (0.0, 0.0)
                        ) -> tuple[float, float, str, float]:
    """Retorna (tiros_90, sot_rate, fuente, xg_per_shot) para un jugador.

    Jerarquia: intl_confirmado > club_escalado(json) > club_fallback(FBref) > shadow.
    club_fallback (tiros_90, sot_rate) viene de FBref directo: rescata a jugadores que
    NO estan en tiros_intl.json por no haber jugado los torneos StatsBomb (Noruega,
    Uzbekistan, etc.), pero que SI tienen datos de club. xg_per_shot: fallback = conv_rate.
    """
    meta = datos_tiros.get("meta", {})
    escala = meta.get("escala_tiros_seleccion", 1.0)
    sot_default = meta.get("sot_rate_default", 0.333)
    conv_rate = meta.get("conversion_rate_xg", meta.get("conversion_rate_intl", 0.091))
    shadow = meta.get("shadow_tiros_90", {"FW": 2.46, "MF": 1.33, "DF": 0.5, "GK": 0.1})

    jugadores = datos_tiros.get("jugadores", {})
    if nombre_norm in jugadores:
        j = jugadores[nombre_norm]
        n = j.get("n_intl", 0)
        min_intl = meta.get("min_partidos_intl", 3)
        xg_per_shot = j.get("xg_per_shot_intl", conv_rate)
        if n >= min_intl and j["tiros_pp_intl"] > 0:
            return j["tiros_pp_intl"], j.get("sot_rate", sot_default), "intl", xg_per_shot
        if j.get("tiros_90_club", 0) > 0:
            # sin historial intl de calidad: usamos media global como prior
            return j["tiros_90_club"] * escala, j.get("sot_rate", sot_default), "club", conv_rate

    # Fallback: tasa de club desde FBref (jugador fuera de tiros_intl.json). Escalada
    # club->seleccion igual que el tier "club" del json. Esto hace que props funcione
    # para selecciones del Mundial que no jugaron los torneos StatsBomb 2018-2024.
    t90_club, sot_club = club_fallback
    if t90_club > 0:
        return t90_club * escala, sot_club if sot_club > 0 else sot_default, "club", conv_rate

    pos_key = pos1 if pos1 in shadow else "MF"
    return shadow.get(pos_key, 1.0), sot_default, "shadow", conv_rate


def _pos1(posicion: str) -> str:
    """Posicion primaria ('FW,MF' -> 'FW')."""
    return posicion.split(",")[0].split("-")[0] if posicion else "MF"


def _club_rate_fbref(sub: pd.DataFrame) -> tuple[float, float]:
    """(tiros_90, sot_rate) desde la fila FBref del jugador (temporada con mas minutos).
    Fallback para jugadores fuera de tiros_intl.json. (0,0) si no hay datos suficientes."""
    if sub.empty or sub["noventas"].max() < 3:
        return 0.0, 0.0
    row = sub.loc[sub["minutos"].idxmax()]
    n90, tiros, arco = float(row["noventas"]), float(row["tiros"]), float(row["tiros_arco"])
    if n90 <= 0 or tiros <= 0:
        return 0.0, 0.0
    return tiros / n90, arco / tiros


# ------------------------------------------------------------------ #
def calcular_props_equipo(xi_names: list[str], nacion: str,
                          lam_goles: float, jugadores_df: pd.DataFrame,
                          datos_tiros: dict) -> dict:
    """Props de tiros para un equipo.

    Devuelve {nombre_jugador: {"lam": ..., "p_over_0_5": ..., "p_over_1_5": ...,
                                "p_sot_1": ..., "fuente": ..., "pos": ...}}
    """
    meta = datos_tiros.get("meta", {})
    conv_rate = meta.get("conversion_rate_xg", meta.get("conversion_rate_intl", 0.091))

    lam_shots_team = max(LAMBDA_TEAM_MIN, min(LAMBDA_TEAM_MAX, lam_goles / conv_rate))

    # Construir tabla de tasas por jugador del XI
    filas = []
    for nombre in xi_names:
        nrm = _norm(nombre)
        sub = jugadores_df[jugadores_df["player"].apply(_norm) == nrm]
        pos = _pos1(sub.iloc[0]["posicion"] if not sub.empty else "MF")
        club_fb = _club_rate_fbref(sub)
        tiros_90, sot_rate, fuente, xg_per_shot = _tasa_tiros_jugador(nrm, pos, datos_tiros, club_fb)
        filas.append({"nombre": nombre, "nrm": nrm, "pos": pos,
                      "tiros_90": tiros_90, "sot_rate": sot_rate,
                      "fuente": fuente, "xg_per_shot": xg_per_shot})

    if not filas:
        return {}

    sum_tiros = sum(f["tiros_90"] for f in filas)
    # Si el XI mapea menos de 11 (selecciones con poca cobertura FBref, ej. Australia 3/11),
    # los jugadores NO mapeados igual consumen volumen de tiros. Sin esto, los pocos mapeados
    # absorben todo -> usage al tope (0.35) -> lambda inflada (defensores con lambda ~4). Padear
    # el denominador con los faltantes al ritmo sombra de un mediocampista reparte realista; los
    # pocos mapeados caen a usage sensato (y si quedan bajo el umbral, no se reportan, que es lo
    # honesto: no tenemos data para nombrar al disparador de ese equipo).
    shadow = meta.get("shadow_tiros_90", {"FW": 2.46, "MF": 1.33, "DF": 0.5, "GK": 0.1})
    n_falt = max(0, 11 - len(filas))
    sum_tiros += n_falt * shadow.get("MF", 1.33)
    if sum_tiros <= 0:
        sum_tiros = 1.0

    resultado = {}
    for f in filas:
        usage = max(USAGE_MIN, min(USAGE_MAX, f["tiros_90"] / sum_tiros))
        lam_j = lam_shots_team * usage
        lam_sot = lam_j * f["sot_rate"]
        xG_base = round(lam_j * f["xg_per_shot"], 3)
        resultado[f["nombre"]] = {
            "lam": round(lam_j, 3),
            "xG_base": xG_base,
            "p_over_0_5": round(prob_over_k(lam_j, 0.5), 3),
            "p_over_1_5": round(prob_over_k(lam_j, 1.5), 3),
            "p_over_2_5": round(prob_over_k(lam_j, 2.5), 3),
            "p_sot_1": round(prob_over_k(lam_sot, 0.5), 3),
            "fuente": f["fuente"],
            "pos": f["pos"],
        }
    return resultado


def calcular_props_partido(info: dict, jugadores_df: pd.DataFrame,
                           datos_tiros: dict) -> dict:
    """Props completas del partido. info tiene keys: xi_l, xi_v, cod_l, cod_v, res.

    res viene de mundial_engine.correr() y tiene goles_esp_local / goles_esp_visitante.
    """
    r = info.get("res", {})
    lam_l = r.get("goles_esp", [1.2, 1.2])[0] if isinstance(r.get("goles_esp"), (list, tuple)) else r.get("goles_esp_local", 1.2)
    lam_v = r.get("goles_esp", [1.2, 1.2])[1] if isinstance(r.get("goles_esp"), (list, tuple)) else r.get("goles_esp_visitante", 1.2)

    props_l = calcular_props_equipo(
        info.get("xi_l") or [], info.get("cod_l", ""),
        lam_l, jugadores_df, datos_tiros)
    props_v = calcular_props_equipo(
        info.get("xi_v") or [], info.get("cod_v", ""),
        lam_v, jugadores_df, datos_tiros)

    return {
        "local": {"nacion": info.get("cod_l", ""), "jugadores": props_l},
        "visitante": {"nacion": info.get("cod_v", ""), "jugadores": props_v},
        "meta": {
            "lam_goles_l": round(lam_l, 3),
            "lam_goles_v": round(lam_v, 3),
            "conv_rate": datos_tiros.get("meta", {}).get(
            "conversion_rate_xg",
            datos_tiros.get("meta", {}).get("conversion_rate_intl", 0.091)),
        },
    }


def top_props(props_partido: dict, min_lam: float = LAMBDA_JUGADOR_MIN_REPORTAR,
              top_n: int = 5) -> list[dict]:
    """Lista plana de los N jugadores con mayor λ de tiros de AMBOS equipos.

    Solo incluye jugadores con fuente != 'shadow' y lam >= min_lam.
    Ordenado por P(over 1.5 tiros) desc.
    """
    out = []
    for lado in ("local", "visitante"):
        nac = props_partido[lado]["nacion"]
        for nombre, v in props_partido[lado]["jugadores"].items():
            if v["fuente"] == "shadow" or v["lam"] < min_lam:
                continue
            out.append({"nombre": nombre, "nacion": nac, "lado": lado, **v})
    return sorted(out, key=lambda x: x["p_over_1_5"], reverse=True)[:top_n]
