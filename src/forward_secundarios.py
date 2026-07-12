"""Forward testing de MERCADOS SECUNDARIOS de clubes (tiros a puerta y corners).

Por que existe: no hay cuotas historicas gratis de corners/SOT para backtestear, asi que
el edge se valida EN VIVO, jornada a jornada, SIN plata (paper trading) desde agosto 2026.

La senal (validada retro sobre la 2526, n=2832 equipo-partidos, sin fuga):
  - TIROS A PUERTA del equipo: media rolling (ultimos 10 en esa condicion) vs la linea.
    Cuando |mu - linea| cae en el tercil alto: skill +10.7% y acierto del pick 74%.
    El filtro que funciona es la DISTANCIA a la linea, NO la sigma baja (verificado:
    la hipotesis 'sd baja = mas predecible' no se sostiene en los datos).
  - CORNERS: SIN resolucion con este metodo (skill -0.7%); se loguean como CONTROL,
    no como apuesta.

Flujo (cron diario del server desde agosto):
  registrar  -> fixtures de HOY+MANANA (API-Football por fecha, gratis) de las 4 ligas;
                por equipo: mu/sd rolling desde partidos.csv; PICK si dist >= UMBRAL_DIST;
                intenta capturar cuotas de corners/SOT si la API las ofrece.
  liquidar   -> baja Corner Kicks / Shots on Goal reales de los partidos terminados.
  reporte    -> hit rate del pick, Brier vs base-rate, ROI simulado si hubo cuota.

Log: data/processed/predicciones_secundarios.csv (matchea el patron gitignored
predicciones_*.csv: es un log operativo, no un artefacto del repo).

REGLA DE DISCIPLINA (pactada): nada de plata real hasta >= 100 picks liquidados con
hit rate y ROI simulado positivos. El retro dice 74%; el forward dice si las CUOTAS
reales de las casas ya lo pricean (probablemente en parte: las lineas por equipo se
ajustan). Esa es exactamente la pregunta que este script responde.

Uso:
    python -m src.forward_secundarios registrar [--dia YYYY-MM-DD]
    python -m src.forward_secundarios liquidar
    python -m src.forward_secundarios reporte
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.stats import poisson

from src import config
from src.fixtures import _api_get

LOG = config.DATA_PROC / "predicciones_secundarios.csv"

# API-Football: ids de las 4 grandes ligas
LIGAS_API = {140: "SP1", 39: "E0", 78: "D1", 135: "I1"}

# Mercados: (nombre, columna en partidos.csv, lineas candidatas)
MERCADOS = [
    ("sot", "tiros_arco", [2.5, 3.5, 4.5]),      # estrella (skill retro +10.7% en dist alta)
    ("corners", "corners", [3.5, 4.5, 5.5]),     # CONTROL (sin skill retro; no apostar)
]
UMBRAL_DIST = 1.0     # |mu - linea| minimo para marcar PICK (tercil alto del retro)
ROLLING = 10
MIN_PREVIOS = 6

COLUMNAS = ["fixture_id", "fecha", "liga", "equipo_api", "equipo_fd", "condicion",
            "mercado", "linea", "mu", "sd", "prob_over", "pick", "dist",
            "cuota_over", "cuota_under", "real", "acierto", "registrado_en"]


# --------------------------------------------------------------------------- #
#  Rolling por equipo+condicion desde partidos.csv (la misma senal del retro)
# --------------------------------------------------------------------------- #
def _historial_fd() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROC / "partidos.csv", parse_dates=["fecha"])
    df = df[df["liga"].isin(LIGAS_API.values())].sort_values("fecha")
    lados = []
    for lado in ("local", "visitante"):
        lados.append(pd.DataFrame({
            "fecha": df["fecha"], "equipo": df[lado], "cond": lado,
            "tiros_arco": df[f"tiros_arco_{lado}"], "corners": df[f"corners_{lado}"]}))
    return pd.concat(lados).dropna().sort_values("fecha")


def mu_sd(hist: pd.DataFrame, equipo_fd: str, cond: str, col: str) -> tuple[float, float] | None:
    """Media y sd de los ultimos ROLLING partidos del equipo EN ESA CONDICION."""
    s = hist[(hist["equipo"] == equipo_fd) & (hist["cond"] == cond)][col].tail(ROLLING)
    if len(s) < MIN_PREVIOS:
        return None
    return float(s.mean()), float(s.std(ddof=1))


def _mapa_api_a_fd(hist: pd.DataFrame) -> dict:
    """Nombre API-Football -> nombre football-data (fuzzy, cache simple en memoria)."""
    return sorted(hist["equipo"].unique())


def _match_fd(nombre_api: str, candidatos: list[str]) -> str | None:
    m = process.extractOne(nombre_api, candidatos, scorer=fuzz.WRatio)
    return m[0] if m and m[1] >= 80 else None


# --------------------------------------------------------------------------- #
#  registrar: fixtures proximos -> predicciones + picks (+ cuotas si hay)
# --------------------------------------------------------------------------- #
def _cuotas_secundarias(fixture_id: int) -> dict:
    """Intenta capturar cuotas de corners del fixture (si el plan free las ofrece).

    API-Football /odds trae mercados tipo 'Total Corners' segun casa/plan. De SOT por
    equipo casi nunca hay pre-match en el free: se registra None y el reporte usa solo
    hit-rate para ese mercado (la disciplina de cuota minima se decide a mano).
    """
    out = {}
    try:
        resp = _api_get("odds", {"fixture": fixture_id})
        for b in (resp[0].get("bookmakers", []) if resp else []):
            for bet in b.get("bets", []):
                if "corner" in bet["name"].lower() and "over/under" in bet["name"].lower():
                    for v in bet["values"]:
                        out.setdefault(str(v.get("value")), float(v["odd"]))
    except Exception:
        pass
    return out


def registrar(dia: str | None = None) -> None:
    config.cargar_env()
    hist = _historial_fd()
    candidatos = _mapa_api_a_fd(hist)
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLUMNAS)
    ya = set(zip(log.get("fixture_id", []), log.get("mercado", []),
                 log.get("equipo_api", []), log.get("linea", [])))

    dias = [dia] if dia else [
        (datetime.now(timezone.utc) + timedelta(days=d)).strftime("%Y-%m-%d") for d in (0, 1)]
    nuevos = []
    for fecha in dias:
        for f in _api_get("fixtures", {"date": fecha}):
            liga_id = f["league"]["id"]
            if liga_id not in LIGAS_API or f["fixture"]["status"]["short"] != "NS":
                continue
            fid = f["fixture"]["id"]
            cuotas = _cuotas_secundarias(fid)
            for lado_api, cond in (("home", "local"), ("away", "visitante")):
                nom_api = f["teams"][lado_api]["name"]
                fd = _match_fd(nom_api, candidatos)
                if not fd:
                    print(f"  (sin match FD para '{nom_api}' -> se saltea; agregar override)")
                    continue
                for mercado, col, lineas in MERCADOS:
                    ms = mu_sd(hist, fd, cond, col)
                    if ms is None:
                        continue
                    mu, sd = ms
                    # la linea mas cercana a mu es la que ofrecen las casas; evaluamos todas
                    for linea in lineas:
                        clave = (fid, mercado, nom_api, linea)
                        if clave in ya:
                            continue
                        p_over = float(1 - poisson.cdf(int(linea), mu))
                        dist = abs(mu - linea)
                        pick = ""
                        if mercado == "sot" and dist >= UMBRAL_DIST:
                            pick = "over" if mu > linea else "under"
                        nuevos.append({
                            "fixture_id": fid, "fecha": f["fixture"]["date"][:10],
                            "liga": LIGAS_API[liga_id], "equipo_api": nom_api,
                            "equipo_fd": fd, "condicion": cond, "mercado": mercado,
                            "linea": linea, "mu": round(mu, 2), "sd": round(sd, 2),
                            "prob_over": round(p_over, 4), "pick": pick,
                            "dist": round(dist, 2),
                            "cuota_over": cuotas.get(f"Over {linea}"),
                            "cuota_under": cuotas.get(f"Under {linea}"),
                            "real": "", "acierto": "",
                            "registrado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        })
    if nuevos:
        log = pd.concat([log, pd.DataFrame(nuevos)], ignore_index=True)
        log.to_csv(LOG, index=False)
    picks = sum(1 for n in nuevos if n["pick"])
    print(f"[forward] {len(nuevos)} lineas registradas ({picks} PICKS de SOT) -> {LOG.name}")


# --------------------------------------------------------------------------- #
#  liquidar: resultados reales de los fixtures terminados
# --------------------------------------------------------------------------- #
def liquidar() -> None:
    config.cargar_env()
    if not LOG.exists():
        print("[forward] sin log todavia.")
        return
    log = pd.read_csv(LOG)
    pendientes = log[log["real"].isna() | (log["real"].astype(str) == "")]
    if pendientes.empty:
        print("[forward] nada pendiente de liquidar.")
        return
    stats_cache: dict[int, dict] = {}
    n = 0
    for fid in pendientes["fixture_id"].unique():
        fid = int(fid)
        if fid not in stats_cache:
            resp = _api_get("fixtures/statistics", {"fixture": fid})
            equipo_stats = {}
            for eq in resp:
                vals = {s["type"]: s["value"] for s in eq.get("statistics", [])}
                equipo_stats[eq["team"]["name"]] = {
                    "corners": vals.get("Corner Kicks"), "sot": vals.get("Shots on Goal")}
            stats_cache[fid] = equipo_stats
        est = stats_cache[fid]
        mask = (log["fixture_id"] == fid) & ((log["real"].isna()) | (log["real"].astype(str) == ""))
        for i in log[mask].index:
            eq = log.at[i, "equipo_api"]
            merc = log.at[i, "mercado"]
            v = (est.get(eq) or {}).get(merc)
            if v is None:
                continue
            log.at[i, "real"] = int(v)
            supero = v > log.at[i, "linea"]
            pick = str(log.at[i, "pick"] or "")
            if pick:
                log.at[i, "acierto"] = int((pick == "over") == supero)
            n += 1
    log.to_csv(LOG, index=False)
    print(f"[forward] {n} lineas liquidadas.")


# --------------------------------------------------------------------------- #
#  reporte: el veredicto acumulado del forward test
# --------------------------------------------------------------------------- #
def reporte() -> None:
    if not LOG.exists():
        print("[forward] sin log todavia. Corre 'registrar' cuando arranque la liga.")
        return
    log = pd.read_csv(LOG)
    hechas = log[pd.to_numeric(log["real"], errors="coerce").notna()].copy()
    hechas["real"] = hechas["real"].astype(float)
    print("=" * 60)
    print(f"  FORWARD TEST secundarios | {len(hechas)} lineas liquidadas")
    print("=" * 60)
    for mercado, _, _ in MERCADOS:
        m = hechas[hechas["mercado"] == mercado]
        if m.empty:
            continue
        y = (m["real"] > m["linea"]).astype(float)
        bs_mod = float(((m["prob_over"] - y) ** 2).mean())
        base = float(y.mean())
        bs_base = float(((base - y) ** 2).mean())
        skill = (1 - bs_mod / bs_base) * 100 if bs_base > 0 else 0
        tag = "ESTRELLA" if mercado == "sot" else "control (no apostar)"
        print(f"\n  [{mercado.upper()}] ({tag})  n={len(m)}  skill vs base-rate {skill:+.1f}%")
        picks = m[m["pick"].astype(str).str.len() > 0]
        if len(picks):
            acierto = pd.to_numeric(picks["acierto"], errors="coerce").mean()
            print(f"    PICKS (dist>={UMBRAL_DIST}): {len(picks)} | acierto {acierto*100:.0f}% "
                  f"(retro decia 74%)")
            con_cuota = picks[picks["cuota_over"].notna() | picks["cuota_under"].notna()]
            if len(con_cuota):
                pnl = 0.0
                for _, r in con_cuota.iterrows():
                    c = r["cuota_over"] if r["pick"] == "over" else r["cuota_under"]
                    if pd.isna(c):
                        continue
                    pnl += (c - 1.0) if r["acierto"] == 1 else -1.0
                print(f"    ROI simulado (con cuota, stake 1u): {pnl/len(con_cuota)*100:+.1f}% "
                      f"sobre {len(con_cuota)} picks")
    print(f"\n  DISCIPLINA: plata real recien con >=100 picks liquidados Y acierto/ROI"
          f"\n  sostenidos. Antes de eso, esto es un experimento, no un cajero.")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "registrar":
        dia = args[args.index("--dia") + 1] if "--dia" in args else None
        registrar(dia)
    elif args[0] == "liquidar":
        liquidar()
    elif args[0] == "reporte":
        reporte()
    else:
        print("Uso: python -m src.forward_secundarios [registrar [--dia YYYY-MM-DD] | liquidar | reporte]")


if __name__ == "__main__":
    main()
