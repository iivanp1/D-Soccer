"""Factor de rigurosidad de FALTAS por arbitro (data-driven, StatsBomb Open Data).

Offline: lee dsoccer_historico.db y computa el factor relativo de cada arbitro
(cuanto mas/menos faltas que la media global sanciona). Shrinkage empirical-Bayes
hacia 1.0 (neutro) con K=8: con n=5 partidos el factor es solo 38% de los datos
reales + 62% global -> protege contra overfit en arbitros con poca muestra.

El factor REEMPLAZA la constante ESCALA_FALTAS_SELECCION (1.21) cuando el arbitro
es conocido: factor_final = 1.21 * shrunk_factor. Clampeado a [ARB_MIN, ARB_MAX].
Arbitro desconocido o sin datos -> factor = 1.21 (comportamiento actual intacto).

Artefacto comprometido en git (como calibracion.json / xg_ajuste.csv): el server
solo necesita 'git pull', no re-cosecha la DB.

Uso:
    python -m src.arbitros_faltas          # construye arbitros_faltas.json
    python -m src.arbitros_faltas --test   # muestra top/bottom arbitros
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from src import config
from src.jugadores_model import _norm

DB = config.DATA_PROC / "dsoccer_historico.db"
SALIDA = config.DATA_PROC / "arbitros_faltas.json"

# Shrinkage: partidos equivalentes que valen la media global (K alto = mas conservador).
# Con K=8 y n=5: factor = 38% raw + 62% global -> correccion maxima ~7-8% con 5 partidos.
K_ARB = 8

# Clamp del factor RELATIVO (antes de multiplicar por 1.21). 0.75 = max -25% vs global.
ARB_MIN = 0.75
ARB_MAX = 1.35


def construir(db: Path = DB, salida: Path = SALIDA) -> dict:
    """Lee la DB de StatsBomb, computa factores por arbitro y guarda el JSON."""
    if not db.exists():
        print(f"DB no encontrada: {db}")
        print("Corre primero: python -m src.ingesta_historica")
        return {}

    con = sqlite3.connect(db)
    filas = con.execute(
        "SELECT p.arbitro, e.faltas FROM equipo_partido_stats e "
        "JOIN partidos p ON p.match_id = e.match_id "
        "WHERE e.faltas IS NOT NULL AND p.arbitro IS NOT NULL"
    ).fetchall()
    con.close()

    if not filas:
        print("Sin datos de faltas/arbitro en la DB. Revisa la ingesta historica.")
        return {}

    all_faltas = [f for _, f in filas]
    media_global = sum(all_faltas) / len(all_faltas)

    por_arb: dict[str, list[float]] = defaultdict(list)
    for arb, f in filas:
        # "Said Martinez, Honduras" -> "said martinez" (normalizar solo el nombre)
        arb_norm = _norm(arb.split(",")[0].strip())
        por_arb[arb_norm].append(float(f))

    arbitros = {}
    for arb_norm, lst in sorted(por_arb.items()):
        n = len(lst)
        media_arb = sum(lst) / n
        factor_raw = media_arb / media_global  # >1 = permisivo, <1 = estricto
        # Empirical Bayes: con K=8 y n=5, el factor se tira 62% hacia neutral (1.0)
        factor_shrunk = (n * factor_raw + K_ARB * 1.0) / (n + K_ARB)
        factor_shrunk = max(ARB_MIN, min(ARB_MAX, factor_shrunk))
        arbitros[arb_norm] = {
            "n": n,
            "media": round(media_arb, 2),
            "factor_raw": round(factor_raw, 4),
            "factor_shrunk": round(factor_shrunk, 4),
        }

    datos = {
        "meta": {
            "n_equipos": len(all_faltas),
            "media_global": round(media_global, 2),
            "k_arb": K_ARB,
        },
        "arbitros": arbitros,
    }
    salida.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[arbitros_faltas] {len(arbitros)} arbitros | media global {media_global:.1f} faltas/equipo "
          f"| guardado en {salida.name}")
    return datos


def cargar(path: Path = SALIDA) -> dict:
    """Carga el JSON de factores. Devuelve {} si no existe (factor=1.0 para todos)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def factor_faltas(arbitro: str | None, datos: dict) -> tuple[float, str]:
    """Retorna (factor_final, descripcion) para el arbitro dado.

    factor_final ya incluye ESCALA_FALTAS_SELECCION (1.21). Listo para pasar
    directamente a disciplina_seleccion(factor_faltas=...).
    Si el arbitro es desconocido o datos esta vacio, devuelve (1.21, descripcion)
    -> comportamiento identico al anterior.
    """
    base = config.ESCALA_FALTAS_SELECCION
    if not datos or not arbitro:
        return base, f"arbitro no especificado -> escala global ({base:.2f}x)"

    arbs = datos.get("arbitros", {})
    # Normalizar: "Szymon Marciniak, Poland" -> "szymon marciniak"
    arb_norm = _norm(arbitro.split(",")[0].strip())

    if arb_norm in arbs:
        info = arbs[arb_norm]
        f_shrunk = info["factor_shrunk"]
        f_final = max(ARB_MIN * base, min(ARB_MAX * base, base * f_shrunk))
        pct = (f_shrunk - 1.0) * 100
        signo = "+" if pct >= 0 else ""
        return f_final, (f"{arbitro.split(',')[0].strip()}: {signo}{pct:.0f}% vs global "
                         f"({info['n']} partidos, escala {f_final:.2f}x)")

    return base, f"{arbitro.split(',')[0].strip()}: sin historial StatsBomb -> escala global ({base:.2f}x)"


def main() -> None:
    datos = construir()
    if not datos:
        return
    if "--test" in sys.argv and datos:
        arbs = sorted(datos["arbitros"].items(), key=lambda x: x[1]["factor_shrunk"])
        media_g = datos["meta"]["media_global"]
        print(f"\n  media global: {media_g:.1f} faltas/equipo-partido\n")
        print("  Top 5 ESTRICTOS (factor_shrunk bajo = menos faltas que la media):")
        for a, v in arbs[:5]:
            print(f"    {a:<32} n={v['n']:3d}  media={v['media']:.1f}  "
                  f"factor={v['factor_shrunk']:.3f}  ({(v['factor_shrunk']-1)*100:+.0f}%)")
        print("  Top 5 PERMISIVOS:")
        for a, v in arbs[-5:]:
            print(f"    {a:<32} n={v['n']:3d}  media={v['media']:.1f}  "
                  f"factor={v['factor_shrunk']:.3f}  ({(v['factor_shrunk']-1)*100:+.0f}%)")


if __name__ == "__main__":
    main()
