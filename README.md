# ⚽ Motor de predicciones de fútbol

Sistema personal para predecir resultados, goles, tarjetas y faltas combinando
**modelos estadísticos** (Poisson / Dixon-Coles) con datos históricos reales.
Pensado como base para análisis del Mundial.

## Filosofía

- Los **números** (goles, tarjetas) los predicen los modelos estadísticos, que están
  calibrados con miles de partidos. Es lo que mejor funciona.
- La fuerza de una selección NO se mide por los duelos directos entre selecciones
  (hay poquísimos), sino agregando la **forma actual de sus jugadores en sus clubes**.
  → esa es una capa futura que se monta encima de este motor.
- La IA generativa (capa final) sirve para *explicar* y dar *contexto*, no para
  inventar los números.

## Cómo arrancar (setup para colaboradores)

Requiere **Python 3.11+** y, para la ingesta de jugadores, **Google Chrome** instalado
(soccerdata usa un navegador para scrapear FBref).

```bash
# 1. Clonar y entrar
git clone <URL-del-repo>
cd ambicion

# 2. Entorno virtual (recomendado)
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  Linux/Mac:  source .venv/bin/activate

# 3. Instalar dependencias
#    OJO: soccerdata arrastra 'pip' como dependencia, asi que hay que usar 'python -m pip'
python -m pip install -r requirements.txt

# 4. Bajar los datos (se regeneran, NO van en el repo)
python -m src.ingest             # ~7000 partidos de clubes (rapido, football-data.co.uk)
python -m src.ingest_jugadores   # ~12000 jugadores de 10 ligas (LENTO: scraping FBref)
                                 # auto-instala el league_dict.json de soccerdata solo

# 5. Calibrar el Motor Mundialista con datos reales del Mundial
python -m src.calibrar_internacional

# 6. Correr
python -m src.demo "Real Madrid" "Barcelona"        # modelo de clubes (goles)
python -m src.montecarlo "Real Madrid" "Barcelona"  # simulacion 10k
python -m src.mundial_engine ARG FRA --arbitro "X"  # Motor Mundialista completo
python -m src.backtest                              # validar el modelo de goles
```

> **Nota datos**: `data/raw/` y `data/processed/*.csv` estan en `.gitignore` (se regeneran
> con los `ingest`). Si querés evitarle a tu amigo el scraping lento de FBref, podés
> commitear `data/processed/jugadores.csv` aparte (son ~1.3 MB).

## Estructura

```
src/
  config.py                # ligas, temporadas, coef. calidad, ratings sombra, mapeos nacion
  ingest.py                # descarga clubes (football-data.co.uk) -> partidos.csv
  ingest_jugadores.py      # descarga jugadores (FBref/soccerdata) -> jugadores.csv
  dixon_coles.py           # modelo de goles de clubes (validado, Brier 0.587)
  estilos_model.py         # tarjetas/faltas/corners + factor arbitro (validado)
  jugadores_model.py       # Motor Mundialista: jugadores -> fuerza de seleccion (con xG)
  enriquecer_xg.py         # corrige npg_90 con xG real StatsBomb -> xg_ajuste.csv
  arbitros_faltas.py       # factor faltas por arbitro Empirical Bayes -> arbitros_faltas.json
  montecarlo.py            # simulador 10k iteraciones, minuto a minuto (Factor Caos)
  mundial_engine.py        # orquestador: jugadores + Elo + montecarlo + arbitro + calibracion
  calibrar_internacional.py# ancla goles a la media real del Mundial (1.39 goles/equipo)
  ingesta_historica.py     # cosecha StatsBomb gratis -> dsoccer_historico.db (xG real)
  props_data.py            # calibracion offline tiros+xG -> tiros_intl.json
  props_lineups.py         # state machine de alineaciones (max 3 API calls/fixture)
  props_model.py           # Poisson props: lambda_shots, xG_base por jugador
  validar_props.py         # compuerta Brier para props (gate +13.9%)
  validar_statsbomb.py     # tunea w (Elo vs jugadores), gate LOO (w=0.85)
  autorun.py               # cron: props (20-120min) + registro (20-45min) + audit trail
  telegram_alert.py        # push: reporte partido + props por equipo con xG
  fixtures.py              # API-Football: fixtures, alineaciones, cuotas (hardened)
  backtest.py              # validacion walk-forward goles de clubes
  train_ml.py              # LightGBM benchmarkeado (perdio; documentado)
data/
  raw/dsoccer_historico.db # SQLite StatsBomb (gitignored, se regenera)
  processed/               # jugadores.csv, calibracion.json, elo.json,
                           # arbitros_faltas.json, tiros_intl.json, xg_ajuste.csv
```

## Estado / Roadmap

- [x] **1. Ingesta** de datos reales (7 ligas, 3 temporadas, ~7166 partidos)
- [x] **2. Dixon-Coles** para goles (resultado 1X2, marcador, over/under)
- [x] **3. Backtesting** walk-forward + Brier Score → **0.587 vs 0.652 benchmark (+10%)**,
      gana en las 7 ligas. (`python -m src.backtest`)
- [x] **4. Modelo de estilos** (tiros/córners/faltas/tarjetas) con factor árbitro
      + shrinkage. (`src/estilos_model.py`, validado con `src/backtest_estilos.py`)
      MAE vs benchmark: faltas +6.8%, tarjetas +2.1% (aportan); tiros +0.1%,
      córners −0.9% (el total es casi inpredecible, la señal está en el reparto).
- [x] **4b. Factor árbitro relativo a SU liga** (método observado/esperado): arregla el
      confounding liga/árbitro. Neutro en el backtest europeo (+3.0% → +2.1%, dentro del
      ruido) pero ESENCIAL para el Mundial, donde los árbitros cruzan confederaciones con
      culturas de arbitraje muy distintas. El ranking ahora significa "estricto para su liga".
- [x] **5. Motor Mundialista** (ratings por jugador → fuerza de selección, bottom-up).
      Datos: FBref vía soccerdata (`src/ingest_jugadores.py`). **12.144 jugadores, 137 países,
      10 ligas**: Big-5 + MLS (Messi), Brasileirão, Saudí (Cristiano), Eredivisie, Primeira.
      Modelo: `src/jugadores_model.py` (rating ofensivo/defensivo con shrinkage por minutos +
      calidad de liga; XI probable por CALIDAD por posición; agregación convexa; mapeo a Poisson).
      **xG corregido**: `npg_90` se ajusta con factor desde StatsBomb (Cristiano +23%, Lukaku +26%,
      Musiala -19%, Gakpo -19%). Ver ítem 17.
- [x] **6. Simulador de Montecarlo** (`src/montecarlo.py`): juega el partido 10.000 veces
      minuto a minuto con dependencias dinámicas (Factor Caos: rojas debilitan, el que
      pierde arriesga). Agnóstico al modelo: toma tasas de Dixon-Coles/estilos/jugadores
      y devuelve distribución completa (1X2, moda de marcadores, over/under goles-tarjetas-
      córners, prob. de roja). Coherente con Dixon-Coles (RM-Barça 43/26/31).
- [x] **7. ML (LightGBM) benchmarkeado** (`src/train_ml.py`): features de forma reciente
      (promedios móviles sin fuga) + objetivo Poisson, head-to-head justo contra Dixon-Coles.
      **RESULTADO: el ML PIERDE (Brier 0.612 vs 0.592 de Dixon-Coles).** El modelo estadístico
      bien especificado gana, como predice la literatura. Dixon-Coles SIGUE siendo el núcleo.
      La disciplina de "benchmark antes de reemplazar" evitó degradar el sistema un 3.4%.
      Para que el ML compita: usar ratings DC como features, o ensemble (no reemplazar).
- [x] **8. Orquestador final** (`src/mundial_engine.py`): fusiona Motor Mundialista +
      Montecarlo. Jugadores → tasas (goles + faltas/tarjetas agregadas de la disciplina
      del XI + factor árbitro observado/esperado) → 10.000 sims → distribución completa
      de mercado con CUOTAS IMPLÍCITAS (1X2, marcadores, O/U goles-tarjetas-córners, roja).
      `python -m src.mundial_engine ARG FRA [--arbitro "X"] [--xi-local "a,b,..."]`
- [x] **9. Calibración internacional** (`src/calibrar_internacional.py`): ancla el mapeo
      rating→goles a la media real de goles del Mundial (medida de FBref: 1.39 goles/equipo
      sobre 128 partidos de 2018+2022). Centra ataque/defensa en el promedio de 71 selecciones
      (arregla el desajuste de escala: ataque_ref 2.02 vs defensa_ref 1.30). `mundial_engine`
      carga `calibracion.json` y lo aplica solo. ARG-FRA: 4.47 → 3.16 goles totales (realista).
      El NIVEL es riguroso (dato real); la COMPRESIÓN (0.45) es un prior, no ajustable sin
      alineaciones históricas. Tunable en calibrar_internacional.py si se quieren partidos más cerrados.
- [x] **10. Imputación por niveles (Tiered Imputation)** para el "long tail": selecciones con
      jugadores en ligas locales no scrapeadas. `JugadoresModel._shadow_rating` (cascada
      país→confederación→default, en `config.CONFED_BASELINE`/`NACION_SHADOW`/`NACION_CONFED`).
      `construir_fuerza_seleccion`/`disciplina_seleccion` rellenan hasta 11 con ratings sombra;
      la curva convexa preserva el efecto estrella (Almirón/Mitoma arrastran el Alpha). El motor
      corre CUALQUIER cruce (ej. `python -m src.mundial_engine NZL ARG`). ⚠️ Sombras = priors.
- [x] **11. Shrinkage de disciplina por minutos** (empirical Bayes): `faltas_90`, `amarillas_90`
      y `rojas_90` se suavizan hacia el prior de su POSICIÓN con K≈900 min (10 partidos).
      Arregla el ruido de jugadores con pocos minutos. NZ: faltas **19.4 → 11.9** (realista);
      equipos bien muestreados casi no cambian (ARG 12.5). `tarjetas_90 = amarillas_90 + rojas_90`.
- [x] **13. Rating ofensivo tipo-xG**: `rating_ofensivo = 0.7·npg_90 + 0.3·ast_90`. El `npg_90`
      se corrige con xG real de StatsBomb (ítem 17); los datos de club no exponen xG vía soccerdata
      pero el enriquecimiento intl compensa los casos extremos. `C_MAX` en 1.7.
- [x] **14. StatsBomb Open Data** (`src/ingesta_historica.py`): cosecha gratuita de ~314 partidos
      de selecciones 2018-24 con XI real, xG evento a evento y árbitro → `dsoccer_historico.db`.
      Sin cuota, sin API key, idempotente por `match_id`. Usado para calibrar todos los modelos intl.
- [x] **15. Árbitros por faltas — Empirical Bayes** (`src/arbitros_faltas.py`, Fase 1A):
      reemplaza el escalar estático 1.21 con factores por árbitro desde StatsBomb (K=8 shrinkage).
      66 árbitros calibrados. MAE LOO: **+10.7% vs baseline** (GATE PASADO).
      Letexier -10%, Zwayer -12%, Artur Dias -15%. Se muestra en el reporte Telegram junto al
      factor de tarjetas de `estilos_model` (son capas independientes).
- [x] **16. Player Props Poisson** (`src/props_model.py`, Fase 2 + 3A):
      modelo de tiros/SOT por jugador confirmado en el XI. Distribuye λ_goles del engine entre
      el XI usando usage-rate de historial internacional StatsBomb + xG real:
      ```
      conv_rate_xg = Σ(xg) / Σ(tiros) = 0.121   (más estable que goles reales)
      λ_shots_i    = (λ_goles / conv_rate_xg) × (tiros_intl_i / Σ tiros_j)
      xG_base_i    = λ_shots_i × xG_per_shot_i   (calidad de posición, display)
      ```
      Pipeline: `props_data.py` calibra → `props_lineups.py` state machine (max 3 API calls/fixture)
      → `props_model.py` → Telegram ~70-90min antes del KO.
      Validación (Brier vs baseline): **+13.9%** (1018 jugador-partido). GATE PASADO.
      Alerta Telegram muestra top 4 por equipo con λ, xG_base, P(>1.5), P(SOT≥1).
- [x] **17. xG Enrichment jugadores** (`src/enriquecer_xg.py`, Fase 3B):
      corrige `npg_90` de FBref con xG real internacional (StatsBomb, match difuso por nación):
      `factor = clamp((xG+K)/(goles+K), 0.6, 1.6)^0.5` con K=3. 724 jugadores matcheados;
      183 con factor ≠ 1. Premia generadores (Cristiano +23%, Osimhen +22%, Lukaku +26%);
      castiga suertudos (Musiala -19%, Gakpo -19%). Artefacto `xg_ajuste.csv` en git.
- [x] **18. Pipeline robusto + Telegram completo** (Fase 1B):
      AUDIT TRAIL por corrida, `COMPETICIONES_WC` set de variantes, `_api_get()` con 3 reintentos
      y backoff 429, `autorun.log` rotativo. `registrar_props()` (20-120min) corre antes que
      `registrar_proximos()` (20-45min). Multi-destinatario vía `suscriptores.txt`.

### Estado: **V2 MUNDIALISTA COMPLETA** — Fases 1A/1B/2/3 validadas y desplegadas.
- El sistema predice 1X2 + goles + faltas + tarjetas + **Player Props (tiros/xG/SOT)** por jugador.
- Avisa por Telegram ~30-40min antes (reporte partido) y ~70-90min antes (props con XI confirmado).
- Edge validado: faltas MAE +10.7% · props Brier +13.9% · 1X2 ≈ mercado (Elo backbone).

> ⚠️ **Honestidad antes de apostar** (ver CONTEXTO.md §5): ganarle al benchmark ≠ ganarle al mercado
> ≠ ser rentable. El Motor de **selecciones** es un PRIOR sin validar (el edge en 1X2 es hipótesis;
> lo gana el Elo). Mercados de **Over/Under totales NO tienen edge** (validado impredecible). Para medir
> edge de verdad se agregó instrumentación (commit `bc5a1a5`): **CLV** (proxy más rápido y de menor
> varianza, `validacion.py`), **reliability diagram** (calibración, expone sobreconfianza), y **detector
> vs Pinnacle de-marginado** (separa edge-modelo de line-shopping; ya no marca valor por el outlier que
> más paga). Regla: el detector da hipótesis, **el CLV las confirma o las mata.**
- [x] **12. Compresión dinámica por brecha de calidad** (corrige la sub-diferenciación en
      partidos asimétricos): la compresión `c` deja de ser fija; crece de forma no-lineal con
      la brecha de fuerza (`c = c_base + GAP_AMP·gap^2`, cap `C_MAX`). Parejo → c_base (ARG-FRA
      intacto); asimétrico → c sube y estira la ventaja del favorito. **ARG-NZL: 48% → 79%**
      (cuota 1.27, nivel casas). Preserva el nivel base (en gap=0, c=c_base, ratios=1 → 1.39).
      Es un prior tuneado al consenso de las casas, no validado con datos.
      Nota: persiste sub-diferenciación LEVE en brechas medias (ESP-USA ~39/33); el extremo está resuelto.
- [x] **13b. Híbrido Elo + jugadores** (`src/elo.py` + `jugadores_model`): arregla la
      sub-diferenciación de RAÍZ. Elo de selecciones (eloratings.net, gratis) como columna
      vertebral + modelo de jugadores como ajuste. `λ = w·λ_elo + (1−w)·λ_player` (w=0.5 provisional).
      RESULTADO: MEX-RSA 47%→67% (=mercado 66%, real 2-0), POR-NGA 38%→56%; parejos se mantienen
      (ARG-FRA 37/36). El reporte muestra el desglose Elo vs jugadores.
- [ ] Pendiente: TUNEAR `w` empíricamente en el harvest 2024 (Brier, train/test); comparar
      híbrido vs bottom-up; usar Elo ofensivo/defensivo por separado.

## Fuentes de datos

- **[football-data.co.uk](https://www.football-data.co.uk/)** — partidos de clubes: goles, tiros, faltas, córners, tarjetas, árbitro. Gratis, sin API key.
- **[StatsBomb Open Data](https://github.com/statsbomb/open-data)** — partidos internacionales 2018-24 con xG real evento a evento, XI titulares, árbitro. Gratis, sin cuota. Usado para calibrar árbitros, props y xG enrichment.
- **[FBref / soccerdata](https://soccerdata.readthedocs.io/)** — ~12k jugadores de 10 ligas: tiros, goles, asistencias, faltas, minutos. Requiere scraping con Chrome.
- **[API-Football](https://www.api-football.com/)** — fixtures del Mundial en vivo, alineaciones confirmadas, cuotas. Plan gratis: 100 req/día (suficiente con el state machine de props).
- **[eloratings.net](https://www.eloratings.net/)** — Elo histórico de selecciones (backbone del modelo híbrido).
