# ⚽ D-Soccer — Motor de analítica predictiva de fútbol

Sistema de predicción de resultados, goles, tarjetas y faltas en fútbol, combinando **modelos estadísticos** (Poisson / Dixon-Coles), **ratings Elo**, y **simulación Monte Carlo**, calibrados con datos históricos reales. Desarrollado como motor de analítica para el Mundial 2026, con arquitectura extensible a ligas de clubes.

## Filosofía

- Los **números** (goles, tarjetas, faltas) los predicen modelos estadísticos calibrados con miles de partidos reales — no estimaciones subjetivas.
- La fuerza de una selección se modela combinando **ratings Elo históricos** (columna vertebral) con la **forma actual de sus jugadores en sus clubes** (ajuste bottom-up).
- Todo modelo se **valida por benchmark antes de reemplazar el anterior** — por ejemplo, un modelo de Machine Learning (LightGBM) fue descartado tras comparación directa porque el modelo estadístico bien especificado (Dixon-Coles) lo superó en precisión (Brier Score).
- Las probabilidades del modelo se comparan contra las **cuotas de mercado de casas de apuestas** como benchmark externo de calibración — es una técnica estándar en forecasting deportivo para medir qué tan bien calibradas están las predicciones frente al consenso del mercado.
- La IA generativa se usa solo para *explicar* y dar *contexto* a los números, nunca para generarlos.

## Cómo arrancar (setup para colaboradores)

Requiere **Python 3.11+** y, para la ingesta de jugadores, **Google Chrome** instalado
(soccerdata usa un navegador para scrapear FBref).

```bash
# 1. Clonar y entrar
git clone <URL-del-repo>
cd d-soccer

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

# 5. Calibrar el motor con datos reales del Mundial
python -m src.calibrar_internacional

# 6. Correr
python -m src.demo "Real Madrid" "Barcelona"        # modelo de clubes (goles)
python -m src.montecarlo "Real Madrid" "Barcelona"  # simulacion 10k
python -m src.mundial_engine ARG FRA --arbitro "X"  # motor completo de selecciones
python -m src.backtest                              # validar el modelo de goles

# 7. Dashboard interactivo
python -m streamlit run app.py
```

> **Nota datos**: `data/raw/` y `data/processed/*.csv` estan en `.gitignore` (se regeneran
> con los `ingest`). Si querés evitarle a alguien más el scraping lento de FBref, podés
> commitear `data/processed/jugadores.csv` aparte (son ~1.3 MB).

## Estructura

```
src/
  config.py                # ligas, temporadas, coef. calidad, ratings sombra, mapeos nacion
  ingest.py                # descarga clubes (football-data.co.uk) -> partidos.csv
  ingest_jugadores.py      # descarga jugadores (FBref/soccerdata) -> jugadores.csv
  dixon_coles.py           # modelo de goles de clubes (validado, Brier 0.587)
  estilos_model.py         # tarjetas/faltas/corners + factor arbitro (validado)
  jugadores_model.py       # motor de selecciones: jugadores -> fuerza de seleccion (con xG)
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
  telegram_alert.py        # notificaciones: reporte de partido + props por jugador con xG
  fixtures.py              # API-Football: fixtures, alineaciones, cuotas (hardened)
  backtest.py              # validacion walk-forward goles de clubes
  train_ml.py              # LightGBM benchmarkeado (perdio; documentado)
paginas/
  dashboard.py             # Streamlit: simulacion Montecarlo + scatter xPts vs Puntos
  perfil_equipo.py         # Streamlit: KPIs de equipo + lineas de mercado
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
- [x] **5. Motor de selecciones** (ratings por jugador → fuerza de selección, bottom-up).
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
- [x] **8. Orquestador final** (`src/mundial_engine.py`): fusiona motor de selecciones +
      Montecarlo. Jugadores → tasas (goles + faltas/tarjetas agregadas de la disciplina
      del XI + factor árbitro observado/esperado) → 10.000 sims → distribución completa
      de probabilidades (1X2, marcadores, O/U goles-tarjetas-córners, roja).
- [x] **9. Calibración internacional** (`src/calibrar_internacional.py`): ancla el mapeo
      rating→goles a la media real de goles del Mundial (medida de FBref: 1.39 goles/equipo
      sobre 128 partidos de 2018+2022). Centra ataque/defensa en el promedio de 71 selecciones.
      ARG-FRA: 4.47 → 3.16 goles totales (realista).
- [x] **10. Imputación por niveles (Tiered Imputation)** para el "long tail": selecciones con
      jugadores en ligas locales no scrapeadas. Cascada país→confederación→default. El motor
      corre CUALQUIER cruce (ej. `python -m src.mundial_engine NZL ARG`).
- [x] **11. Shrinkage de disciplina por minutos** (empirical Bayes): `faltas_90`, `amarillas_90`
      y `rojas_90` se suavizan hacia el prior de su POSICIÓN. Arregla el ruido de jugadores
      con pocos minutos. NZ: faltas **19.4 → 11.9** (realista).
- [x] **12. Compresión dinámica por brecha de calidad**: corrige la sub-diferenciación en
      partidos asimétricos. **ARG-NZL: 48% → 79%** (nivel de mercado).
- [x] **13. Rating ofensivo tipo-xG**: `rating_ofensivo = 0.7·npg_90 + 0.3·ast_90`, corregido
      con xG real de StatsBomb.
- [x] **13b. Híbrido Elo + jugadores**: Elo de selecciones (eloratings.net) como columna
      vertebral + modelo de jugadores como ajuste. MEX-RSA 47%→67% (=mercado 66%, real 2-0).
- [x] **14. StatsBomb Open Data**: cosecha gratuita de ~314 partidos de selecciones 2018-24
      con XI real, xG evento a evento y árbitro. Usado para calibrar todos los modelos.
- [x] **15. Árbitros por faltas — Empirical Bayes**: 66 árbitros calibrados desde StatsBomb.
      MAE LOO: **+10.7% vs baseline**.
- [x] **16. Player Props Poisson**: modelo de tiros/SOT por jugador confirmado en el XI,
      usando usage-rate de historial internacional + xG real. Validación Brier **+13.9%**
      (1018 jugador-partido).
- [x] **17. xG Enrichment jugadores**: corrige `npg_90` de FBref con xG real internacional
      (StatsBomb). 724 jugadores matcheados; 183 con factor ≠ 1.
- [x] **18. Pipeline robusto**: audit trail por corrida, reintentos con backoff, logging
      rotativo, notificaciones multi-destinatario.

### Estado: **V2 completa** — motor de selecciones + player props validados y desplegados.
- El sistema predice 1X2 + goles + faltas + tarjetas + **Player Props (tiros/xG/SOT)** por jugador.
- Calibración validada contra mercado: faltas MAE +10.7% · props Brier +13.9% · 1X2 ≈ consenso de mercado (Elo backbone).

> **Nota sobre validación**: superar un benchmark interno no es lo mismo que superar el consenso
> de mercado en general — el motor de **selecciones** es un *prior* sin validación exhaustiva
> (el motor de clubes sí está validado con backtesting walk-forward completo). Para medir la
> calidad de la calibración se agregó instrumentación de **CLV** (closest-line value, proxy de
> menor varianza) y **reliability diagrams** (exponen sobreconfianza del modelo).

## Fuentes de datos

- **[football-data.co.uk](https://www.football-data.co.uk/)** — partidos de clubes: goles, tiros, faltas, córners, tarjetas, árbitro. Gratis, sin API key.
- **[StatsBomb Open Data](https://github.com/statsbomb/open-data)** — partidos internacionales 2018-24 con xG real evento a evento, XI titulares, árbitro. Gratis, sin cuota. Usado para calibrar árbitros, props y xG enrichment.
- **[FBref / soccerdata](https://soccerdata.readthedocs.io/)** — ~12k jugadores de 10 ligas: tiros, goles, asistencias, faltas, minutos. Requiere scraping con Chrome.
- **[API-Football](https://www.api-football.com/)** — fixtures del Mundial en vivo, alineaciones confirmadas, cuotas de mercado (usadas como benchmark de calibración). Plan gratis: 100 req/día.
- **[eloratings.net](https://www.eloratings.net/)** — Elo histórico de selecciones (backbone del modelo híbrido).
