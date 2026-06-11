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
  config.py                # ligas, temporadas, coef. de calidad, ratings sombra
  ingest.py                # descarga clubes (football-data.co.uk) -> partidos.csv
  ingest_jugadores.py      # descarga jugadores (FBref/soccerdata) -> jugadores.csv
  dixon_coles.py           # modelo de goles (validado, Brier 0.587)
  estilos_model.py         # tarjetas/faltas/corners + factor arbitro (validado)
  jugadores_model.py       # Motor Mundialista: jugadores -> fuerza de seleccion
  montecarlo.py            # simulador 10k iteraciones, minuto a minuto (Factor Caos)
  mundial_engine.py        # orquestador final (jugadores + montecarlo + calibracion)
  calibrar_internacional.py# ancla el mapeo de goles a la media real del Mundial
  backtest.py / backtest_estilos.py  # validacion walk-forward (Brier / MAE)
  train_ml.py              # LightGBM benchmarkeado vs Dixon-Coles (perdio; documentado)
  demo.py                  # demo del modelo de clubes
data/
  raw/        # CSVs crudos descargados (gitignored)
  processed/  # partidos.csv, jugadores.csv, calibracion.json
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
      Ingest robusto (pausas anti-bloqueo, try/except por liga, esquema de salida fijo).
      Nombres FBref correctos: "Campeonato Brasileiro Série A" (comp 24), "Saudi Pro League"
      (comp 70). Fix: la Bundesliga venía con league=NaN (ß de "Fußball-Bundesliga" no traduce);
      se rellena en el ingest. Coef. de calidad provisionales por liga en config.LEAGUE_QUALITY.
      Modelo: `src/jugadores_model.py` (rating ofensivo/defensivo con shrinkage por minutos +
      calidad de liga; XI probable por CALIDAD por posición —incluye a Messi pese a sus minutos
      en MLS—; agregación convexa; mapeo a Poisson).
      ⚠️ SIN VALIDAR (es un prior). Limitación restante: sin xG (sobrevalora goleadores de racha).
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
- [x] **13. Rating ofensivo tipo-xG (V1)**: `rating_ofensivo = 0.7·npg_90 + 0.3·ast_90` (en
      `config`/`jugadores_model`: `W_NPG`/`W_AST_OF`). ⚠️ xG REAL no disponible en FBref vía
      soccerdata (verificado: no expone columnas Expected), así que se usa goles SIN PENAL por 90
      como "xG realizado". Honesto: NO arregla el sesgo de goleadores de racha (Sørloth sigue > Mbappé);
      para eso haría falta Understat (solo Big-5, join por nombres frágil) — documentado como futuro.
      `C_MAX` se mantiene en 1.7 (un 0.85 reintroduciría la sub-diferenciación).

### Estado: **V1 CONSOLIDADA** — el sistema corre cualquier cruce del Mundial de punta a punta.
- [ ] Mejoras pendientes (post-V1): xG real vía Understat (Big-5), derivar coeficientes de
      liga/sombra de datos reales, ensemble DC+ML, validar priors vs cuotas reales del Mundial.
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

## Fuente de datos

[football-data.co.uk](https://www.football-data.co.uk/) — gratis, trae goles, tiros,
faltas, córners, tarjetas y árbitro por partido.
