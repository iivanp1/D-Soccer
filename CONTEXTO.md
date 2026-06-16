# 📕 D-SOCCER — Documento de Contexto Completo

> Documento de traspaso para retomar el proyecto en otro chat/sesión con contexto total.
> Última actualización: junio 2026 — **Fases 1A/1B/2/3 completas** (Árbitros Bayes + Player Props
> Poisson + xG Enrichment). Commit `f559818`, rama `feature/api-football`. Próximo sprint: validar
> mercados secundarios con datos StatsBomb (§11).

---

## 1. QUÉ ES D-SOCCER

Sistema personal de **predicción de fútbol para el Mundial 2026**. Predice resultados (1X2),
goles, tarjetas, faltas y córners; los compara contra las **cuotas reales** de las casas,
detecta **valor (EV)**, y **avisa por Telegram** ~30-40 min antes de cada partido. Corre
**solo en un servidor** (cron) acumulando datos para validarse.

**Objetivo del dueño (Iván)**: usarlo para apostar con criterio y, a largo plazo, retirarse.

### Filosofía (decisiones de diseño no negociables)
- Los **números** los predicen modelos estadísticos. La IA NO inventa números.
- La fuerza de una selección = **Elo histórico (columna vertebral) + forma de los jugadores
  en sus clubes (ajuste)**. No los duelos directos entre selecciones (hay pocos).
- **Validar antes de creer**: lo de clubes está validado; lo de selecciones es un *prior*.
- **Disciplina de cuotas**: "lo más seguro" ≠ rentable. Solo hay valor si `prob_modelo > prob_implícita`.
- **Honestidad brutal**: el sistema dice dónde NO confiar en él. Se juzga por proceso (EV a la
  larga), no por un resultado puntual.

---

## 2. ARQUITECTURA (flujo completo)

```
FUENTES                 MODELOS (predicen tasas)          MOTOR             SALIDA
football-data.co.uk ─▶ dixon_coles  ✅ goles (clubes)   ┐
(clubes)               estilos_model ✅ tarjetas/faltas  │
FBref/soccerdata ────▶ jugadores_model ⚠️ (Mundialista) ├▶ montecarlo ─▶ mundial_engine ─▶ mercado
(jugadores, 10 ligas)    + calibración + Elo híbrido     │   (10k sims)    (orquestador)    + cuotas
eloratings/Kaggle ───▶ elo.py / elo_history (columna     │       │
                          vertebral)                      ┘       ▼
API-Football ────────▶ fixtures.py (XI, árbitro, cuotas) ──────▶ valor.py (EV) ─▶ telegram_alert
                                                                       │
                                              autorun.py (cron) ──────▶ validacion.py (log + vs mercado)
```

---

## 3. MÓDULOS (`src/`)

| Archivo | Qué hace | Estado |
|---------|----------|--------|
| `config.py` | Ligas, temporadas, coef. calidad, ratings sombra, mapeos nación↔confederación, league_dict FBref auto-instalable, `cargar_env()` (.env) | ✅ |
| `ingest.py` | Baja ~7.166 partidos de clubes → `partidos.csv` | ✅ |
| `ingest_jugadores.py` | Baja ~12.144 jugadores, 10 ligas (FBref/soccerdata) → `jugadores.csv` | ✅ |
| `dixon_coles.py` | Modelo de goles de clubes | ✅ **Brier 0.587 (+10%)** |
| `estilos_model.py` | Tarjetas/faltas/córners + factor árbitro (observado/esperado + shrinkage) | ✅ **MAE validado** |
| `jugadores_model.py` | **Motor Mundialista**: jugadores→selección. Shrinkage por minutos, imputación en cascada, calibración, **híbrido Elo+jugadores** | ⚠️ prior |
| `montecarlo.py` | 10.000 sims minuto a minuto, Factor Caos (rojas, marcador) | ✅ |
| `mundial_engine.py` | Orquestador: jugadores+Elo+montecarlo+calibración → mercado con cuotas | ✅ |
| `calibrar_internacional.py` | Ancla goles a la media real del Mundial (1.39) → `calibracion.json` | ✅ |
| `elo.py` | Elo de selecciones de eloratings.net → `elo.json` | ✅ |
| `elo_history.py` | Computa Elo propio desde 49k partidos (Kaggle/GitHub) + **calibra mapeo Elo→goles** → `elo_propio.json` | ✅ |
| `backtest.py` / `backtest_estilos.py` | Validación walk-forward (Brier / MAE) | ✅ |
| `train_ml.py` | LightGBM benchmarkeado | ❌ perdió vs Dixon-Coles (0.612 vs 0.592) |
| `fixtures.py` | Cliente API-Football (fixtures, árbitro, alineaciones, **cuotas**) + cruce de nombres (rapidfuzz) + auto-runner | ✅ |
| `valor.py` | **Detector de valor**: EV = prob·cuota−1, rankea +EV (1X2 + O/U) | ✅ |
| `ev_calculator.py` | Función pura `calcular_ev(prob, cuota)` | ✅ |
| `telegram_alert.py` | Notificaciones push (solo `requests`, lee `.env`): **reporte completo del partido + alerta de Player Props** (formato por equipo con λ, xG_base, P(>1.5), P(SOT)) | ✅ |
| `validacion.py` | Log de predicciones forward + cuotas + resultados + Brier vs **mercado**. Log **canónico (server) vs -dev (local)** por `D_SOCCER_CANONICAL` + comando `consolidar` (dedup por `fixture_id`) | ✅ |
| `tunear_w.py` | Tunea **`w`** (Elo vs jugadores) sobre el harvest minimizando Brier + LOO. Carga el Elo (que `harvest.py` no hacía). Cero gasto de API | ✅ |
| `ingesta_historica.py` | **Cosechador StatsBomb Open Data** (gratis, sin cuota): ~314 partidos de selecciones 2018-24 con XI real + **xG real** (`.statsbomb_xg` en shot events) → SQLite `dsoccer_historico.db`. Idempotente por `match_id` | ✅ |
| `validar_statsbomb.py` | Tunea `w` sobre 83-110 partidos reales de StatsBomb (XI + outcomes). **Halló: Elo > jugadores en 1X2 (82% vs 74%)** → `w=0.85` | ✅ |
| `enriquecer_xg.py` | **Fase 3B**: corrige `npg_90` con xG real internacional (match difuso por nación) → `xg_ajuste.csv`. Factor=clamp((xG+K)/(G+K), 0.6, 1.6)^0.5. Cristiano +23%, Lukaku +26%, Musiala -19%, Gakpo -19% | ✅ |
| `arbitros_faltas.py` | **Fase 1A**: factor de faltas por árbitro con Empirical Bayes (K=8) desde StatsBomb → `arbitros_faltas.json`. MAE +10.7% vs baseline. Letexier -10%, Zwayer -12% | ✅ |
| `props_data.py` | **Fase 2/3A**: calibración offline tiros+xG → `tiros_intl.json`. Compute conv_rate_xg=0.121 (más estable que goles reales), xg_per_shot_intl por jugador, ESCALA_TIROS=0.822 | ✅ |
| `props_lineups.py` | **Fase 2**: state machine de alineaciones (UNKNOWN→PENDING→CONFIRMED→PROPS_SENT). Max 3 API calls/fixture; una vez CONFIRMED, cero calls. Cache en disco | ✅ |
| `props_model.py` | **Fase 2/3A**: Poisson usage-rate (λ_shots=λ_goles/conv_rate_xg; usage=tiros_intl/sum); agrega xG_base=λ_shots×xG_per_shot por jugador. Brier +13.9% vs baseline | ✅ |
| `validar_props.py` | **Fase 2**: compuerta Brier para Player Props. Brier 0.1675 vs baseline 0.1944 → +13.9% pasada | ✅ |
| `harvest.py` | Validación histórica (internacionales 2024 con alineaciones, ratings período-correctos) | ✅ |
| `autorun.py` | **Cron** con AUDIT TRAIL: registra Props (20-120min antes) + Registro principal (20-45min) + Telegram + actualiza resultados. Loguea CADA fixture WC con su decisión | ✅ |
| `demo.py` | Demo del modelo de clubes | ✅ |

---

## 4. EL MODELO EN DETALLE

### Goles de clubes — Dixon-Coles (`dixon_coles.py`) ✅ VALIDADO
Poisson con ataque/defensa por equipo, ventaja de local (γ), corrección `rho` para marcadores
bajos, decaimiento temporal `xi`. **Brier 0.587 vs 0.652 benchmark (+10%)**, gana en las 7 ligas.

### Tarjetas/faltas/córners — `estilos_model.py` ✅ VALIDADO
Poisson de tasas. **Factor árbitro observado/esperado** relativo a su liga (con shrinkage).
Validado por MAE: faltas +6.8%, tarjetas +2-3%. **Córners NO predice** (−0.9%, reprobó).

### Motor Mundialista — `jugadores_model.py` ⚠️ PRIOR (en validación)
Pipeline bottom-up: rating ofensivo/defensivo por jugador (`0.7·npg_90 + 0.3·ast_90`, ajustado
por calidad de liga, shrinkage por minutos), agregación convexa (estrellas pesan más),
imputación en cascada (país→confederación→default) para selecciones de "larga cola".

**HÍBRIDO con Elo** (la corrección de raíz a la sub-diferenciación):
```
λ_final = w · λ_elo + (1−w) · λ_player      (w = 0.85 DATA-DRIVEN, jun 2026)
```
- `λ_elo` calibrado con 49k partidos: **+400 de Elo = +2.09 goles de ventaja** (no el `2E` inicial).
- El Elo aplasta la sub-diferenciación (MEX-RSA 67% vs 47% bottom-up solo).

#### `w` calibrado con datos reales — la lección clave (jun 2026)

`src/validar_statsbomb.py` tuneó `w` sobre **83-110 partidos internacionales reales** (StatsBomb,
con alineación confirmada). Hallazgos, en orden:
1. **El Elo le gana al modelo de jugadores en 1X2**: acierto del favorito **82% (Elo) vs 74%
   (jugadores)**; el óptimo de Brier dio `w=1.0` (curva monótona). `w=0.65` confiaba de más en el
   bottom-up.
2. **Atacamos la raíz, no el síntoma** (3 experimentos con datos): (a) enriquecer el rating con
   **xG real** de StatsBomb (`src/enriquecer_xg.py` → `xg_ajuste.csv`; Cristiano +17%, Osimhen +18%,
   suertudos como Gakpo/Musiala −13/−15%) mejoró los ratings **individuales** pero **no** la
   diferenciación de equipos; (b) descomprimir el mapeo `^c` apenas movió el Brier; (c) la agregación
   convexa **sí** rankea bien (74% favoritos).
3. **Conclusión: es una brecha de SEÑAL, no un knob.** La forma de club predice partidos
   internacionales *decentemente* pero peor que el Elo (que codifica resultados internacionales
   reales). Ningún knob cierra esos 8 puntos — es la naturaleza del traspaso club→selección.
4. **Decisión:** `w = 0.85` (no 1.0) — el Elo manda en 1X2, pero se conserva ~15% del modelo de
   jugadores para **alineaciones rotadas** (que la muestra de torneo, con XI full, no mide). El
   modelo de jugadores + xG **no se descarta**: sigue 100% activo para los **mercados secundarios**.

**Pivote estratégico:** el edge del modelo de jugadores NO es el 1X2 (ahí gana el Elo), sino los
**mercados secundarios** (tarjetas/faltas/córners vía Montecarlo) y los desvíos por alineación.
- El modelo de jugadores aporta el ajuste por alineación real.

### Árbitros por faltas — `arbitros_faltas.py` ✅ VALIDADO (Fase 1A)

Factor de faltas por árbitro desde StatsBomb internacional (K=8 Empirical Bayes):
```
factor_shrunk = (n · factor_raw + 8 · 1.0) / (n + 8)
```
66 árbitros calibrados. Con n=5: 38% dato real, 62% prior global (no sobreajusta muestras chicas).
Gate pasado: MAE **+10.7% vs baseline**.

Separado del factor de tarjetas (`estilos_model`, datos de clubes). El árbitro aparece en
el reporte Telegram como dos líneas distintas: "Árbitro (tarjetas): ..." y "Árbitro (faltas): ...".

### Player Props — `props_model.py` ✅ VALIDADO (Fase 2 + 3A)

Modelo Poisson con usage-rate por historial internacional real:
```
conv_rate_xg   = Σ(xg_jugador) / Σ(tiros)  = 0.121   (xG más estable que goles reales)
λ_shots_team   = λ_goles_team / conv_rate_xg           (12-14 tiros/equipo; antes 16 — calibrado)
usage_i        = tiros_pp_intl_i / Σ(tiros_pp_j)       (clamp: [1%, 35%])
λ_shots_i      = λ_shots_team × usage_i
xG_base_i      = λ_shots_i × xG_per_shot_i             (calidad de posición, DISPLAY solamente)
P(tiros > k)   = 1 − Σ_{j=0}^{⌈k⌉} e^{−λ} · λ^j / j!
```

**conv_rate_xg vs conv_rate_goles**: el xG es el prior correcto (calibrado para predecir goles);
los goles reales son el resultado ruidoso. El cambio bajó la estimación de tiros de 16.5 → 12.4
por equipo/partido (el dato real de StatsBomb es 12.5). Ese es el origen de la mejora de Brier.

**xG_base**: informa al apostador sobre la calidad de posición. Messi xG/tiro=0.202 (central,
peligroso) vs Tchouameni xG/tiro=0.033 (tiros de lejos). Con λ_shots similar, el xG_base revela
quién genera peligro real. Es un display metric, no cambia P(tiros > k).

Gate pasado: Brier 0.1675 vs 0.1944 baseline → **+13.9%** (Fase 3A; Fase 2 sola fue +5.1%).

### Pipeline robusto — `autorun.py` / `fixtures.py` ✅ (Fase 1B)

Motivación: Belgium vs Egypt no fue notificado. Root cause: filtro de competición demasiado rígido.

Soluciones:
- `COMPETICIONES_WC` = set de variantes del nombre en la API (8 variantes conocidas).
- AUDIT TRAIL por corrida: cada partido WC loguea su decisión con `grep "AUDIT" autorun.log`.
- `_api_get()` con 3 reintentos, backoff exponencial, manejo de 429 (sleep 30/60/90s).
- `autorun.log` rotativo (5MB × 3 backups).
- `registrar_props()` corre ANTES que `registrar_proximos()` (ventana más amplia, 20-120min).

### Montecarlo — `montecarlo.py` ✅
Juega 10.000 partidos minuto a minuto. **Factor Caos**: rojas debilitan, el que pierde
arriesga. Agnóstico al modelo. Devuelve distribución completa de mercados.

### Calibración — `calibrar_internacional.py` / `elo_history.py`
`base_real = 1.39` goles/equipo (media real del Mundial, cross-validada con 49k partidos: 1.37).

---

## 5. ESTADO DE VALIDACIÓN

| Componente | Estado | Evidencia |
|-----------|--------|-----------|
| Goles clubes (Dixon-Coles) | ✅ **Validado** | Brier 0.587 vs 0.652 benchmark (+10%), 7 ligas |
| Tarjetas/faltas/árbitro (clubes) | ✅ **Validado** | MAE faltas +6.8%, tarjetas +2.1% |
| **Factor árbitro faltas (intl)** | ✅ **Validado** | MAE **+10.7%** vs global, LOO StatsBomb |
| Montecarlo | ✅ Coherente | Consistente con Dixon-Coles |
| Elo (columna vertebral) | ✅ Sólido | 76.6% de decididos acertados (49k partidos) |
| Motor Mundialista híbrido (w=0.85) | 🟡 **Prior en validación** | Brier LOO 0.575 (83 partidos) |
| **Player Props Poisson + xG** | ✅ **Validado** | Brier **+13.9%** vs baseline (1018 jugador-partido) |
| xG enrichment jugadores (npg_90) | ✅ Activo | Rankings ±15-25% en casos extremos; Brier global marginal |
| LightGBM | ❌ Descartado | Perdió el backtest vs Dixon-Coles (0.612 vs 0.592) |

**Honesto**: clubes = sólido. Selecciones en 1X2 = prior prometedor (Elo manda, w=0.85).
El edge del modelo de jugadores + xG está en **Player Props** (+13.9%) y en **mercados secundarios**
(faltas/tarjetas con alineación real) — esos mercados son menos eficientes que el 1X2.

**Calibración de props**: el modelo sobreestima en valores altos (~20pp). Causa: validación
usa goles reales como proxy de λ (muy volátiles). En producción con λ del engine (1.3-1.7) el
efecto es más moderado. El aviso aparece en cada mensaje de Telegram.

---

## 6. DATOS / REGISTROS (`data/`)

| Archivo | Qué es | ¿En git? |
|---------|--------|:--:|
| `processed/jugadores.csv` | 12.144 jugadores, 10 ligas, 137 países | ✅ |
| `processed/partidos.csv` | 7.166 partidos de clubes | ✅ |
| `processed/calibracion.json` | base_real 1.39, atk_ref, def_ref, compresión | ✅ |
| `processed/elo.json` / `elo_propio.json` | Elo de selecciones (eloratings / propio) | ✅ |
| `processed/predicciones_log.csv` | Log: predicción + cuota + resultado + Brier | ❌ local (server) |
| `raw/*.csv`, `raw/api_cache/`, `raw/results.csv` | Crudos + caché API + Kaggle | ❌ local |

**10 ligas de jugadores**: Premier, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Primeira,
MLS (Messi), Saudí (Cristiano), Brasileirão.

---

## 7. DESPLIEGUE (el server)

- **Servidor Debian** (homelab Proxmox, Tailscale), proyecto en `/root/D-Soccer`, rama
  `feature/api-football`, venv en `venv/`.
- **Bot de Telegram**: `@D_SoccerBot`. Credenciales en `.env` (NO en git).
- **Cron (3 tareas)** — ⚠️ **CADA línea DEBE empezar con `cd /root/D-Soccer &&`**. Sin el `cd`,
  cron corre desde `/root` y `python -m src.autorun` falla con `ModuleNotFoundError: No module named
  'src'` (cron NO se para en el dir del proyecto). Este bug tuvo el aviso de Telegram caído un tiempo.
  Instalá el crontab completo de una con `crontab - <<'EOF' ... EOF`:
  ```cron
  */15 * * * * cd /root/D-Soccer && /root/D-Soccer/venv/bin/python -m src.autorun registrar  >> /root/D-Soccer/autorun.log 2>&1
  0 * * * *    cd /root/D-Soccer && /root/D-Soccer/venv/bin/python -m src.autorun actualizar >> /root/D-Soccer/autorun.log 2>&1
  0 5 * * 0    cd /root/D-Soccer && /root/D-Soccer/venv/bin/python -m src.elo_history --refrescar >> /root/D-Soccer/elo_refresh.log 2>&1
  ```
  Verificar: `crontab -l` (las 3 con `cd`) + correr `registrar` a mano → debe imprimir
  `[autorun] ... | 0 partidos en ventana` SIN `No module named src`. El crontab vive en el sistema,
  NO en git → un `git pull` no lo toca (igual que el `.env`, que es gitignored).
- **Flujo automático**: ~30-40 min antes de cada partido del Mundial, el server predice con la
  alineación confirmada, compara con las cuotas reales, detecta valor, y manda el **reporte
  completo a Telegram**. Después del partido baja el resultado → el log crece para validar.

### Credenciales (`.env` en la raíz, gitignored)
```
API_FOOTBALL_KEY=...        # api-football.com, plan gratis (100 req/día, solo 2022-2024 por liga)
TELEGRAM_TOKEN=...          # @BotFather
TELEGRAM_CHAT_ID=...        # de getUpdates tras escribirle al bot
D_SOCCER_CANONICAL=1        # ⚠️ SOLO en el server: lo declara escritor del log canónico.
                            #    En local NO ponerla (escribe a predicciones_log_dev.csv).
```

---

## 8. CÓMO CORRER TODO

```bash
python -m pip install -r requirements.txt        # (o requirements-server.txt para el loop liviano)
# Datos:
python -m src.ingest                 # clubes (rápido)
python -m src.ingest_jugadores       # jugadores (lento, scraping FBref, necesita Chrome)
python -m src.calibrar_internacional # calibra a 1.39
python -m src.elo_history            # Elo propio + calibra mapeo Elo→goles (baja 49k de GitHub)
# Predecir:
python -m src.mundial_engine MEX RSA --arbitro "X"
python -m src.fixtures --partido <id>            # auto (baja XI + árbitro de API)
python -m src.valor <id>                         # detector de valor (vs cuotas)
# Servidor (cron):
python -m src.autorun registrar                  # registra próximos + Telegram
python -m src.autorun actualizar                 # baja resultados
python -m src.validacion reporte                 # métricas (modelo vs mercado)
python -m src.validacion consolidar [logs...]    # une logs SIN duplicar (clave fixture_id)
# Validar / pulir:
python -m src.backtest                           # goles de clubes
python -m src.harvest 4 2024 46                  # histórico (4=Euro, 9=Copa, 10=amistosos)
python -m src.tunear_w [ligas...]                # tunea w (Elo vs jugadores) sobre el harvest
python -m src.ingesta_historica --desde 2018 --export-csv   # cosecha StatsBomb -> SQLite (gratis)
```

---

## 9. DECISIONES CLAVE Y LECCIONES

1. **Sub-diferenciación** (el problema #1): el bottom-up comprime (selecciones flojas con 2-3
   cracks parecen fuertes). **Arreglado de raíz anclando a Elo** (híbrido).
2. **ML perdió**: LightGBM no le ganó a Dixon-Coles → no se reemplazó el núcleo validado.
3. **El `2E` era un invento**: se calibró con 49k partidos (+400 Elo = +2.09 goles).
4. **El valor es tan confiable como el modelo**: el detector marca "valor" en underdogs/over por
   el sesgo del modelo (en validación). Usar con criterio, no como orden.
5. **No se le gana al 1X2 del mercado** copiándolo. El edge está en mercados 2rios y decisión.
6. **La IA NO decide apuestas** (es una trampa): la decisión es EV (fórmula). La IA solo explica.

---

## 10. BUGS / LIMITACIONES CONOCIDAS

- Motor de selecciones: **validado en 1X2** con 83 partidos reales (ver §4). Ahí el **Elo gana**;
  falta validar los **mercados secundarios** (faltas/tarjetas/corners con alineación real).
- `w = 0.85` data-driven: el bottom-up no le gana al Elo en 1X2 (brecha de señal, no de modelo).
- **xG integrado en dos capas**: (a) `xg_ajuste.csv` corrige `npg_90` de jugadores para ratings de equipos;
  (b) `conv_rate_xg + xG_base` en props corrige la estimación de tiros y añade calidad de posición.
  OJO: sigue sin xG de **CLUB** (FBref no lo expone vía soccerdata).
- **Player Props calibración**: el modelo sobreestima ~20pp en valores altos. Limitación inherente
  a la muestra (pocos partidos intl por jugador). El ranking es válido; las probs son orientativas.
- **Córners** no se predice bien (backtest reprobó con -0.9% MAE).
- Ratings **sombra** = priors (selecciones de liga local sin FBref).
- Cuota API plan gratis: 2025/2026 **solo por fecha/ID** (OK para el Mundial); detalle por liga bloqueado.

---

## 11. PRÓXIMOS PASOS (en orden)

**Completado en sesiones anteriores:**
- ✅ Fase 1A: Árbitros Bayes (faltas, MAE +10.7%)
- ✅ Fase 1B: Pipeline robusto (AUDIT TRAIL, reintentos, ventanas)
- ✅ Fase 2: Player Props Poisson (Brier +13.9%)
- ✅ Fase 3A: xG en props (conv_rate_xg, xG_base, Telegram por equipo)
- ✅ Fase 3B: xG en jugadores (xg_ajuste.csv, factor npg_90)

**Próximo sprint:**
1. **(Primera tarea)** Regenerar artefactos en el server tras `git pull`:
   ```bash
   python -m src.props_data       # conv_rate_xg, xg_per_shot (135 → 314 partidos)
   python -m src.enriquecer_xg   # xg_ajuste.csv (135 → 314, ~900 jugadores)
   python -m src.arbitros_faltas  # ~80-90 árbitros vs 66 locales
   ```
2. **Validar mercados secundarios** (faltas/tarjetas) con StatsBomb: `ingesta_historica.py`
   ya captura faltas/tarjetas/corners por equipo (`equipo_partido_stats`). Falta el script
   de validación contra `estilos_model` / `disciplina_seleccion` con XI real.
3. **xA para asistencias**: re-cosechar key passes de StatsBomb (campo `key_pass_id`);
   enriquecer `ast_90` en jugadores_model con xA real.
4. **PR a main** (housekeeping): `feature/api-football` tiene 3+ commits grandes, todos probados.
5. **Acumular predicciones_log.csv** en el server durante el Mundial → `python -m src.validacion reporte`
   para medir el edge real vs mercado.

---

## 12. REPO / GIT

- GitHub **privado**: `https://github.com/iivanp1/D-Soccer`
- `main` = V1 consolidada (commit inicial). `feature/api-football` = todo lo nuevo (Elo, Telegram,
  valor, server). **PR a main pendiente** (housekeeping).
- Identidad git: iivanp1. `gh` CLI no instalado (push manual).
- ⚠️ El repo local está en OneDrive (posible conflicto de sync con `.git`).

---

**En una frase**: D-Soccer es un sistema completo y autónomo que predice cada partido del Mundial
(híbrido Elo + jugadores, calibrado), lo compara con el mercado, detecta valor, avisa por Telegram
~30 min antes, y se valida solo en un server — sabiendo con honestidad dónde confiar y dónde no.
**Hoy sabemos, con datos reales, que en 1X2 manda el Elo (`w=0.85`); el edge del modelo de jugadores
+ xG está en los mercados secundarios — validarlos es el próximo sprint.**
