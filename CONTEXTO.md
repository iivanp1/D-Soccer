# 📕 D-SOCCER — Documento de Contexto Completo

> Documento de traspaso para retomar el proyecto en otro chat/sesión con contexto total.
> Última actualización: junio 2026 (commit `4dfd8b4`, rama `feature/api-football`).

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
| `telegram_alert.py` | Notificaciones push (solo `requests`, lee `.env`): reporte completo del partido | ✅ |
| `validacion.py` | Log de predicciones forward + cuotas + resultados + Brier vs **mercado**. Log **canónico (server) vs -dev (local)** por `D_SOCCER_CANONICAL` + comando `consolidar` (dedup por `fixture_id`) | ✅ |
| `tunear_w.py` | Tunea **`w`** (Elo vs jugadores) sobre el harvest minimizando Brier + LOO. Carga el Elo (que `harvest.py` no hacía). Cero gasto de API | ✅ |
| `ingesta_historica.py` | **Cosechador StatsBomb Open Data** (gratis, sin cuota): ~314 partidos de selecciones 2018-24 con XI real + **xG real** → SQLite `dsoccer_historico.db`. Aislado, idempotente por `match_id` | ✅ |
| `harvest.py` | Validación histórica (internacionales 2024 con alineaciones, ratings período-correctos) | ✅ |
| `autorun.py` | **Punto de entrada del cron**: registra partidos 20-45 min antes (alineación real) + Telegram + actualiza resultados | ✅ |
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
λ_final = w · λ_elo + (1−w) · λ_player      (w = 0.65 PROVISIONAL)
```
- `λ_elo` calibrado con 49k partidos: **+400 de Elo = +2.09 goles de ventaja** (no el `2E` inicial).
- El Elo aplasta la sub-diferenciación (MEX-RSA 67% vs 47% bottom-up solo).
- El modelo de jugadores aporta el ajuste por alineación real.

### Montecarlo — `montecarlo.py` ✅
Juega 10.000 partidos minuto a minuto. **Factor Caos**: rojas debilitan, el que pierde
arriesga. Agnóstico al modelo. Devuelve distribución completa de mercados.

### Calibración — `calibrar_internacional.py` / `elo_history.py`
`base_real = 1.39` goles/equipo (media real del Mundial, cross-validada con 49k partidos: 1.37).

---

## 5. ESTADO DE VALIDACIÓN

| Componente | Estado | Evidencia |
|-----------|--------|-----------|
| Goles clubes (Dixon-Coles) | ✅ **Validado** | Brier 0.587, miles de partidos |
| Tarjetas/faltas/árbitro | ✅ **Validado** | MAE |
| Montecarlo | ✅ Coherente | Consistente con Dixon-Coles |
| Elo (columna vertebral) | ✅ Sólido | Acierta 76.6% de decididos (49k partidos) |
| **Motor Mundialista (selecciones)** | 🟡 **Prior, en validación** | Híbrido Elo+jugadores ~= mercado en 1X2 |
| LightGBM | ❌ Descartado | Perdió el backtest |

**Honesto**: clubes = sólido. Selecciones = prior prometedor (con Elo diferencia bien) pero
**falta validar con outcomes** (el server los acumula). El edge real NO es ganarle al 1X2 del
mercado, sino mercados menos eficientes / soporte de decisión.

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
- **Cron (3 tareas)**:
  ```
  */15 * * * *  python -m src.autorun registrar    # cada 15 min: registra 20-45 min antes + Telegram
  0 * * * *     python -m src.autorun actualizar   # cada hora: baja resultados
  0 5 * * 0     python -m src.elo_history --refrescar  # domingos: recalibra el Elo
  ```
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

- Motor de selecciones **sin validar con outcomes** (el server los junta).
- `w = 0.65` es **provisional** (falta tunear con datos del log).
- Sin **xG real** (FBref no lo expone) → sobrevalora goleadores de racha.
- **Córners** no se predice bien (lo dice el backtest).
- Ratings **sombra** = priors (selecciones de liga local).
- Cuota API plan gratis: **2025/2026 bloqueado** por liga/temporada (pero fecha/id de 2026 sí anda).

---

## 11. PRÓXIMOS PASOS (en orden)

1. **Dejar acumular** el `predicciones_log.csv` en el server (~15-30 partidos).
2. **`python -m src.validacion reporte`** → ¿el modelo le gana al mercado? (Brier modelo vs mercado).
3. **Tunear `w`** (Elo vs jugadores) con los outcomes reales → el último prior por convertir en dato.
4. Refinamientos: **xG** (Understat/Sofascore) para goleadores de racha; **capa IA explicativa**
   (asesora, no decide); derivar coeficientes de liga/sombra de datos.

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
**Lo que falta es acumular datos y tunear `w` para saber si tiene edge real.**
