"""Configuracion central del proyecto.

Fuente de datos: football-data.co.uk
Cada CSV trae goles, tiros, faltas, tarjetas, corners y el arbitro por partido.
URL: https://www.football-data.co.uk/mmz4281/{temporada}/{liga}.csv
  temporada: '2324' = 2023-24, '2425' = 2024-25, etc.
  liga: codigo de division (ver LIGAS abajo)
"""

from pathlib import Path

# --- Rutas ---
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"

# --- Ligas a descargar (codigos de football-data.co.uk) ---
# Mas ligas = mas partidos = mejor calibracion del modelo.
LIGAS = {
    "E0": "Premier League (Inglaterra)",
    "SP1": "La Liga (Espana)",
    "I1": "Serie A (Italia)",
    "D1": "Bundesliga (Alemania)",
    "F1": "Ligue 1 (Francia)",
    "N1": "Eredivisie (Holanda)",
    "P1": "Primeira Liga (Portugal)",
}

# --- Temporadas a descargar ---
# Mas temporadas = mas historia, pero la forma reciente pesa mas (ver decaimiento
# temporal en dixon_coles.py). 3-4 temporadas suele ser un buen balance.
# 2526 agregada (jul 2026) para cruzar tiros/corners con el xG de Understat (clubes 2026-27).
TEMPORADAS = ["2223", "2324", "2425", "2526"]

# --- Columnas que nos interesan y su nombre limpio ---
# (las dejamos en ingles corto porque son estandar en football-data.co.uk)
COLUMNAS = {
    "Date": "fecha",
    "HomeTeam": "local",
    "AwayTeam": "visitante",
    "FTHG": "goles_local",
    "FTAG": "goles_visitante",
    "FTR": "resultado",        # H=local, D=empate, A=visitante
    "HS": "tiros_local",
    "AS": "tiros_visitante",
    "HST": "tiros_arco_local",
    "AST": "tiros_arco_visitante",
    "HF": "faltas_local",
    "AF": "faltas_visitante",
    "HC": "corners_local",
    "AC": "corners_visitante",
    "HY": "amarillas_local",
    "AY": "amarillas_visitante",
    "HR": "rojas_local",
    "AR": "rojas_visitante",
    "Referee": "arbitro",
}

URL_BASE = "https://www.football-data.co.uk/mmz4281"


def cargar_env() -> None:
    """Carga variables de un archivo .env en la raiz hacia os.environ.

    No pisa las que ya esten seteadas (asi el cron puede traer la API key inline y el
    .env aporta el resto: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, etc.). Sin dependencias.
    """
    import os
    env = RAIZ / ".env"
    if not env.exists():
        return
    for linea in env.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))

# =========================================================================== #
#  DATOS A NIVEL JUGADOR (FBref via soccerdata) -- para el Motor Mundialista
# =========================================================================== #
# Usamos la liga combinada "Big 5" de FBref: trae las 5 grandes ligas en pocas
# requests (mas eficiente y menos riesgo de bloqueo que pedirlas una por una).
LIGAS_FBREF = "Big 5 European Leagues Combined"
TEMPORADAS_FBREF = ["2324", "2425"]

# Ligas adicionales (no-Big5) para cubrir jugadores fuera de Europa: Messi (MLS),
# Cristiano (Saudi), joyas del Brasileirao, etc. OJO: cada liga usa SU formato de
# temporada. Las de ano calendario (MLS, Brasileirao) van como "2024"/"2025"; las
# cruzadas (Ago-May) como "2324"/"2425". Requieren el league_dict.json custom de
# soccerdata (~/soccerdata/config/league_dict.json) que mapea estas ligas a FBref.
LIGAS_FBREF_EXTRA = {
    "USA-Major League Soccer":   ["2024", "2025"],   # ano calendario (Feb-Dic)
    "BRA-Campeonato Brasileiro": ["2024", "2025"],   # ano calendario (Abr-Dic)
    "KSA-Professional League":   ["2324", "2425"],   # cruzada Ago-May
    "NED-Eredivisie":            ["2324", "2425"],   # cruzada Ago-May
    "POR-Primeira Liga":         ["2324", "2425"],   # cruzada Ago-May
}

# Mapeo que soccerdata necesita para reconocer estas ligas en FBref (nombres EXACTOS
# de fbref.com/en/comps/). Se auto-instala en ~/soccerdata/config/league_dict.json al
# correr la ingesta, asi el proyecto es reproducible al clonarlo (cero config manual).
FBREF_LEAGUE_DICT = {
    "USA-Major League Soccer":   {"FBref": "Major League Soccer", "season_code": "single-year"},
    "BRA-Campeonato Brasileiro": {"FBref": "Campeonato Brasileiro Série A", "season_code": "single-year"},
    "KSA-Professional League":   {"FBref": "Saudi Pro League", "season_start": "Aug", "season_end": "May"},
    "NED-Eredivisie":            {"FBref": "Eredivisie", "season_start": "Aug", "season_end": "May"},
    "POR-Primeira Liga":         {"FBref": "Primeira Liga", "season_start": "Aug", "season_end": "May"},
}


def instalar_league_dict_fbref() -> None:
    """Escribe/actualiza ~/soccerdata/config/league_dict.json con FBREF_LEAGUE_DICT.

    Idempotente y no destructivo: fusiona sin pisar ligas que el usuario haya agregado,
    pero garantiza que las nuestras (con los nombres FBref correctos) esten presentes.
    Llamada automaticamente por ingest_jugadores -> el repo funciona al clonarlo.
    """
    import json
    from pathlib import Path

    cfg_dir = Path.home() / "soccerdata" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    destino = cfg_dir / "league_dict.json"

    actual = {}
    if destino.exists():
        try:
            actual = json.loads(destino.read_text(encoding="utf-8"))
        except Exception:
            actual = {}

    fusion = {**actual, **FBREF_LEAGUE_DICT}  # las nuestras prevalecen (nombres correctos)
    if fusion != actual:
        destino.write_text(json.dumps(fusion, ensure_ascii=False, indent=2), encoding="utf-8")

# Indice de calidad de liga (League Quality Index). PROVISIONAL y subjetivo:
# lo correcto es CALIBRARLO con resultados entre ligas (Champions/Europa) o valores
# de mercado. Por ahora sirve para normalizar el rendimiento segun el nivel rival.
# Mismo nombre de liga que devuelve FBref en la columna 'league'.
LEAGUE_QUALITY = {
    "ENG-Premier League": 1.00,
    "ESP-La Liga": 0.95,
    "ITA-Serie A": 0.92,
    "GER-Bundesliga": 0.92,
    "FRA-Ligue 1": 0.88,
    # Ligas no-Big5 (PROVISIONAL, subjetivo: idealmente calibrar con Champions/Libertadores).
    # Importan para no sobrevalorar a Messi (MLS) o Cristiano (Saudi).
    "NED-Eredivisie": 0.78,
    "POR-Primeira Liga": 0.75,
    "BRA-Campeonato Brasileiro": 0.72,
    "USA-Major League Soccer": 0.65,
    "KSA-Professional League": 0.60,
}
QUALITY_DEFAULT = 0.70  # ligas aun no mapeadas

# =========================================================================== #
#  RATINGS SOMBRA (Tiered Imputation) -- para el problema de la "Larga Cola"
# =========================================================================== #
# Selecciones como NZ, Paraguay o muchas de AFC/CAF tienen jugadores en ligas
# locales que NO scrapeamos. En vez de fallar o devolver 0, imputamos un rendimiento
# basal segun el nivel de su federacion. MISMA escala que el modelo (1.0 = jugador
# promedio del dataset, que es de liga top; un local rinde por debajo).
# PRIORS puros (no calibrados): hacen el motor robusto, no precisos para esas selecciones.
CONFED_BASELINE = {           # (ofensivo_base, defensivo_base)
    "UEFA":     (0.78, 0.82),  # ligas europeas chicas
    "CONMEBOL": (0.82, 0.88),  # Sudamerica (defensivamente solidos)
    "CONCACAF": (0.55, 0.58),
    "AFC":      (0.55, 0.55),  # Asia
    "CAF":      (0.62, 0.62),  # Africa
    "OFC":      (0.40, 0.40),  # Oceania
}
CONFED_DEFAULT = (0.55, 0.55)

# Overrides por pais (encima de la confederacion). Incluye los ejemplos del diseno.
NACION_SHADOW = {
    "PAR": (0.85, 0.90),   # Paraguay
    "JPN": (0.75, 0.70),   # Japon (top AFC)
    "NZL": (0.40, 0.40),   # Nueva Zelanda
}

# Nacion (codigo FBref) -> confederacion. Cubre las selecciones probables del Mundial.
NACION_CONFED = {
    # CONMEBOL
    "ARG": "CONMEBOL", "BRA": "CONMEBOL", "URU": "CONMEBOL", "PAR": "CONMEBOL",
    "CHI": "CONMEBOL", "COL": "CONMEBOL", "ECU": "CONMEBOL", "PER": "CONMEBOL",
    "BOL": "CONMEBOL", "VEN": "CONMEBOL",
    # UEFA
    "FRA": "UEFA", "GER": "UEFA", "ESP": "UEFA", "ITA": "UEFA", "ENG": "UEFA",
    "POR": "UEFA", "NED": "UEFA", "BEL": "UEFA", "CRO": "UEFA", "SUI": "UEFA",
    "DEN": "UEFA", "POL": "UEFA", "SRB": "UEFA", "WAL": "UEFA", "SCO": "UEFA",
    "AUT": "UEFA", "UKR": "UEFA", "SWE": "UEFA", "NOR": "UEFA", "TUR": "UEFA",
    "CZE": "UEFA", "HUN": "UEFA", "ALB": "UEFA", "SVN": "UEFA", "SVK": "UEFA",
    "ROU": "UEFA", "GEO": "UEFA", "BIH": "UEFA",
    # AFC
    "JPN": "AFC", "KOR": "AFC", "KSA": "AFC", "IRN": "AFC", "AUS": "AFC",
    "QAT": "AFC", "IRQ": "AFC", "UAE": "AFC", "UZB": "AFC", "JOR": "AFC",
    # CONCACAF
    "USA": "CONCACAF", "MEX": "CONCACAF", "CAN": "CONCACAF", "CRC": "CONCACAF",
    "PAN": "CONCACAF", "JAM": "CONCACAF", "HON": "CONCACAF", "CUW": "CONCACAF",
    "HAI": "CONCACAF",
    # CAF
    "MAR": "CAF", "SEN": "CAF", "NGA": "CAF", "EGY": "CAF", "GHA": "CAF",
    "CMR": "CAF", "CIV": "CAF", "TUN": "CAF", "ALG": "CAF", "MLI": "CAF",
    "RSA": "CAF", "CPV": "CAF", "COD": "CAF",
    # OFC
    "NZL": "OFC",
}

# Disciplina sombra (faltas/tarjetas por 90 de un jugador promedio). Defaults sensatos:
# 11 x 1.1 = ~12 faltas/equipo, 11 x 0.18 = ~2 tarjetas/equipo.
SHADOW_FALTAS_90 = 1.10
SHADOW_TARJETAS_90 = 0.18

# Calibracion club->internacional de FALTAS (DATA-DRIVEN, src/validar_estilos_statsbomb.py):
# las selecciones cometen mas faltas que el promedio de club. Sobre 244 equipos-partido de
# StatsBomb, el modelo predecia 11.9 y la realidad internacional era 14.4 -> factor ~1.21.
# Recalibrado, el modelo le gana al baseline en faltas (MAE 3.50 vs 3.67, +4.8%). Se aplica en
# disciplina_seleccion. Las tarjetas NO se escalan (su media ya coincidia; ademas no hay senal).
ESCALA_FALTAS_SELECCION = 1.21

# =========================================================================== #
#  ANCLA DE PINNACLE (ingenieria inversa de lambda desde el mercado sharp)
# =========================================================================== #
# Peso del lambda implicito de Pinnacle al mezclarlo con el lambda del modelo:
#   lam_final = ALPHA * lam_pinnacle + (1 - ALPHA) * lam_modelo
# 0.35 = el mercado corrige ~1/3 del lambda del modelo (reduce el error de calibracion de
# goles) conservando 2/3 de senal propia (preserva el edge potencial; no copia al mercado).
# TUNEABLE: subir acerca al mercado (menos edge, menos varianza); bajar confia mas en el
# modelo. Wilkens 2026 (Bundesliga, 11 temporadas) encontro el optimo ~0.40 en ROI.
ALPHA_ANCLA_PINNACLE = 0.35

# =========================================================================== #
#  CORRECCIONES DE GOLES PARA SELECCIONES (Montecarlo)
# =========================================================================== #
# RETIRADO (jul 2026): el ZIP hard-cap y el rho de Dixon-Coles calibraban los AGREGADOS
# (over 42%, empates 34%) pero con la COMPOSICION invertida: metian la masa de empates en el
# 0-0 (18.9% modelo vs 12.1% real, y 8-11% en Mundiales) cuando la realidad la pone en el 1-1
# (17.4% real vs 10.5% modelo). Ademas el ZIP DOBLE-corregia en modo anclado (pisoteaba el
# total que ya traia el mercado). Reemplazados por FRAC3 (bivariate Poisson) + GAMMA (abajo).
# Los parametros pi_zip/rho siguen existiendo en montecarlo (default 0) para experimentar.
PI_ZIP_SELECCIONES = 0.0
RHO_DIXON_COLES = 0.0

# BIVARIATE POISSON (Karlis-Ntzoufras 2003): goles correlacionados via componente comun.
#   X = W1 + W3, Y = W2 + W3, con lambda3 = FRAC3 * min(lam_l, lam_v)  (marginales intactas).
# Es el modelo profesional estandar para el exceso de empates: sube la DIAGONAL (1-1, 2-2)
# -no solo el 0-0 como el ZIP- y baja el over via la correlacion positiva. CALIBRACION:
# el grid del Brier es PLANO (0.20-0.55 empatan dentro del ruido, n=132), asi que decide la
# calibracion por-partido: 0.20 da 0-0 ~10%, 1-1 ~13%, empate ~30%, over ~44% (rangos reales:
# WC22 10.9/7.8/23.4/47, muestra 2024 12.1/17.4/34.8/42) SIN inflar los lambda de la inversion
# del ancla (frac3 alto obliga a lambdas absurdos para reproducir el empate del mercado, que
# es internamente consistente con correlacion BAJA). El Factor Caos del Montecarlo ya agrega
# ~+2pp de empates encima (el que pierde arriesga), parte de la correlacion real.
FRAC3_GOLES_COMUNES = 0.20

# Deflactor de goles del MODELO PURO (sin ancla de mercado) para selecciones: el bottom-up
# genera ~2.85 goles/partido vs 2.69 real WC22 / ~2.45 torneos 2024. gamma=0.92 es el punto
# medio multi-torneo (la muestra 2024 pedia 0.88 pero un Mundial es mas ofensivo que
# Euro+AFCON). NO se aplica al lambda anclado (el mercado ya trae el total correcto).
GAMMA_GOLES_SELECCIONES = 0.92

# Escala de faltas especifica del MUNDIAL (WC 2022 real: 13.9 faltas/equipo sobre 64 partidos
# vs 11.9 del modelo pre-escala -> 1.17). La global 1.21 quedo calibrada al mix de torneos
# 2023-24 (AFCON/Copa America inflan; Euro baja). mundial_engine usa ESTA para el WC.
ESCALA_FALTAS_WC = 1.17

# =========================================================================== #
#  CLUBES 2026-27: ingesta de xG/xPts desde Understat (src/ingest_xg.py)
# =========================================================================== #
# Codigos de liga de soccerdata para Understat (las 4 grandes del plan de clubes).
# Understat NO cubre Champions (esa ira via FBref read_schedule cuando arranque).
LIGAS_UNDERSTAT = ["ESP-La Liga", "ENG-Premier League", "GER-Bundesliga", "ITA-Serie A"]
# 2526 = temporada pasada (base de analisis); 2627 = la nueva (vacia hasta ~15 de agosto,
# la ingesta la saltea con gracia y el cron semanal del server la ira llenando).
TEMPORADAS_XG = ["2526", "2627"]
