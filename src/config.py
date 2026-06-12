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
TEMPORADAS = ["2223", "2324", "2425"]

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
    "ROU": "UEFA", "GEO": "UEFA",
    # AFC
    "JPN": "AFC", "KOR": "AFC", "KSA": "AFC", "IRN": "AFC", "AUS": "AFC",
    "QAT": "AFC", "IRQ": "AFC", "UAE": "AFC", "UZB": "AFC",
    # CONCACAF
    "USA": "CONCACAF", "MEX": "CONCACAF", "CAN": "CONCACAF", "CRC": "CONCACAF",
    "PAN": "CONCACAF", "JAM": "CONCACAF", "HON": "CONCACAF",
    # CAF
    "MAR": "CAF", "SEN": "CAF", "NGA": "CAF", "EGY": "CAF", "GHA": "CAF",
    "CMR": "CAF", "CIV": "CAF", "TUN": "CAF", "ALG": "CAF", "MLI": "CAF",
    "RSA": "CAF",
    # OFC
    "NZL": "OFC",
}

# Disciplina sombra (faltas/tarjetas por 90 de un jugador promedio). Defaults sensatos:
# 11 x 1.1 = ~12 faltas/equipo, 11 x 0.18 = ~2 tarjetas/equipo.
SHADOW_FALTAS_90 = 1.10
SHADOW_TARJETAS_90 = 0.18
