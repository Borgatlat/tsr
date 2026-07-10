"""Shared paths and helpers used across the prediction pipeline."""

from pathlib import Path

# Project root is one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MATCHES_RAW = DATA_DIR / "results.csv"
MATCHES_FILTERED = DATA_DIR / "matches_last_10_years.csv"
FIFA_RAW = DATA_DIR / "fifa_rankings.csv"
FEATURES_PATH = DATA_DIR / "features.csv"

OUTCOME_MODEL_PATH = MODELS_DIR / "outcome_classifier.joblib"
NEUTRAL_OUTCOME_MODEL_PATH = MODELS_DIR / "neutral_outcome_classifier.joblib"
GOALS_MODEL_PATH = MODELS_DIR / "goals_poisson.joblib"
GOALS_ENCODERS_PATH = MODELS_DIR / "goals_encoders.joblib"

MATCHES_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
FIFA_URL = (
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/"
    "ranking_fifa_historical.csv"
)

# Align FIFA country names with martj42/international_results team names.
TEAM_ALIASES = {
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "Hong Kong, China": "Hong Kong",
    "Curaçao": "Curacao",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Republic of Ireland": "Ireland",
}


def normalize_team(name: str) -> str:
    """Return a canonical team name shared by match and ranking datasets."""
    if not isinstance(name, str):
        return name
    return TEAM_ALIASES.get(name.strip(), name.strip())


def ensure_dirs() -> None:
    """Create data/ and models/ if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
