import os
import json
from pydantic import BaseModel

MODEL_PATH = "data/trained_model.json"


# FIX M1: Modelo Pydantic para mantener consistencia de tipos
class TeamRating(BaseModel):
    attack: float
    defense: float


# FIX M1: Estado global con tipo unificado y consistente
TEAMS: dict[str, TeamRating] = {}


def normalize_team_name(name: str) -> str:
    """Normaliza espacios y mayúsculas en los nombres de equipos."""
    return " ".join(name.strip().split()).title()


def get_team(name: str) -> TeamRating | None:
    """Obtiene el rating de un equipo usando su nombre normalizado."""
    norm_name = normalize_team_name(name)
    return TEAMS.get(norm_name)


def save_trained_ratings(teams_dict: dict, gamma: float, rho: float) -> None:
    """Guarda los ratings de los equipos, gamma y rho en un archivo JSON físico."""
    global TEAMS
    # FIX M1: Convertir dicts planos a TeamRating para consistencia en memoria
    TEAMS = {
        name: TeamRating(**stats) if isinstance(stats, dict) else stats
        for name, stats in teams_dict.items()
    }

    # Asegurar que el directorio data/ exista
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Guardar como dicts planos para serialización JSON
    payload = {
        "global_parameters": {"home_advantage_gamma": gamma, "rho_correction": rho},
        "teams": teams_dict,
    }

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"💾 [Persistencia] Modelo guardado exitosamente en {MODEL_PATH}")


def load_trained_model() -> dict:
    """Carga el archivo JSON si existe para inicializar los parámetros del sistema."""
    global TEAMS
    if not os.path.exists(MODEL_PATH):
        print(
            "ℹ️ [Persistencia] No se encontró un modelo previo. Usando valores vacíos/por defecto."
        )
        return {}

    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # FIX M1: Convertir dicts cargados a TeamRating para consistencia
        teams_raw = data.get("teams", {})
        TEAMS = {
            name: TeamRating(**stats) if isinstance(stats, dict) else stats
            for name, stats in teams_raw.items()
        }
        print(
            f"🎯 [Persistencia] Modelo cargado con éxito. {len(TEAMS)} equipos listos."
        )
        return data
    except Exception as e:
        print(f"❌ [Persistencia] Error al cargar el modelo: {e}. Iniciando limpio.")
        return {}
