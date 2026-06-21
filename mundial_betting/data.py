import os
import json
from pydantic import BaseModel

MODEL_PATH = "data/trained_model.json"


# Modelo Pydantic para mantener consistencia de tipos
class TeamRating(BaseModel):
    attack: float
    defense: float


# Estado global en memoria para los equipos
TEAMS: dict[str, dict[str, float]] = {}


def normalize_team_name(name: str) -> str:
    """Normaliza espacios y mayúsculas en los nombres de equipos."""
    return " ".join(name.strip().split()).title()


def get_team(name: str) -> dict[str, float] | None:
    """Obtiene el rating de un equipo usando su nombre normalizado."""
    norm_name = normalize_team_name(name)
    return TEAMS.get(norm_name)


def save_trained_ratings(teams_dict: dict, gamma: float, rho: float) -> None:
    """Guarda los ratings de los equipos, gamma y rho en un archivo JSON físico."""
    global TEAMS
    TEAMS = teams_dict  # Actualiza la memoria RAM de inmediato

    # Asegurar que el directorio data/ exista
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

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

        TEAMS = data.get("teams", {})
        print(
            f"🎯 [Persistencia] Modelo cargado con éxito. {len(TEAMS)} equipos listos."
        )
        return data
    except Exception as e:
        print(f"❌ [Persistencia] Error al cargar el modelo: {e}. Iniciando limpio.")
        return {}
