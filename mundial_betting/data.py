import os
import json
from pydantic import BaseModel

MODEL_PATH = "data/trained_model.json"


# =============================================================================
# CONSTANTS - Fuente única de verdad para nombres de equipos y aliases
# =============================================================================

FIFA_TEAMS = {
    "Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola",
    "Anguilla", "Antigua And Barbuda", "Argentina", "Armenia", "Aruba", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados",
    "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia",
    "Bosnia And Herzegovina", "Botswana", "Brazil", "British Virgin Islands",
    "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon",
    "Canada", "Cape Verde", "Cayman Islands", "Central African Republic", "Chad",
    "Chile", "China", "Colombia", "Comoros", "Congo", "Cook Islands",
    "Costa Rica", "Croatia", "Cuba", "Curacao", "Cyprus", "Czech Republic",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Dr Congo",
    "Ecuador", "Egypt", "El Salvador", "England", "Equatorial Guinea",
    "Estonia", "Eswatini", "Ethiopia", "Faroe Islands", "Fiji",
    "Finland", "France", "French Guiana", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Gibraltar", "Greece", "Grenada", "Guadeloupe",
    "Guam", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
    "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia",
    "Iran", "Iraq", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan",
    "Jordan", "Kazakhstan", "Kenya", "Kosovo", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Martinique",
    "Mauritania", "Mauritius", "Mexico", "Moldova", "Mongolia", "Montenegro",
    "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal",
    "Netherlands", "New Caledonia", "New Zealand", "Nicaragua", "Niger",
    "Nigeria", "North Korea", "North Macedonia", "Northern Ireland",
    "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palestine",
    "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Puerto Rico", "Qatar", "Republic Of Ireland", "Romania",
    "Russia", "Rwanda", "Saint Kitts And Nevis", "Saint Lucia", "Saint Martin",
    "Saint Vincent And The Grenadines", "Samoa", "San Marino", "Saudi Arabia",
    "Scotland", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore",
    "Sint Maarten", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Tahiti", "Taiwan",
    "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad And Tobago", "Tunisia", "Turkey", "Turkmenistan",
    "Turks And Caicos Islands", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United States", "United States Virgin Islands",
    "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Wales",
    "Yemen", "Zambia", "Zimbabwe",
}

NAME_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Republic of Ireland": "Republic Of Ireland",
    "Trinidad and Tobago": "Trinidad And Tobago",
    "Bosnia and Herzegovina": "Bosnia And Herzegovina",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Cote d'Ivoire": "Ivory Coast",
    "Sao Tome and Principe": "Sao Tome And Principe",
    "Antigua and Barbuda": "Antigua And Barbuda",
    "Saint Kitts and Nevis": "Saint Kitts And Nevis",
    "Saint Vincent and the Grenadines": "Saint Vincent And The Grenadines",
    "Turks and Caicos Islands": "Turks And Caicos Islands",
    "Northern Mariana Islands": "Northern Mariana Islands",
    "American Samoa": "American Samoa",
    "Cook Islands": "Cook Islands",
    "New Caledonia": "New Caledonia",
    "Curacao": "Curacao",
}


class TeamRating(BaseModel):
    attack: float
    defense: float


# Estado global — nunca reasignar esta variable, solo mutarla
TEAMS: dict[str, TeamRating] = {}


def normalize_team_name(name: str) -> str:
    """Normaliza espacios y mayúsculas en los nombres de equipos."""
    name = name.strip()
    # Aplicar aliases primero
    if name in NAME_ALIASES:
        name = NAME_ALIASES[name]
    return " ".join(name.split()).title()


def get_team(name: str) -> TeamRating | None:
    """Obtiene el rating de un equipo usando su nombre normalizado."""
    norm_name = normalize_team_name(name)
    return TEAMS.get(norm_name)


def save_trained_ratings(teams_dict: dict, gamma: float, rho: float) -> None:
    """Guarda los ratings de los equipos, gamma y rho en un archivo JSON físico."""
    # FIX M1: Mutar TEAMS en lugar de reasignar para que todas las importaciones vean el cambio
    TEAMS.clear()
    for name, stats in teams_dict.items():
        TEAMS[name] = TeamRating(**stats) if isinstance(stats, dict) else stats

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
    if not os.path.exists(MODEL_PATH):
        print(
            "ℹ️ [Persistencia] No se encontró un modelo previo. Usando valores vacíos/por defecto."
        )
        return {}

    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # FIX M1: Mutar TEAMS en lugar de reasignar
        teams_raw = data.get("teams", {})
        TEAMS.clear()
        for name, stats in teams_raw.items():
            TEAMS[name] = TeamRating(**stats) if isinstance(stats, dict) else stats

        print(
            f"🎯 [Persistencia] Modelo cargado con éxito. {len(TEAMS)} equipos listos."
        )
        return data
    except Exception as e:
        print(f"❌ [Persistencia] Error al cargar el modelo: {e}. Iniciando limpio.")
        return {}
