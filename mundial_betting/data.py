from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamRating:
    attack: float
    defense: float
    flag: str
    host: bool = False


TEAMS: dict[str, TeamRating] = {
    "Alemania": TeamRating(1.58, 0.80, "DE"),
    "Arabia Saudita": TeamRating(1.02, 1.12, "SA"),
    "Argelia": TeamRating(1.10, 1.08, "DZ"),
    "Argentina": TeamRating(1.82, 0.65, "AR"),
    "Australia": TeamRating(1.15, 1.04, "AU"),
    "Austria": TeamRating(1.30, 0.94, "AT"),
    "Bosnia y Herzegovina": TeamRating(1.08, 1.08, "BA"),
    "Brasil": TeamRating(1.68, 0.74, "BR"),
    "Belgica": TeamRating(1.52, 0.84, "BE"),
    "Cabo Verde": TeamRating(0.88, 1.18, "CV"),
    "Canada": TeamRating(1.22, 1.02, "CA", True),
    "Colombia": TeamRating(1.55, 0.82, "CO"),
    "Congo DR": TeamRating(0.92, 1.14, "CD"),
    "Corea del Sur": TeamRating(1.20, 1.00, "KR"),
    "Costa de Marfil": TeamRating(1.14, 1.04, "CI"),
    "Croacia": TeamRating(1.42, 0.88, "HR"),
    "Curacao": TeamRating(0.85, 1.22, "CW"),
    "Ecuador": TeamRating(1.28, 0.96, "EC"),
    "Egipto": TeamRating(1.06, 1.10, "EG"),
    "Escocia": TeamRating(1.18, 1.02, "SCO"),
    "Espana": TeamRating(1.80, 0.66, "ES"),
    "Estados Unidos": TeamRating(1.28, 0.98, "US", True),
    "Francia": TeamRating(1.78, 0.68, "FR"),
    "Ghana": TeamRating(1.12, 1.06, "GH"),
    "Haiti": TeamRating(0.88, 1.20, "HT"),
    "Inglaterra": TeamRating(1.72, 0.72, "ENG"),
    "Irak": TeamRating(0.90, 1.16, "IQ"),
    "Iran": TeamRating(1.08, 1.08, "IR"),
    "Japon": TeamRating(1.35, 0.92, "JP"),
    "Jordania": TeamRating(0.90, 1.18, "JO"),
    "Marruecos": TeamRating(1.30, 0.94, "MA"),
    "Mexico": TeamRating(1.32, 0.95, "MX", True),
    "Noruega": TeamRating(1.38, 0.90, "NO"),
    "Nueva Zelanda": TeamRating(0.85, 1.22, "NZ"),
    "Panama": TeamRating(0.95, 1.15, "PA"),
    "Paraguay": TeamRating(1.12, 1.04, "PY"),
    "Paises Bajos": TeamRating(1.56, 0.82, "NL"),
    "Portugal": TeamRating(1.62, 0.78, "PT"),
    "Qatar": TeamRating(0.98, 1.15, "QA"),
    "Republica Checa": TeamRating(1.24, 0.98, "CZ"),
    "Senegal": TeamRating(1.28, 0.95, "SN"),
    "Sudafrica": TeamRating(1.00, 1.12, "ZA"),
    "Suecia": TeamRating(1.22, 0.98, "SE"),
    "Suiza": TeamRating(1.32, 0.93, "CH"),
    "Tunez": TeamRating(1.05, 1.10, "TN"),
    "Turquia": TeamRating(1.26, 0.96, "TR"),
    "Uruguay": TeamRating(1.48, 0.86, "UY"),
    "Uzbekistan": TeamRating(0.92, 1.16, "UZ"),
}

ALIASES = {
    "Bélgica": "Belgica",
    "Canadá": "Canada",
    "Curaçao": "Curacao",
    "España": "Espana",
    "Haití": "Haiti",
    "Irán": "Iran",
    "Japón": "Japon",
    "México": "Mexico",
    "Panamá": "Panama",
    "Países Bajos": "Paises Bajos",
    "República Checa": "Republica Checa",
    "Sudáfrica": "Sudafrica",
    "Túnez": "Tunez",
    "Turquía": "Turquia",
    "Uzbekistán": "Uzbekistan",
}

DISPLAY_NAMES: dict[str, str] = {canonical: accented for accented, canonical in ALIASES.items()}


def normalize_team_name(name: str) -> str:
    cleaned = " ".join(name.strip().split())
    return ALIASES.get(cleaned, cleaned)


def get_display_name(name: str) -> str:
    normalized = normalize_team_name(name)
    return DISPLAY_NAMES.get(normalized, normalized)


def get_team(name: str) -> TeamRating:
    normalized = normalize_team_name(name)
    try:
        return TEAMS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown team: {name}") from exc
