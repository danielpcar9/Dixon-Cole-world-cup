#!/usr/bin/env python3
"""
train_local.py
==============
Entrena el modelo Dixon-Coles localmente sin pasar por la API HTTP.
Mucho mas rapido para datasets grandes.
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from io import StringIO

import pandas as pd
import requests

# Importar directamente desde tu paquete
from mundial_betting.dixon_coles import train_ratings
from mundial_betting.data import save_trained_ratings
from mundial_betting.models import MatchData


def load_previous_ratings():
    """Carga ratings previos para warm-start desde data/trained_model.json."""
    path = "data/trained_model.json"
    if not os.path.exists(path):
        print("No se encontro trained_model.json, entrenando desde cero...")
        return None
    
    with open(path) as f:
        data = json.load(f)
    
    # Formato que espera train_ratings
    ratings = dict(data.get("teams", {}))
    gp = data.get("global_parameters", {})
    ratings["__gamma__"] = gp.get("home_advantage_gamma", 1.15)
    ratings["__rho__"] = gp.get("rho_correction", -0.10)
    print(f"Warm-start cargado desde {path} con {len(ratings) - 2} equipos")
    return ratings

# =============================================================================
# CONFIGURACION
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

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 3.0,
    "FIFA World Cup qualification": 2.5,
    "FIFA World Cup Qualification": 2.5,
    "Copa America": 2.0,
    "Copa América": 2.0,
    "UEFA Euro": 2.0,
    "UEFA Euro qualification": 1.8,
    "AFC Asian Cup": 1.5,
    "AFC Asian Cup qualification": 1.3,
    "Africa Cup of Nations": 1.5,
    "Africa Cup of Nations qualification": 1.3,
    "CONCACAF Gold Cup": 1.5,
    "CONCACAF Nations League": 1.4,
    "Oceania Nations Cup": 1.2,
    "UEFA Nations League": 1.3,
    "FIFA Confederations Cup": 1.8,
    "Friendly": 0.3,
    "Kirin Cup": 0.3,
    "China Cup": 0.3,
    "King's Cup": 0.3,
    "Jordan International Tournament": 0.3,
    "Intercontinental Cup": 0.4,
    "Olympics": 0.2,
}

DEFAULT_WEIGHT = 0.5


def normalize_name(name: str) -> str:
    name = name.strip()
    if name in NAME_ALIASES:
        name = NAME_ALIASES[name]
    return " ".join(name.split()).title()


def download_dataset(url: str = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv") -> pd.DataFrame:
    print("Descargando dataset desde GitHub...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    print(f"Dataset descargado: {len(resp.content):,} bytes")
    return pd.read_csv(StringIO(resp.text))


def filter_dataset(df: pd.DataFrame, min_year: int = 2000, min_matches: int = 15) -> pd.DataFrame:
    """
    Filtra el dataset por año mínimo y elimina equipos con pocos partidos.
    
    Args:
        df: DataFrame con los resultados.
        min_year: Año mínimo para incluir partidos (None para todos).
        min_matches: Número mínimo de partidos que debe tener un equipo para ser incluido.
    
    Returns:
        DataFrame filtrado.
    """
    initial_count = len(df)
    
    # Preprocesamiento básico
    print(f"Filtrando dataset (desde {min_year}, solo selecciones FIFA, min {min_matches} partidos)...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= f"{min_year}-01-01"]
    
    df["home_team"] = df["home_team"].apply(normalize_name)
    df["away_team"] = df["away_team"].apply(normalize_name)
    
    mask = df["home_team"].isin(FIFA_TEAMS) & df["away_team"].isin(FIFA_TEAMS)
    df = df[mask].copy()
    
    df["weight"] = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
    df = df[df["weight"] > 0]
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
    
    after_preprocessing = len(df)
    print(f"Despues de preprocesamiento: {initial_count} -> {after_preprocessing} partidos")
    
    # Filtro estructural: Excluir equipos con menos de 'min_matches' partidos
    if min_matches > 0:
        all_teams = list(df["home_team"]) + list(df["away_team"])
        team_counts = Counter(all_teams)
        
        # Guardar cantidad antes del filtro para el mensaje
        pre_filter_count = len(df)
        
        # Identificar equipos eliminados
        teams_excluded = {team for team, count in team_counts.items() if count < min_matches}
        
        df = df[
            (df["home_team"].map(team_counts) >= min_matches) & 
            (df["away_team"].map(team_counts) >= min_matches)
        ]
        
        print(f"Filtro estructural (min {min_matches} partidos): {pre_filter_count} -> {len(df)} partidos")
        if teams_excluded:
            print(f"Equipos eliminados por baja exposición ({len(teams_excluded)}): {sorted(teams_excluded)}")

    print(f"\nPartidos filtrados: {len(df):,}")
    print(f"Equipos unicos: {pd.concat([df['home_team'], df['away_team']]).nunique()}")
    print(f"Rango de fechas: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\nDistribucion por torneo (top 10):")
    print(df["tournament"].value_counts().head(10).to_string())

    return df


def build_matches(df: pd.DataFrame) -> list[MatchData]:
    """Construye lista de partidos usando itertuples para mayor eficiencia."""
    matches = []
    for row in df.itertuples(index=False):
        # Access columns by index since column names with special chars may cause issues
        # Columns: date, home_team, away_team, home_score, away_score, tournament, neutral, weight
        matches.append(
            MatchData(
                home_team=row.home_team,
                away_team=row.away_team,
                home_goals=int(row.home_score),
                away_goals=int(row.away_score),
                is_neutral=bool(getattr(row, 'neutral', False)),
                weight=float(row.weight) if hasattr(row, 'weight') else 1.0,
                match_date=row.date.date() if pd.notna(row.date) else None,
            )
        )
    return matches


def print_results(data: dict):
    gp = data.get("global_parameters", {})
    teams = data.get("teams", {})

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"   gamma (home advantage) : {gp.get('home_advantage_gamma')}")
    print(f"   rho (tau correction)   : {gp.get('rho_correction')}")
    print(f"   lambda (regularizacion): {gp.get('lambda_reg')}")
    print(f"   half-life (dias)       : {gp.get('half_life_days')}")
    print(f"   NLL final              : {gp.get('negative_log_likelihood')}")
    print(f"   Equipos entrenados     : {len(teams)}")
    print(f"   Fecha referencia       : {gp.get('reference_date')}")
    print("=" * 60)

    print("\nTop 10 Ataque:")
    sorted_atk = sorted(teams.items(), key=lambda x: x[1]["attack"], reverse=True)[:10]
    for name, r in sorted_atk:
        print(f"   {name:25s}  attack={r['attack']:.4f}  defense={r['defense']:.4f}")

    print("\nTop 10 Defensa (menor = mejor):")
    sorted_def = sorted(teams.items(), key=lambda x: x[1]["defense"])[:10]
    for name, r in sorted_def:
        print(f"   {name:25s}  attack={r['attack']:.4f}  defense={r['defense']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Entrena Dixon-Coles localmente")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=0.5)
    parser.add_argument("--half-life", dest="half_life_days", type=float, default=730.0)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--min-matches", dest="min_matches", type=int, default=15,
                        help="Minimo de partidos para incluir un equipo (filtro estructural)")
    args = parser.parse_args()

    if args.csv:
        print(f"Cargando CSV local: {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        df = download_dataset()

    df = filter_dataset(df, min_year=args.min_year, min_matches=args.min_matches)

    if len(df) == 0:
        print("No quedaron partidos despues de filtrar.")
        sys.exit(1)

    print(f"\nConstruyendo {len(df):,} objetos MatchData...")
    matches = build_matches(df)

    # Cargar ratings previos para warm-start
    previous_ratings = load_previous_ratings()

    print(f"Entrenando modelo con {len(matches):,} partidos...")
    result = train_ratings(
        matches,
        lambda_reg=args.lambda_reg,
        half_life_days=args.half_life_days,
        reference_date=date.today(),
        previous_ratings=previous_ratings,  # Warm-start activado
    )

    gp = result.get("global_parameters", {})
    save_trained_ratings(
        result["teams"],
        gamma=gp.get("home_advantage_gamma", 1.0),
        rho=gp.get("rho_correction", -0.13),
    )

    print_results(result)
    print("\nModelo guardado en data/trained_model.json")


if __name__ == "__main__":
    main()
