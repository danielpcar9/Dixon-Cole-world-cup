#!/usr/bin/env python3
"""
retrain_model.py
================
Descarga el dataset internacional de martj42, filtra partidos relevantes
para predicciones de nivel mundialista, y reentrena el modelo Dixon-Coles
via la API local /train.

Uso:
    python retrain_model.py
    python retrain_model.py --port 8000 --half-life 365 --lambda 0.3
"""

import argparse
import sys
from datetime import date
from io import StringIO

import requests
import pandas as pd

# =============================================================================
# CONFIGURACION — Selecciones FIFA reconocidas
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
    print(f"Descargando dataset desde GitHub...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    print(f"Dataset descargado: {len(resp.content):,} bytes")
    return pd.read_csv(StringIO(resp.text))


def filter_dataset(df: pd.DataFrame, min_year: int = 2000) -> pd.DataFrame:
    print(f"Filtrando dataset (desde {min_year}, solo selecciones FIFA)...")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= f"{min_year}-01-01"]

    df["home_team"] = df["home_team"].apply(normalize_name)
    df["away_team"] = df["away_team"].apply(normalize_name)

    mask = df["home_team"].isin(FIFA_TEAMS) & df["away_team"].isin(FIFA_TEAMS)
    df = df[mask].copy()

    df["weight"] = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]

    print(f"Partidos filtrados: {len(df):,}")
    print(f"Equipos unicos: {pd.concat([df['home_team'], df['away_team']]).nunique()}")
    print(f"Rango de fechas: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\nDistribucion por torneo (top 10):")
    print(df["tournament"].value_counts().head(10).to_string())

    return df


def build_train_payload(df: pd.DataFrame, lambda_reg: float, half_life_days: float) -> dict:
    matches = []
    for _, row in df.iterrows():
        matches.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "is_neutral": bool(row.get("neutral", False)),
            "weight": float(row["weight"]),
            "match_date": row["date"].strftime("%Y-%m-%d"),
        })

    return {
        "matches": matches,
        "lambda_reg": lambda_reg,
        "half_life_days": half_life_days,
        "reference_date": date.today().isoformat(),
    }


def send_train_request(payload: dict, base_url: str) -> dict:
    url = f"{base_url}/train"
    print(f"\nEnviando {len(payload['matches']):,} partidos a {url}...")
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


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

    import json, os
    backup_path = "data/trained_model_backup.json"
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nBackup guardado en: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description="Reentrena el modelo Dixon-Coles con datos internacionales filtrados.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL de la API")
    parser.add_argument("--min-year", type=int, default=2000, help="Ano minimo de partidos")
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=0.5, help="Lambda de regularizacion")
    parser.add_argument("--half-life", dest="half_life_days", type=float, default=730.0, help="Half-life en dias")
    parser.add_argument("--csv", default=None, help="Ruta a CSV local (opcional)")
    args = parser.parse_args()

    if args.csv:
        print(f"Cargando CSV local: {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        df = download_dataset()

    df = filter_dataset(df, min_year=args.min_year)

    if len(df) == 0:
        print("No quedaron partidos despues de filtrar.")
        sys.exit(1)

    payload = build_train_payload(df, args.lambda_reg, args.half_life_days)

    try:
        result = send_train_request(payload, args.url)
        print_results(result)
    except requests.exceptions.ConnectionError:
        print(f"No se pudo conectar a {args.url}/train. Esta corriendo el servidor?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error del servidor: {e}")
        try:
            print(e.response.json())
        except Exception:
            print(e.response.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
