import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from mundial_betting.dixon_coles import train_ratings
from mundial_betting.data import save_trained_ratings

# 🎛️ CONFIGURACIÓN
DESDE_ANO = 2022
FILTRAR_AMISTOSOS = False


def train_with_json():
    print("📂 Cargando partidos...")
    with open("mundial_betting/sample_matches.json", "r", encoding="utf-8") as f:
        matches_dict = json.load(f)

    print(f"📋 Total en base de datos: {len(matches_dict)} partidos.")

    # 1. Filtro de año
    filtered_dict = [
        m for m in matches_dict if int(m["date"].split("-")[0]) >= DESDE_ANO
    ]

    # 2. Filtro de Amistosos (Reduce drásticamente los equipos irrelevantes para el Mundial)
    if FILTRAR_AMISTOSOS:
        filtered_dict = [m for m in filtered_dict if m.get("tournament") != "Friendly"]

    print(
        f"✂️ Filtro aplicado (>= {DESDE_ANO} | Oficiales: {FILTRAR_AMISTOSOS}): {len(filtered_dict)} partidos seleccionados."
    )

    class Match:
        def __init__(self, data):
            self.home_team = data["home_team"]
            self.away_team = data["away_team"]
            self.home_score = data["home_score"]
            self.away_score = data["away_score"]
            self.home_goals = data["home_score"]
            self.away_goals = data["away_score"]

            self.date = data["date"]
            try:
                self.match_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
            except ValueError:
                clean_date = data["date"].split("T")[0]
                self.match_date = datetime.strptime(clean_date, "%Y-%m-%d").date()

            self.weight = data.get("weight", 1.0)
            self.tournament = data.get("tournament", "Friendly")
            self.neutral = data.get("neutral", False)
            self.is_neutral = data.get("neutral", False)

    matches = [Match(m) for m in filtered_dict]

    print("🚀 Iniciando optimización Dixon-Coles rápida...")
    result = train_ratings(matches)

    if not result.get("trained_at"):
        result["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global_params = result.get("global_parameters", {})

    # 💡 FIX DEFINITIVO: Extraer las llaves reales reveladas por tu consola
    rho_val = global_params.get("rho_correction", 0.0)
    home_adv = global_params.get("home_advantage_gamma", 1.0)

    print("\n🎉 ¡Entrenamiento completado exitosamente!")
    print(f"Fecha: {result.get('trained_at')}")
    print(f"Equipos calibrados: {len(result.get('teams', {}))}")
    print(f"Factor de Localía (Home Advantage): {home_adv}")
    print(f"Corrección de Empate (Rho Correction): {rho_val}")

    # Guardar ratings pasándole el valor real de rho_correction
    save_trained_ratings(result.get("teams", {}), rho_val)
    print("💾 Ratings guardados en disco con éxito.")

    return result


if __name__ == "__main__":
    train_with_json()
