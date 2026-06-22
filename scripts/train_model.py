import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from mundial_betting.dixon_coles import train_ratings
from mundial_betting.data import save_trained_ratings


DESDE_ANO = 2022
FILTRAR_AMISTOSOS = (
    False  # En False para mantener los puentes intercontinentales recientes
)


class Match:
    def __init__(self, data):
        self.home_team = data["home_team"]
        self.away_team = data["away_team"]
        self.tournament = data.get("tournament", "Friendly")
        self.date = data["date"]

        # FIX m6: Manejo robusto de fechas con mensaje de error claro
        date_str = data.get("date", "")
        try:
            self.match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                clean_date = date_str.split("T")[0]
                self.match_date = datetime.strptime(clean_date, "%Y-%m-%d").date()
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Formato de fecha no reconocido: {date_str!r}") from e

        self.neutral = data.get("neutral", False)
        self.is_neutral = data.get("neutral", False)

        # FIX C2: Mantener goles enteros para respetar el modelo Poisson.
        # Los coeficientes regionales se aplican al PESO, no a los goles.
        home_g = int(data["home_score"])
        away_g = int(data["away_score"])

        # Peso base del partido
        weight_factor = 1.0

        # El Mundial absoluto NO se toca, es nuestra ancla de verdad intercontinental
        es_mundial = (
            "FIFA World Cup" in self.tournament
            and "qualification" not in self.tournament.lower()
        )

        if not es_mundial:
            # Asia (AFC)
            if "AFC" in self.tournament or "Asian" in self.tournament:
                weight_factor = 0.65
            # Norte y Centroamérica (CONCACAF)
            elif "CONCACAF" in self.tournament or "Gold Cup" in self.tournament:
                weight_factor = 0.70
            # África (CAF)
            elif "CAF" in self.tournament or "Africa" in self.tournament:
                weight_factor = 0.80
            # Oceanía (OFC)
            elif "OFC" in self.tournament or "Oceania" in self.tournament:
                weight_factor = 0.40
            # UEFA y CONMEBOL conservan 1.0

        self.weight = data.get("weight", 1.0) * weight_factor

        # Mapeo de las variables que dixon_coles.py busca internamente
        self.home_score = home_g
        self.away_score = away_g
        self.home_goals = home_g
        self.away_goals = away_g


def train_with_json():
    print("📂 Cargando partidos…")
    with open("mundial_betting/sample_matches.json", "r", encoding="utf-8") as f:
        matches_dict = json.load(f)

    print(f"📋 Total en base de datos: {len(matches_dict)} partidos.")

    # 1. Filtro de año
    filtered_dict = [
        m for m in matches_dict if int(m["date"].split("-")[0]) >= DESDE_ANO
    ]

    # 2. Filtro de Amistosos
    if FILTRAR_AMISTOSOS:
        filtered_dict = [m for m in filtered_dict if m.get("tournament") != "Friendly"]

    print(
        f"✂️ Filtro aplicado (>= {DESDE_ANO} | Conservar Amistosos: {not FILTRAR_AMISTOSOS}): {len(filtered_dict)} partidos seleccionados."
    )

    # Instanciar las clases usando la definición global
    matches = [Match(m) for m in filtered_dict]

    print("🚀 Iniciando optimización Dixon-Coles rápida...")
    result = train_ratings(matches)

    if not result.get("trained_at"):
        result["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global_params = result.get("global_parameters", {})

    # Extraer llaves mapeadas reales de la consola
    rho_val = global_params.get("rho_correction", 0.0)
    home_adv = global_params.get("home_advantage_gamma", 1.0)

    print("\n🎉 ¡Entrenamiento completado exitosamente!")
    print(f"Fecha: {result.get('trained_at')}")
    print(f"Equipos calibrados: {len(result.get('teams', {}))}")
    print(f"Factor de Localía (Home Advantage): {home_adv}")
    print(f"Corrección de Empate (Rho Correction): {rho_val}")

    # FIX C3 / M5: Guardar en disco con gamma y rho correctos usando kwargs explícitos
    save_trained_ratings(
        result.get("teams", {}),
        gamma=home_adv,
        rho=rho_val,
    )

    print("💾 Ratings guardados en disco con éxito.")
    return result


if __name__ == "__main__":
    train_with_json()
