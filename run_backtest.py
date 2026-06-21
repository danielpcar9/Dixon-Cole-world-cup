import json
from datetime import datetime
import numpy as np
from types import SimpleNamespace

# Imports corregidos para sincronizar estados de memoria
from mundial_betting.data import normalize_team_name, TEAMS
from mundial_betting.dixon_coles import (
    train_ratings,
    predict_match,
    set_trained_gamma,
    set_trained_rho,
)


def run_historical_backtest():
    path_matches = "mundial_betting/sample_matches.json"

    try:
        with open(path_matches, "r", encoding="utf-8") as f:
            all_matches = json.load(f)
    except FileNotFoundError:
        print(
            f"❌ No se encontró {path_matches}. Ejecuta primero load_historical_data.py"
        )
        return

    # Ordenar partidos por fecha para un backtest cronológico real
    all_matches.sort(key=lambda x: x["date"])

    print(f"📦 Total de partidos disponibles para simulación: {len(all_matches)}")

    # 1. Definir Ventanas Temporales (Filtro optimizado a 5 años de entrenamiento)
    train_set = [m for m in all_matches if "2018-01-01" <= m["date"] < "2023-01-01"]
    test_set = [m for m in all_matches if m["date"] >= "2023-01-01"]

    print(f"🏋️ Partidos de entrenamiento optimizados (2018-2022): {len(train_set)}")
    print(f"🧪 Partidos de evaluación (2023-Presente): {len(test_set)}")

    if len(test_set) == 0:
        print(
            "⚠️ No hay suficientes partidos en el set de evaluación para calcular métricas."
        )
        return

    # 2. Entrenar el modelo con el set histórico base
    print("\n⚙️ Calibrando parámetros Dixon-Coles con SciPy...")

    formatted_train = []
    for m in train_set:
        match_date_obj = datetime.strptime(m["date"], "%Y-%m-%d").date()

        home_norm = normalize_team_name(m["home_team"])
        away_norm = normalize_team_name(m["away_team"])

        formatted_train.append(
            SimpleNamespace(
                home_team=home_norm,
                away_team=away_norm,
                home_score=m["home_score"],
                away_score=m["away_score"],
                home_goals=m["home_score"],
                away_goals=m["away_score"],
                is_neutral=m["neutral"],
                neutral=m["neutral"],
                weight=1.0,
                match_date=match_date_obj,
                date=match_date_obj,
            )
        )

    training_results = train_ratings(
        formatted_train, lambda_reg=0.4, half_life_days=730
    )
    print("✅ Entrenamiento del bloque histórico completado con éxito.")

    # Inyectar parámetros globales en memoria
    global_params = training_results.get("global_parameters", {})
    set_trained_gamma(global_params.get("home_advantage_gamma", 1.0))
    set_trained_rho(global_params.get("rho_correction", -0.13))

    # CORRECCIÓN CRÍTICA: Convertir los dicts de los equipos a SimpleNamespace para que tengan el atributo .attack
    TEAMS.clear()
    for team_name, stats in training_results["teams"].items():
        if isinstance(stats, dict):
            TEAMS[team_name] = SimpleNamespace(**stats)
        else:
            TEAMS[team_name] = stats

    print(f"🔑 Equipos calibrados y cargados en memoria activa: {len(TEAMS)}")

    # 3. Ciclo de Evaluación (Backtesting)
    brier_scores = []
    log_losses = []
    correct_predictions = 0
    total_evaluated = 0

    print("\n🚀 Ejecutando predicciones sobre el set de evaluación...")

    for match in test_set:
        home_norm = normalize_team_name(match["home_team"])
        away_norm = normalize_team_name(match["away_team"])

        if home_norm not in TEAMS or away_norm not in TEAMS:
            continue

        home_goals = match["home_score"]
        away_goals = match["away_score"]

        actual_outcome = "draw"
        if home_goals > away_goals:
            actual_outcome = "home"
        elif away_goals > home_goals:
            actual_outcome = "away"

        try:
            pred = predict_match(
                home_team=home_norm,
                away_team=away_norm,
                neutral=match["neutral"],
                odds=None,
            )
            probs = pred["probabilities"]

            p_home = probs["home"]
            p_draw = probs["draw"]
            p_away = probs["away"]
        except Exception as e:
            print(
                f"⚠️ Error prediciendo {match['home_team']} vs {match['away_team']}: {str(e)}"
            )
            continue

        # --- Cálculo de Métricas Estadísticas ---
        y_real = np.array(
            [
                1.0 if actual_outcome == "home" else 0.0,
                1.0 if actual_outcome == "draw" else 0.0,
                1.0 if actual_outcome == "away" else 0.0,
            ]
        )

        y_pred = np.array([p_home, p_draw, p_away])

        brier = float(np.sum((y_pred - y_real) ** 2))
        brier_scores.append(brier)

        prob_actual = probs[actual_outcome]
        log_loss = -float(np.log(max(prob_actual, 1e-15)))
        log_losses.append(log_loss)

        outcomes_keys = ["home", "draw", "away"]
        most_likely_outcome = outcomes_keys[int(np.argmax(y_pred))]
        if most_likely_outcome == actual_outcome:
            correct_predictions += 1

        total_evaluated += 1

    # 4. Mostrar Resultados del Reporte de Calidad Operativa
    print("\n" + "=" * 45)
    print("📊 REPORTE DE RENDIMIENTO DIXON-COLES (2023-2026)")
    print("=" * 45)
    print(f"🎯 Partidos evaluados de forma efectiva: {total_evaluated}")

    if total_evaluated > 0:
        avg_brier = np.mean(brier_scores)
        avg_log_loss = np.mean(log_losses)
        accuracy = (correct_predictions / total_evaluated) * 100

        print(f"📉 Avg Brier Score: {avg_brier:.4f}  (Objetivo: < 0.22)")
        print(f"🪵 Avg Log Loss:    {avg_log_loss:.4f}  (Objetivo: < 1.00)")
        print(f"🎯 Accuracy Direccional: {accuracy:.2f}%")
        print("-" * 45)
        print("💡 Nota: Un Accuracy > 45% en fútbol internacional es un")
        print("   excelente indicador base considerando los empates.")
    else:
        print(
            "❌ No se pudieron cruzar suficientes equipos entrenados con el set de prueba."
        )
    print("=" * 45 + "\n")


if __name__ == "__main__":
    run_historical_backtest()
