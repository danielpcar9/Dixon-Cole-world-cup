import json
from datetime import datetime
import numpy as np
from types import SimpleNamespace

from mundial_betting.data import normalize_team_name, TEAMS, TeamRating
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

    all_matches.sort(key=lambda x: x["date"])

    print(f"📦 Total de partidos disponibles para simulación: {len(all_matches)}")

    train_set = [m for m in all_matches if "2018-01-01" <= m["date"] < "2023-01-01"]
    test_set = [m for m in all_matches if m["date"] >= "2023-01-01"]

    print(f"🏋️ Partidos de entrenamiento (2018-2022): {len(train_set)}")
    print(f"🧪 Partidos de evaluación (2023-Presente): {len(test_set)}")

    if len(test_set) == 0:
        print("⚠️ No hay suficientes partidos en el set de evaluación.")
        return

    # 1. ENTRENAMIENTO ÚNICO (rápido)
    print("\n⚙️ Calibrando parámetros Dixon-Coles...")
    formatted_train = []
    for m in train_set:
        match_date_obj = datetime.strptime(m["date"], "%Y-%m-%d").date()
        home_norm = normalize_team_name(m["home_team"])
        away_norm = normalize_team_name(m["away_team"])

        formatted_train.append(
            SimpleNamespace(
                home_team=home_norm,
                away_team=away_norm,
                home_goals=m["home_score"],
                away_goals=m["away_score"],
                is_neutral=m["neutral"],
                weight=1.0,
                match_date=match_date_obj,
            )
        )

    training_results = train_ratings(
        formatted_train, lambda_reg=1.0, half_life_days=730
    )
    print("✅ Entrenamiento completado.")

    global_params = training_results.get("global_parameters", {})
    set_trained_gamma(global_params.get("home_advantage_gamma", 1.0))
    set_trained_rho(global_params.get("rho_correction", -0.13))

    # Cargar equipos en memoria como TeamRating
    TEAMS.clear()
    for team_name, stats in training_results["teams"].items():
        TEAMS[team_name] = TeamRating(**stats) if isinstance(stats, dict) else stats

    print(f"🔑 Equipos calibrados: {len(TEAMS)}")

    # 2. EVALUACIÓN (sin reentrenamiento intermedio — rápido)
    brier_scores = []
    log_losses = []
    correct_predictions = 0
    total_evaluated = 0

    print("\n🚀 Evaluando predicciones...")

    for match in test_set:
        home_norm = normalize_team_name(match["home_team"])
        away_norm = normalize_team_name(match["away_team"])

        if home_norm not in TEAMS or away_norm not in TEAMS:
            continue

        home_score = match["home_score"]
        away_score = match["away_score"]

        actual_outcome = "draw"
        if home_score > away_score:
            actual_outcome = "home"
        elif away_score > home_score:
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
                f"⚠️ Error prediciendo {match['home_team']} vs {match['away_team']}: {e}"
            )
            continue

        y_real = np.array(
            [
                1.0 if actual_outcome == "home" else 0.0,
                1.0 if actual_outcome == "draw" else 0.0,
                1.0 if actual_outcome == "away" else 0.0,
            ]
        )
        y_pred = np.array([p_home, p_draw, p_away])

        # FIX C1: Brier como PROMEDIO
        brier = float(np.mean((y_pred - y_real) ** 2))
        brier_scores.append(brier)

        prob_actual = probs[actual_outcome]
        log_loss = -float(np.log(max(prob_actual, 1e-15)))
        log_losses.append(log_loss)

        outcomes_keys = ["home", "draw", "away"]
        most_likely = outcomes_keys[int(np.argmax(y_pred))]
        if most_likely == actual_outcome:
            correct_predictions += 1

        total_evaluated += 1

    # 3. RESULTADOS
    print("\n" + "=" * 50)
    print("📊 REPORTE DE RENDIMIENTO DIXON-COLES (2023-2026)")
    print("=" * 50)
    print(f"🎯 Partidos evaluados: {total_evaluated}")

    if total_evaluated > 0:
        avg_brier = np.mean(brier_scores)
        avg_log_loss = np.mean(log_losses)
        accuracy = (correct_predictions / total_evaluated) * 100

        print(f"📉 Avg Brier Score: {avg_brier:.4f}  (Objetivo: < 0.22)")
        print(f"🪵 Avg Log Loss:    {avg_log_loss:.4f}  (Objetivo: < 1.00)")
        print(f"🎯 Accuracy Direccional: {accuracy:.2f}%")
        print("-" * 50)

        if avg_brier < 0.22:
            print("✅ Brier Score CUMPLE el objetivo (< 0.22)")
        else:
            print("❌ Brier Score NO cumple el objetivo")

        if avg_log_loss < 1.00:
            print("✅ Log Loss CUMPLE el objetivo (< 1.00)")
        else:
            print("❌ Log Loss NO cumple el objetivo")

        if accuracy > 45.0:
            print("✅ Accuracy CUMPLE el objetivo (> 45%)")
        else:
            print("❌ Accuracy NO cumple el objetivo")
    else:
        print("❌ No se pudieron evaluar partidos.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_historical_backtest()
