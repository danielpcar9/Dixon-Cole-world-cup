import json
from datetime import datetime
from typing import Literal
import numpy as np
from types import SimpleNamespace

from mundial_betting.data import normalize_team_name, TEAMS, TeamRating
from mundial_betting.dixon_coles import (
    train_ratings,
    predict_match,
    set_trained_gamma,
    set_trained_rho,
    get_trained_rho,
    get_trained_gamma,
)
from mundial_betting.calibration import (
    ProbabilityCalibrator,
    create_calibrator_from_historical_data,
)


def calculate_kelly_stake(probability: float, decimal_odds: float, kelly_fraction: float = 0.25) -> float:
    """Calcula el stake óptimo usando Criterio de Kelly Fraccional.
    
    Args:
        probability: Probabilidad estimada del evento (0-1)
        decimal_odds: Cuotas decimales del mercado
        kelly_fraction: Fracción de Kelly a aplicar (0.25 = quarter-Kelly, más conservador)
    
    Returns:
        Porcentaje del bankroll a apostar (0 si no hay valor positivo)
    """
    if decimal_odds <= 1:
        return 0.0
    
    b = decimal_odds - 1  # Ganancia neta por unidad apostada
    p = probability
    q = 1 - probability
    
    # Fórmula de Kelly: f* = (bp - q) / b
    kelly = (b * p - q) / b
    
    # Aplicar fracción y asegurar que sea no negativo
    return max(0.0, kelly * kelly_fraction)


def run_historical_backtest(
    calibration_method: Literal["platt", "isotonic", None] = "isotonic",
    kelly_fraction: float = 0.25,
    initial_bankroll: float = 1000.0,
):
    """Ejecuta backtest avanzado con métricas completas de apuestas.
    
    Args:
        calibration_method: Método de calibración ('platt', 'isotonic', o None para sin calibrar)
        kelly_fraction: Fracción de Kelly para gestión de stakes (0.25 = quarter-Kelly)
        initial_bankroll: Bankroll inicial para simulación
    """
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

    # 1. ENTRENAMIENTO ÚNICO (rápido con mejoras H1, H2, H3)
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

    # 1.5 CALIBRACIÓN DE PROBABILIDADES (H4)
    calibrator = None
    if calibration_method:
        print(f"\n📊 Calibrando probabilidades con método {calibration_method}...")
        
        # Generar historial de predicciones en el set de entrenamiento para calibrar
        predictions_history = []
        for match in train_set[:500]:  # Usar subset para velocidad
            home_norm = normalize_team_name(match["home_team"])
            away_norm = normalize_team_name(match["away_team"])
            
            if home_norm not in TEAMS or away_norm not in TEAMS:
                continue
            
            try:
                pred = predict_match(
                    home_team=home_norm,
                    away_team=away_norm,
                    neutral=match["neutral"],
                    odds=None,
                )
                predictions_history.append({
                    "predicted_probs": pred["probabilities"],
                    "actual_result": {
                        "home_goals": match["home_score"],
                        "away_goals": match["away_score"],
                    },
                })
            except Exception:
                continue
        
        if len(predictions_history) > 50:
            calibrator = create_calibrator_from_historical_data(
                predictions_history, method=calibration_method
            )
            calib_eval = calibrator.evaluate_calibration(
                {k: [p[k] for p in predictions_history if k in p] 
                 for k in ["over_25", "btts_yes"]},
                {k: [(p["actual_result"]["home_goals"] + p["actual_result"]["away_goals"]) > 2.5 if k == "over_25" 
                    else (p["actual_result"]["home_goals"] > 0 and p["actual_result"]["away_goals"] > 0)
                    for p in predictions_history if k in p.get("predicted_probs", {})]
                 for k in ["over_25", "btts_yes"]}
            )
            print("✅ Calibración completada. Mejoras en Brier Score:")
            for market, metrics in calib_eval.items():
                print(f"   {market}: {metrics['improvement_pct']:.1f}%")
        else:
            print("⚠️ Insuficientes datos para calibración.")

    # 2. EVALUACIÓN CON MÉTRICAS COMPLETAS (H5)
    brier_scores = []
    log_losses = []
    correct_predictions = 0
    total_evaluated = 0
    
    # Métricas de apuestas (H5)
    bankroll = initial_bankroll
    bankroll_history = [initial_bankroll]
    bets_placed = 0
    bets_won = 0
    total_staked = 0.0
    total_returned = 0.0
    roi_by_market = {"1X2": [], "over_25": [], "btts_yes": []}
    daily_pnl = []
    
    # Para drawdown y sharpe
    cumulative_pnl = 0
    peak_bankroll = initial_bankroll
    max_drawdown = 0.0
    daily_returns = []

    print("\n🚀 Evaluando predicciones y simulando apuestas...")

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
        
        # Resultados reales para otros mercados
        actual_over_25 = (home_score + away_score) > 2.5
        actual_btts_yes = (home_score > 0 and away_score > 0)

        try:
            pred = predict_match(
                home_team=home_norm,
                away_team=away_norm,
                neutral=match["neutral"],
                odds=None,
            )
            probs = pred["probabilities"]
            
            # H4: Aplicar calibración si está disponible
            if calibrator:
                probs = calibrator.calibrate(probs)
            
            p_home = probs["home"]
            p_draw = probs["draw"]
            p_away = probs["away"]
            p_over_25 = probs["over_25"]
            p_btts_yes = probs["btts_yes"]
        except Exception as e:
            print(
                f"⚠️ Error prediciendo {match['home_team']} vs {match['away_team']}: {e}"
            )
            continue

        # Métricas de clasificación (1X2)
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
        
        # H5: SIMULACIÓN DE APUESTAS CON KELLY
        # Simular cuotas justas (sin vig) para cada mercado
        def simulate_decimal_odds(prob: float, margin: float = 0.05) -> float:
            """Simula cuotas decimales con un margen típico de casas de apuestas."""
            fair_odds = 1 / prob if prob > 0 else float('inf')
            # Aplicar margen (la casa paga menos que lo justo)
            return fair_odds * (1 - margin)
        
        # Mercado 1X2
        for outcome_key, prob in [("home", p_home), ("draw", p_draw), ("away", p_away)]:
            if prob < 0.45:  # Solo apostar si hay cierta confianza
                continue
            
            decimal_odds = simulate_decimal_odds(prob)
            stake_pct = calculate_kelly_stake(prob, decimal_odds, kelly_fraction)
            
            if stake_pct > 0.01:  # Mínimo 1% de Kelly
                stake_amount = bankroll * stake_pct
                total_staked += stake_amount
                
                # Determinar resultado
                won = (outcome_key == actual_outcome)
                if won:
                    returns = stake_amount * decimal_odds
                    total_returned += returns
                    bankroll += (returns - stake_amount)
                    bets_won += 1
                else:
                    bankroll -= stake_amount
                
                bets_placed += 1
                roi_by_market["1X2"].append((returns - stake_amount) / stake_amount if won else -1)
        
        # Mercado Over/Under 2.5
        if p_over_25 > 0.45:
            decimal_odds = simulate_decimal_odds(p_over_25)
            stake_pct = calculate_kelly_stake(p_over_25, decimal_odds, kelly_fraction)
            
            if stake_pct > 0.01:
                stake_amount = bankroll * stake_pct
                total_staked += stake_amount
                
                won = actual_over_25
                if won:
                    returns = stake_amount * decimal_odds
                    total_returned += returns
                    bankroll += (returns - stake_amount)
                    bets_won += 1
                else:
                    bankroll -= stake_amount
                
                bets_placed += 1
                roi_by_market["over_25"].append((returns - stake_amount) / stake_amount if won else -1)
        
        # Mercado BTTS
        if p_btts_yes > 0.45:
            decimal_odds = simulate_decimal_odds(p_btts_yes)
            stake_pct = calculate_kelly_stake(p_btts_yes, decimal_odds, kelly_fraction)
            
            if stake_pct > 0.01:
                stake_amount = bankroll * stake_pct
                total_staked += stake_amount
                
                won = actual_btts_yes
                if won:
                    returns = stake_amount * decimal_odds
                    total_returned += returns
                    bankroll += (returns - stake_amount)
                    bets_won += 1
                else:
                    bankroll -= stake_amount
                
                bets_placed += 1
                roi_by_market["btts_yes"].append((returns - stake_amount) / stake_amount if won else -1)
        
        # Tracking diario para drawdown y Sharpe
        bankroll_history.append(bankroll)
        current_pnl = bankroll - initial_bankroll
        if current_pnl > peak_bankroll - initial_bankroll:
            peak_bankroll = bankroll
        
        drawdown = (peak_bankroll - bankroll) / peak_bankroll
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        
        if len(bankroll_history) > 1:
            daily_ret = (bankroll_history[-1] - bankroll_history[-2]) / bankroll_history[-2]
            daily_returns.append(daily_ret)

    # 3. RESULTADOS COMPLETOS
    print("\n" + "=" * 60)
    print("📊 REPORTE COMPLETO DE RENDIMIENTO (2023-2026)")
    print("=" * 60)
    
    # Métricas de clasificación
    print("\n🎯 MÉTRICAS DE CLASIFICACIÓN:")
    print(f"   Partidos evaluados: {total_evaluated}")

    if total_evaluated > 0:
        avg_brier = np.mean(brier_scores)
        avg_log_loss = np.mean(log_losses)
        accuracy = (correct_predictions / total_evaluated) * 100

        print(f"   📉 Avg Brier Score: {avg_brier:.4f}  (Objetivo: < 0.22)")
        print(f"   🪵 Avg Log Loss:    {avg_log_loss:.4f}  (Objetivo: < 1.00)")
        print(f"   🎯 Accuracy Direccional: {accuracy:.2f}%")
        
        if calibration_method:
            print(f"   ✅ Calibración aplicada: {calibration_method}")

    # Métricas de apuestas (H5)
    print("\n💰 MÉTRICAS DE APUESTAS:")
    print(f"   Bankroll Inicial: ${initial_bankroll:,.2f}")
    print(f"   Bankroll Final: ${bankroll:,.2f}")
    print(f"   Beneficio Neto: ${bankroll - initial_bankroll:,.2f}")
    print(f"   ROI Total: {(bankroll - initial_bankroll) / initial_bankroll * 100:.2f}%")
    
    if bets_placed > 0:
        win_rate = (bets_won / bets_placed) * 100
        avg_roi = np.mean([r for rois in roi_by_market.values() for r in rois]) if any(roi_by_market.values()) else 0
        print(f"   Apuestas Realizadas: {bets_placed}")
        print(f"   Tasa de Acierto: {win_rate:.2f}%")
        print(f"   ROI Promedio por Apuesta: {avg_roi * 100:.2f}%")
    
    print(f"\n   ROI por Mercado:")
    for market, rois in roi_by_market.items():
        if rois:
            avg_roi_market = np.mean(rois) * 100
            print(f"      {market}: {avg_roi_market:.2f}% ({len(rois)} apuestas)")
    
    # Drawdown y Sharpe
    print(f"\n📈 GESTIÓN DE RIESGO:")
    print(f"   Max Drawdown: {max_drawdown * 100:.2f}%")
    
    if len(daily_returns) > 1:
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        print(f"   Sharpe Ratio (anualizado): {sharpe_ratio:.2f}")
    
    print("-" * 60)
    
    # Evaluación de objetivos
    print("\n✅ EVALUACIÓN DE OBJETIVOS:")
    if total_evaluated > 0:
        if avg_brier < 0.22:
            print("   ✅ Brier Score CUMPLE el objetivo (< 0.22)")
        else:
            print("   ❌ Brier Score NO cumple el objetivo")

        if avg_log_loss < 1.00:
            print("   ✅ Log Loss CUMPLE el objetivo (< 1.00)")
        else:
            print("   ❌ Log Loss NO cumple el objetivo")

        if accuracy > 45.0:
            print("   ✅ Accuracy CUMPLE el objetivo (> 45%)")
        else:
            print("   ❌ Accuracy NO cumple el objetivo")
    
    if bets_placed > 0:
        if (bankroll - initial_bankroll) / initial_bankroll > 0.05:
            print("   ✅ ROI CUMPLE el objetivo (> 5%)")
        else:
            print("   ❌ ROI NO cumple el objetivo")
        
        if max_drawdown < 0.20:
            print("   ✅ Drawdown CUMPLE el objetivo (< 20%)")
        else:
            print("   ⚠️ Drawdown elevado (>= 20%)")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_historical_backtest()
