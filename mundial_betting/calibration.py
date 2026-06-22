"""Módulo de calibración de probabilidades para corregir sesgos del modelo Poisson.

Implementa Platt Scaling e Isotonic Regression para asegurar que las probabilidades
predichas reflejen fielmente las frecuencias observadas en la realidad.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from typing import Literal, TypedDict


class CalibrationResult(TypedDict):
    """Resultado de la calibración."""
    calibrated_probs: dict[str, float]
    calibration_method: str
    brier_before: float
    brier_after: float
    calibration_improvement: float


class ProbabilityCalibrator:
    """Calibrador de probabilidades usando Platt Scaling o Isotonic Regression.
    
    El modelo de Poisson tiende a subestimar la varianza, especialmente en:
    - Mercados Over/Under 2.5
    - Mercado BTTS (Both Teams To Score)
    - Partidos de alta tensión (fases eliminatorias)
    
    Este calibrador corrige esas desviaciones sistemáticas.
    """
    
    def __init__(self, method: Literal["platt", "isotonic"] = "isotonic"):
        """
        Args:
            method: Método de calibración. 'platt' para Platt Scaling (paramétrico),
                   'isotonic' para Isotonic Regression (no paramétrico, más flexible).
        """
        self.method = method
        self._calibrators: dict[str, IsotonicRegression | tuple[float, float]] = {}
        self._is_fitted = False
    
    def fit(
        self,
        predicted_probs: dict[str, list[float]],
        actual_outcomes: dict[str, list[bool]],
    ) -> None:
        """Ajusta los calibradores usando datos históricos.
        
        Args:
            predicted_probs: Diccionario con listas de probabilidades predichas por mercado.
                            Ej: {"over_25": [0.65, 0.72, ...], "btts_yes": [0.55, 0.48, ...]}
            actual_outcomes: Diccionario con listas de resultados reales (True/False) por mercado.
                            Ej: {"over_25": [True, False, ...], "btts_yes": [True, True, ...]}
        """
        for market in predicted_probs.keys():
            if market not in actual_outcomes:
                continue
            
            probs = np.array(predicted_probs[market])
            outcomes = np.array(actual_outcomes[market], dtype=float)
            
            # Filtrar valores inválidos
            valid_mask = (probs >= 0) & (probs <= 1)
            probs = probs[valid_mask]
            outcomes = outcomes[valid_mask]
            
            if len(probs) < 10:
                # No hay suficientes datos para calibrar este mercado
                continue
            
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(probs, outcomes)
                self._calibrators[market] = calibrator
            else:  # platt
                params = self._fit_platt_scaling(probs, outcomes)
                self._calibrators[market] = params
        
        self._is_fitted = True
    
    def _fit_platt_scaling(
        self, probs: np.ndarray, outcomes: np.ndarray
    ) -> tuple[float, float]:
        """Ajusta Platt Scaling (regresión logística) a las probabilidades.
        
        La fórmula es: P_calibrated = 1 / (1 + exp(A * P_raw + B))
        
        Returns:
            Tupla (A, B) de parámetros óptimos.
        """
        def neg_log_likelihood(params: np.ndarray) -> float:
            A, B = params
            p_calibrated = 1 / (1 + np.exp(A * probs + B))
            p_calibrated = np.clip(p_calibrated, 1e-10, 1 - 1e-10)
            nll = -np.sum(
                outcomes * np.log(p_calibrated)
                + (1 - outcomes) * np.log(1 - p_calibrated)
            )
            return nll
        
        # Valores iniciales
        result = minimize(neg_log_likelihood, x0=[0.0, 0.0], method="L-BFGS-B")
        return float(result.x[0]), float(result.x[1])
    
    def calibrate(self, raw_probs: dict[str, float]) -> dict[str, float]:
        """Aplica calibración a un conjunto de probabilidades crudas.
        
        Args:
            raw_probs: Diccionario de probabilidades sin calibrar.
                      Ej: {"home": 0.45, "draw": 0.28, "away": 0.27, 
                           "over_25": 0.62, "btts_yes": 0.58}
        
        Returns:
            Diccionario con probabilidades calibradas. Si el calibrador no está
            ajustado para un mercado, devuelve la probabilidad original.
        """
        if not self._is_fitted:
            return raw_probs.copy()
        
        calibrated = {}
        for market, prob in raw_probs.items():
            if market in self._calibrators:
                calibrator = self._calibrators[market]
                if self.method == "isotonic":
                    calibrated_prob = float(calibrator.predict([prob])[0])
                else:  # platt
                    A, B = calibrator
                    calibrated_prob = 1 / (1 + np.exp(A * prob + B))
                
                # Asegurar que esté en rango válido
                calibrated[market] = max(0.001, min(0.999, calibrated_prob))
            else:
                calibrated[market] = prob
        
        return calibrated
    
    def evaluate_calibration(
        self,
        predicted_probs: dict[str, list[float]],
        actual_outcomes: dict[str, list[bool]],
    ) -> dict[str, dict[str, float]]:
        """Evalúa la calidad de calibración por mercado.
        
        Calcula:
        - Brier Score antes y después de calibrar
        - Tasa de acierto por decil de probabilidad
        - ECE (Expected Calibration Error)
        
        Returns:
            Diccionario con métricas de calibración por mercado.
        """
        results = {}
        
        for market in predicted_probs.keys():
            if market not in actual_outcomes:
                continue
            
            probs = np.array(predicted_probs[market])
            outcomes = np.array(actual_outcomes[market], dtype=float)
            
            valid_mask = (probs >= 0) & (probs <= 1)
            probs = probs[valid_mask]
            outcomes = outcomes[valid_mask]
            
            if len(probs) < 10:
                continue
            
            # Brier Score antes de calibrar
            brier_before = float(np.mean((probs - outcomes) ** 2))
            
            # Calibrar y calcular Brier después
            if market in self._calibrators:
                calibrator = self._calibrators[market]
                if self.method == "isotonic":
                    calibrated_probs = calibrator.predict(probs)
                else:
                    A, B = calibrator
                    calibrated_probs = 1 / (1 + np.exp(A * probs + B))
                
                brier_after = float(np.mean((calibrated_probs - outcomes) ** 2))
            else:
                brier_after = brier_before
            
            # ECE (Expected Calibration Error)
            ece = self._calculate_ece(probs, outcomes, n_bins=10)
            
            # Tasa de acierto por decil
            decile_accuracy = self._calculate_decile_accuracy(probs, outcomes, n_bins=10)
            
            results[market] = {
                "brier_before": round(brier_before, 6),
                "brier_after": round(brier_after, 6),
                "improvement_pct": round((brier_before - brier_after) / brier_before * 100, 2) if brier_before > 0 else 0,
                "ece": round(ece, 6),
                "decile_accuracy": decile_accuracy,
            }
        
        return results
    
    def _calculate_ece(self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Calcula Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(probs)
        
        for i in range(n_bins):
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if np.sum(mask) == 0:
                continue
            
            bin_probs = probs[mask]
            bin_outcomes = outcomes[mask]
            
            avg_prob = np.mean(bin_probs)
            avg_outcome = np.mean(bin_outcomes)
            
            ece += np.abs(avg_prob - avg_outcome) * np.sum(mask) / total_samples
        
        return ece
    
    def _calculate_decile_accuracy(
        self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> list[dict[str, float]]:
        """Calcula tasa de acierto por decil de probabilidad."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        decile_stats = []
        
        for i in range(n_bins):
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            count = np.sum(mask)
            
            if count == 0:
                decile_stats.append({
                    "bin_min": round(bin_boundaries[i], 2),
                    "bin_max": round(bin_boundaries[i + 1], 2),
                    "predicted_prob": round((bin_boundaries[i] + bin_boundaries[i + 1]) / 2, 2),
                    "observed_frequency": 0.0,
                    "count": 0,
                })
            else:
                observed_freq = float(np.mean(outcomes[mask]))
                decile_stats.append({
                    "bin_min": round(bin_boundaries[i], 2),
                    "bin_max": round(bin_boundaries[i + 1], 2),
                    "predicted_prob": round((bin_boundaries[i] + bin_boundaries[i + 1]) / 2, 2),
                    "observed_frequency": round(observed_freq, 4),
                    "count": int(count),
                })
        
        return decile_stats


def create_calibrator_from_historical_data(
    predictions_history: list[dict],
    method: Literal["platt", "isotonic"] = "isotonic",
) -> ProbabilityCalibrator:
    """Crea y ajusta un calibrador desde un historial de predicciones y resultados.
    
    Args:
        predictions_history: Lista de diccionarios con:
            - "predicted_probs": dict con probabilidades predichas
            - "actual_result": dict con resultados reales (ej: {"home_goals": 2, "away_goals": 1})
        method: Método de calibración ('platt' o 'isotonic')
    
    Returns:
        ProbabilityCalibrator ya ajustado con los datos históricos.
    """
    predicted_probs: dict[str, list[float]] = {}
    actual_outcomes: dict[str, list[bool]] = {}
    
    markets_to_calibrate = ["over_25", "under_25", "btts_yes", "btts_no"]
    
    for entry in predictions_history:
        probs = entry.get("predicted_probs", {})
        result = entry.get("actual_result", {})
        
        home_goals = result.get("home_goals", 0)
        away_goals = result.get("away_goals", 0)
        
        # Calcular outcomes reales
        outcomes = {
            "over_25": (home_goals + away_goals) > 2.5,
            "under_25": (home_goals + away_goals) <= 2.5,
            "btts_yes": (home_goals > 0 and away_goals > 0),
            "btts_no": not (home_goals > 0 and away_goals > 0),
        }
        
        for market in markets_to_calibrate:
            if market in probs:
                if market not in predicted_probs:
                    predicted_probs[market] = []
                    actual_outcomes[market] = []
                
                predicted_probs[market].append(probs[market])
                actual_outcomes[market].append(outcomes[market])
    
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(predicted_probs, actual_outcomes)
    
    return calibrator
