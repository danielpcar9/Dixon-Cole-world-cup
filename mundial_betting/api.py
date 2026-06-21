from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from types import SimpleNamespace
from datetime import date

# Imports desde la capa de datos
from mundial_betting.data import (
    TEAMS,
    TeamRating,
    get_team,
    normalize_team_name,
    save_trained_ratings,
    load_trained_model,
)

# Imports desde el motor matemático
from mundial_betting.dixon_coles import (
    predict_match,
    set_trained_gamma,
    set_trained_rho,
    train_ratings,
)

app = FastAPI(title="Mundial Betting API", version="1.1.0")


# --- Esquemas de Validación Pydantic ---
class Odds(BaseModel):
    home: float
    draw: float
    away: float
    over_25: Optional[float] = None
    under_25: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False
    odds: Optional[Odds] = None
    odds_format: str = "decimal"


class MatchData(BaseModel):
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    is_neutral: bool = False
    weight: float = 1.0
    match_date: date  # CORRECCIÓN: Autocasteo de Pydantic de str -> date


class TrainRequest(BaseModel):
    matches: List[MatchData]
    lambda_reg: float = 0.5
    half_life_days: int = 730
    reference_date: Optional[date] = None  # CORRECCIÓN: Autocasteo de str -> date


# --- Inicialización Automática ---
saved_model = load_trained_model()
if saved_model:
    global_params = saved_model.get("global_parameters", {})
    set_trained_gamma(global_params.get("home_advantage_gamma", 1.0))
    set_trained_rho(global_params.get("rho_correction", -0.13))


# --- Endpoints ---
@app.get("/")
def read_root():
    return {"status": "online", "teams_loaded": len(TEAMS)}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        result = predict_match(
            home_team=payload.home_team,
            away_team=payload.away_team,
            neutral=payload.neutral,
            odds=payload.odds.model_dump() if payload.odds else None,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/train")
def train(payload: TrainRequest) -> dict:
    try:
        formatted_matches = [
            SimpleNamespace(
                home_team=m.home_team,
                away_team=m.away_team,
                home_score=m.home_score,
                away_score=m.away_score,
                home_goals=m.home_score,
                away_goals=m.away_score,
                is_neutral=m.is_neutral,
                neutral=m.is_neutral,
                weight=m.weight,
                match_date=m.match_date,
                date=m.match_date,
            )
            for m in payload.matches
        ]

        result = train_ratings(
            formatted_matches,
            lambda_reg=payload.lambda_reg,
            half_life_days=payload.half_life_days,
            reference_date=payload.reference_date,
        )

        global_params = result.get("global_parameters", {})
        gamma = global_params.get("home_advantage_gamma", 1.0)
        rho = global_params.get("rho_correction", -0.13)

        save_trained_ratings(result["teams"], gamma=gamma, rho=rho)
        set_trained_gamma(gamma)
        set_trained_rho(rho)

        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
