from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from types import SimpleNamespace
from datetime import date

from fastapi.middleware.cors import CORSMiddleware

from mundial_betting.data import (
    TEAMS,
    normalize_team_name,
    save_trained_ratings,
    load_trained_model,
)

from mundial_betting.dixon_coles import (
    predict_match,
    set_trained_gamma,
    set_trained_rho,
    get_trained_gamma,
    get_trained_rho,
    train_ratings,
)

try:
    from mundial_betting.context_store import get_context_for_teams

    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False

app = FastAPI(title="Mundial Betting API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── FRONTEND PRIMERO (antes que los endpoints API) ──
app.frontend("/", directory="frontend")


class Odds(BaseModel):
    home: Optional[float] = None
    draw: Optional[float] = None
    away: Optional[float] = None
    over_25: Optional[float] = None
    under_25: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False
    odds: Optional[Odds] = None
    odds_format: str = "american"


class MatchData(BaseModel):
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    is_neutral: bool = False
    weight: float = 1.0
    match_date: date


class TrainRequest(BaseModel):
    matches: List[MatchData]
    lambda_reg: float = 0.5
    half_life_days: int = 730
    reference_date: Optional[date] = None


# --- Inicialización ---
saved_model = load_trained_model()
if saved_model:
    global_params = saved_model.get("global_parameters", {})
    set_trained_gamma(global_params.get("home_advantage_gamma", 1.0))
    set_trained_rho(global_params.get("rho_correction", -0.13))


# --- Endpoints API ---
@app.get("/status")
def read_status():
    return {
        "status": "online",
        "teams_loaded": len(TEAMS),
        "gamma": get_trained_gamma(),
        "rho": get_trained_rho(),
    }


@app.get("/teams")
def get_teams():
    if not TEAMS:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        name: {
            "display_name": name,
            "attack": round(stats.attack, 4),
            "defense": round(stats.defense, 4),
        }
        for name, stats in TEAMS.items()
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        result = predict_match(
            home_team=payload.home_team,
            away_team=payload.away_team,
            neutral=payload.neutral,
            odds=payload.odds.model_dump() if payload.odds else None,
            odds_format=payload.odds_format,  # type: ignore
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict-auto-context")
def predict_auto_context(payload: PredictRequest):
    try:
        home_norm = normalize_team_name(payload.home_team)
        away_norm = normalize_team_name(payload.away_team)

        context = None
        if CONTEXT_AVAILABLE:
            context = get_context_for_teams(home_norm, away_norm)

        result = predict_match(
            home_team=payload.home_team,
            away_team=payload.away_team,
            neutral=payload.neutral,
            odds=payload.odds.model_dump() if payload.odds else None,
            odds_format=payload.odds_format,  # type: ignore
            context=context,
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
                home_goals=m.home_score,
                away_goals=m.away_score,
                is_neutral=m.is_neutral,
                weight=m.weight,
                match_date=m.match_date,
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "teams_loaded": len(TEAMS),
        "gamma": get_trained_gamma(),
        "rho": get_trained_rho(),
        "context_available": CONTEXT_AVAILABLE,
    }
