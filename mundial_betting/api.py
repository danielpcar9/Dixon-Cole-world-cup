from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from types import SimpleNamespace
from datetime import date
import traceback
from slowapi import SlowAPILimiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

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

# Import de context_store con endpoints
try:
    from mundial_betting.context_store import (
        get_context_for_teams,
        set_team_context,
        add_h2h_match,
        load_team_contexts,
        load_h2h_records,
    )

    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False

app = FastAPI(title="Mundial Betting API", version="2.0.0")

# Rate limiting: 10 requests/min para /predict, 2/hora para /train
limiter = SlowAPILimiter(default_rate_limit="60/minute")
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ INICIALIZACIÓN ============

saved_model = load_trained_model()
if saved_model:
    gp = saved_model.get("global_parameters", {})
    set_trained_gamma(gp.get("home_advantage_gamma", 1.0))
    set_trained_rho(gp.get("rho_correction", -0.13))

if CONTEXT_AVAILABLE:
    load_team_contexts()
    load_h2h_records()


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )


# FRONTEND DESPUÉS DE ENDPOINTS para evitar que intercepte rutas API
app.frontend("/", directory="frontend")


# ============ PYDANTIC MODELS ============


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
    tournament_phase: Optional[str] = None


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


# Contexto models
class PlayerInput(BaseModel):
    name: str
    status: str = "available"
    impact: float = 1.0


class FormInput(BaseModel):
    last_results: List[str] = []
    goals_scored: int = 0
    goals_conceded: int = 0
    btts_count: int = 0
    clean_sheets: int = 0


class TeamContextInput(BaseModel):
    team_name: str
    form: FormInput
    key_players: List[PlayerInput] = []


class H2HInput(BaseModel):
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    date: str


# ============ ENDPOINTS API ============


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
@limiter.limit("30/minute")
def predict(payload: PredictRequest):
    try:
        result = predict_match(
            home_team=payload.home_team,
            away_team=payload.away_team,
            neutral=payload.neutral,
            odds=payload.odds.model_dump() if payload.odds else None,
            odds_format=payload.odds_format,  # type: ignore
            tournament_phase=payload.tournament_phase,
        )
        return result
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-auto-context")
@limiter.limit("30/minute")
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
            tournament_phase=payload.tournament_phase,
        )
        return result
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/train")
@limiter.limit("2/hour")
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

        gp = result.get("global_parameters", {})
        gamma = gp.get("home_advantage_gamma", 1.0)
        rho = gp.get("rho_correction", -0.13)

        save_trained_ratings(result["teams"], gamma=gamma, rho=rho)
        set_trained_gamma(gamma)
        set_trained_rho(rho)

        return result
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc))


# ============ CONTEXT ENDPOINTS ============


@app.get("/context/{team_name}")
def get_team_context_endpoint(team_name: str):
    """Obtiene contexto de un equipo si existe."""
    if not CONTEXT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Contexto no disponible")

    from mundial_betting.context_store import get_team_context

    ctx = get_team_context(team_name)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Sin datos de contexto")
    return ctx


@app.post("/context/{team_name}")
def save_team_context_endpoint(team_name: str, payload: TeamContextInput):
    """Guarda contexto de un equipo."""
    if not CONTEXT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Contexto no disponible")

    from mundial_betting.models import TeamContext, TeamForm, PlayerStatus
    from mundial_betting.context_store import set_team_context

    # Convertir a modelo interno
    form = TeamForm(
        last_results=payload.form.last_results,
        goals_scored=payload.form.goals_scored,
        goals_conceded=payload.form.goals_conceded,
        btts_count=payload.form.btts_count,
        clean_sheets=payload.form.clean_sheets,
    )

    ctx = TeamContext(
        team_name=team_name,
        form=form,
        key_players=[
            PlayerStatus(
                name=p.name,
                status=p.status,
                impact=p.impact,
            )
            for p in payload.key_players
        ],
    )

    set_team_context(ctx)
    return {"status": "ok", "team": team_name}


@app.post("/h2h")
def save_h2h_endpoint(payload: H2HInput):
    """Guarda un partido H2H."""
    if not CONTEXT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Contexto no disponible")

    add_h2h_match(
        team_a=payload.team_a,
        team_b=payload.team_b,
        goals_a=payload.goals_a,
        goals_b=payload.goals_b,
        match_date=payload.date,
    )
    return {"status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "teams_loaded": len(TEAMS),
        "gamma": get_trained_gamma(),
        "rho": get_trained_rho(),
        "context_available": CONTEXT_AVAILABLE,
    }
