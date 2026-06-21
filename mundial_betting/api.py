from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mundial_betting.context_store import (
    add_h2h_match,
    build_match_context,
    get_h2h,
    get_team_context,
    set_h2h,
    set_team_context,
)
from mundial_betting.data import TEAMS, get_display_name
from mundial_betting.dixon_coles import predict_match, set_trained_gamma, train_ratings
from mundial_betting.models import (
    H2HRecord,
    PlayerStatus,
    PredictRequest,
    TeamContext,
    TeamForm,
    TeamResponse,
    TrainRequest,
)

app = FastAPI(title="Mundial Betting API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/teams", response_model=dict[str, TeamResponse])
def teams() -> dict[str, TeamResponse]:
    return {
        name: TeamResponse(
            display_name=get_display_name(name),
            attack=rating.attack,
            defense=rating.defense,
            flag=rating.flag,
            host=rating.host,
        )
        for name, rating in TEAMS.items()
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, object]:
    try:
        result = predict_match(
            payload.home_team,
            payload.away_team,
            neutral=payload.neutral,
            odds=payload.odds,
            odds_format=payload.odds_format,
            context=payload.context,
        )
        from mundial_betting.data import get_ratings_metadata

        if not get_ratings_metadata().get("trained_at"):
            result["warning"] = (
                "Usando ratings previos hardcodeados. Llama a /train con datos históricos para predicciones calibradas."
            )
        return result
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train(payload: TrainRequest) -> dict[str, object]:
    try:
        result = train_ratings(
            payload.matches,
            lambda_reg=payload.lambda_reg,
            half_life_days=payload.half_life_days,
            reference_date=payload.reference_date,
        )
        from mundial_betting.data import save_trained_ratings

        save_trained_ratings(result["teams"])
        set_trained_gamma(result["global_parameters"]["home_advantage_gamma"])
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Optimization failed: {exc}"
        ) from exc


@app.get("/context/{team_name}")
def get_team_context_endpoint(team_name: str) -> dict:
    """Obtiene el contexto persistido de un equipo."""
    ctx = get_team_context(team_name)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"No context found for {team_name}")
    return ctx.model_dump()


@app.post("/context/{team_name}")
def update_team_context(team_name: str, payload: TeamContext) -> dict:
    """Actualiza o crea el contexto de un equipo."""
    set_team_context(payload)
    return {"status": "saved", "team": team_name}


@app.get("/h2h/{team_a}/{team_b}")
def get_h2h_endpoint(team_a: str, team_b: str) -> dict:
    """Obtiene el historial H2H entre dos equipos."""
    record = get_h2h(team_a, team_b)
    if not record:
        raise HTTPException(status_code=404, detail="No H2H record found")
    return record.model_dump()


@app.post("/h2h")
def add_h2h_endpoint(payload: dict) -> dict:
    """Añade un partido al historial H2H."""
    try:
        add_h2h_match(
            payload["team_a"],
            payload["team_b"],
            payload["goals_a"],
            payload["goals_b"],
            payload["date"],
            payload.get("tournament", ""),
        )
        return {"status": "saved"}
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc}") from exc


@app.post("/predict-auto-context")
def predict_auto_context(payload: PredictRequest) -> dict[str, object]:
    """Predice usando contexto persistido si está disponible."""
    auto_ctx = build_match_context(payload.home_team, payload.away_team)
    effective_context = payload.context or auto_ctx

    try:
        return predict_match(
            payload.home_team,
            payload.away_team,
            neutral=payload.neutral,
            odds=payload.odds,
            odds_format=payload.odds_format,
            context=effective_context,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
