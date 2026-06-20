from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mundial_betting.data import TEAMS, get_display_name
from mundial_betting.dixon_coles import predict_match, train_ratings
from mundial_betting.models import PredictRequest, TeamResponse, TrainRequest

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
        return predict_match(
            payload.home_team,
            payload.away_team,
            neutral=payload.neutral,
            odds=payload.odds,
            odds_format=payload.odds_format,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train(payload: TrainRequest) -> dict[str, object]:
    try:
        return train_ratings(payload.matches)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
