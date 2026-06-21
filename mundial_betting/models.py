from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


OddsFormat = Literal["american", "decimal"]


class OddsInput(BaseModel):
    home: float | None = None
    draw: float | None = None
    away: float | None = None
    over_25: float | None = None
    under_25: float | None = None
    btts_yes: float | None = None
    btts_no: float | None = None


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False
    odds_format: OddsFormat = "american"
    odds: OddsInput | None = None


class MatchData(BaseModel):
    home_team: str
    away_team: str
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    is_neutral: bool = False
    weight: float = Field(default=1.0, gt=0)
    match_date: date | None = None

    @model_validator(mode="after")
    def prevent_same_team(self) -> "MatchData":
        if self.home_team.strip().casefold() == self.away_team.strip().casefold():
            raise ValueError("home_team and away_team must be different")
        return self


class TrainRequest(BaseModel):
    matches: list[MatchData] = Field(min_length=1)
    lambda_reg: float = Field(default=0.5, ge=0.0)
    half_life_days: float = Field(default=730.0, gt=0)
    reference_date: date | None = None


class TeamResponse(BaseModel):
    display_name: str
    attack: float
    defense: float
    flag: str
    host: bool