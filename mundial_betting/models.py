from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator


OddsFormat = Literal["american", "decimal"]


class OddsInput(BaseModel):
    home: float | None = None
    draw: float | None = None
    away: float | None = None
    over_25: float | None = None
    under_25: float | None = None
    btts_yes: float | None = None
    btts_no: float | None = None


class MatchContext(BaseModel):
    """Contexto cualitativo para ajustar predicciones con tendencias, H2H y rachas."""

    h2h_home_wins: float = 0.0
    h2h_draws: float = 0.0
    h2h_away_wins: float = 0.0
    h2h_btts_count: float = 0.0
    h2h_total: float = 0.0

    home_btts_streak: int = 0
    away_btts_streak: int = 0
    home_clean_sheets_last5: int = 0
    away_clean_sheets_last5: int = 0

    home_key_players_available: float = Field(default=1.0, ge=0.0, le=1.0)
    away_key_players_available: float = Field(default=1.0, ge=0.0, le=1.0)

    home_late_goals_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    away_late_goals_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    def h2h_home_win_rate(self) -> float:
        if self.h2h_total == 0:
            return 0.5
        return self.h2h_home_wins / self.h2h_total

    def h2h_btts_rate(self) -> float:
        if self.h2h_total == 0:
            return 0.5
        return self.h2h_btts_count / self.h2h_total


class PlayerStatus(BaseModel):
    name: str
    status: Literal["available", "doubtful", "injured", "suspended"]
    impact: float = Field(default=1.0, ge=0.0, le=1.0)


class TeamForm(BaseModel):
    """Resultados de los últimos N partidos oficiales del equipo."""

    last_results: list[Literal["W", "D", "L"]] = Field(
        default_factory=list, max_length=10
    )
    goals_scored: int = 0
    goals_conceded: int = 0
    btts_count: int = 0
    clean_sheets: int = 0

    @computed_field
    @property
    def win_rate(self) -> float:
        if not self.last_results:
            return 0.5
        return self.last_results.count("W") / len(self.last_results)

    @computed_field
    @property
    def btts_rate(self) -> float:
        if not self.last_results:
            return 0.5
        return self.btts_count / len(self.last_results)


class TeamContext(BaseModel):
    """Contexto persistente por equipo: forma, lesiones y metadatos."""

    team_name: str
    form: TeamForm = Field(default_factory=TeamForm)
    key_players: list[PlayerStatus] = Field(default_factory=list)
    last_updated: date = Field(default_factory=date.today)

    def availability_factor(self) -> float:
        """Factor multiplicativo de ataque basado en jugadores disponibles."""
        if not self.key_players:
            return 1.0
        total_impact = sum(
            p.impact for p in self.key_players if p.status == "available"
        )
        max_impact = sum(p.impact for p in self.key_players)
        return 0.7 + 0.3 * (total_impact / max_impact) if max_impact > 0 else 1.0


class H2HRecord(BaseModel):
    """Registro histórico entre dos equipos."""

    team_a: str
    team_b: str
    matches: list[dict] = Field(default_factory=list)

    def win_rate_for(self, team: str) -> float:
        if not self.matches:
            return 0.5
        wins = sum(
            1
            for m in self.matches
            if (m["goals_a"] > m["goals_b"] and team == self.team_a)
            or (m["goals_b"] > m["goals_a"] and team == self.team_b)
        )
        return wins / len(self.matches)

    def btts_rate(self) -> float:
        if not self.matches:
            return 0.5
        btts = sum(1 for m in self.matches if m["goals_a"] > 0 and m["goals_b"] > 0)
        return btts / len(self.matches)

    def total_matches(self) -> int:
        return len(self.matches)


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False
    odds_format: OddsFormat = "american"
    odds: OddsInput | None = None
    context: MatchContext | None = None


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
