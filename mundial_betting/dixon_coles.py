from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import exp, factorial, log

import numpy as np
from scipy.optimize import minimize

from mundial_betting.data import TEAMS, TeamRating, get_team, normalize_team_name
from mundial_betting.models import MatchData, OddsFormat, OddsInput

MAX_GOALS = 10
DEFAULT_RHO = -0.13
DEFAULT_GAMMA = 1.15

_TRAINED_GAMMA: float = DEFAULT_GAMMA


@dataclass(frozen=True)
class ExpectedGoals:
    home: float
    away: float
    home_attack_multiplier: float
    home_defense_multiplier: float


def poisson_pmf(goals: int, expected_goals: float) -> float:
    return exp(-expected_goals) * expected_goals**goals / factorial(goals)


def get_trained_gamma() -> float:
    return _TRAINED_GAMMA


def set_trained_gamma(gamma: float) -> None:
    global _TRAINED_GAMMA
    _TRAINED_GAMMA = gamma


def tau_correction(home_goals: int, away_goals: int, lmbda: float, mu: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        correction = 1 - lmbda * mu * rho
    elif home_goals == 0 and away_goals == 1:
        correction = 1 + lmbda * rho
    elif home_goals == 1 and away_goals == 0:
        correction = 1 + mu * rho
    elif home_goals == 1 and away_goals == 1:
        correction = 1 - rho
    else:
        correction = 1.0
    return min(2.0, max(0.01, correction))


def expected_goals(
    home_team: str,
    away_team: str,
    *,
    neutral: bool = False,
    gamma: float = DEFAULT_GAMMA,
    teams: dict[str, TeamRating] | None = None,
) -> ExpectedGoals:
    ratings = teams or TEAMS
    home = ratings[normalize_team_name(home_team)]
    away = ratings[normalize_team_name(away_team)]

    home_advantage = 1.0 if neutral else gamma

    return ExpectedGoals(
        home=home.attack * away.defense * home_advantage,
        away=away.attack * home.defense,
        home_attack_multiplier=home_advantage,
        home_defense_multiplier=1.0,
    )


def score_matrix(
    home_team: str,
    away_team: str,
    *,
    neutral: bool = False,
    rho: float = DEFAULT_RHO,
    max_goals: int = MAX_GOALS,
    gamma: float = DEFAULT_GAMMA,
    teams: dict[str, TeamRating] | None = None,
) -> tuple[np.ndarray, ExpectedGoals]:
    xg = expected_goals(
        home_team,
        away_team,
        neutral=neutral,
        gamma=gamma,
        teams=teams,
    )
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            correction = tau_correction(home_goals, away_goals, xg.home, xg.away, rho)
            matrix[home_goals, away_goals] = (
                correction
                * poisson_pmf(home_goals, xg.home)
                * poisson_pmf(away_goals, xg.away)
            )

    matrix /= matrix.sum()
    return matrix, xg


def market_probabilities(matrix: np.ndarray) -> dict[str, float]:
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())

    over_25 = 0.0
    btts_yes = 0.0
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            probability = matrix[home_goals, away_goals]
            if home_goals + away_goals > 2.5:
                over_25 += probability
            if home_goals > 0 and away_goals > 0:
                btts_yes += probability

    return {
        "home": home_win,
        "draw": draw,
        "away": away_win,
        "over_25": float(over_25),
        "under_25": float(1 - over_25),
        "btts_yes": float(btts_yes),
        "btts_no": float(1 - btts_yes),
    }


def top_exact_scores(matrix: np.ndarray, limit: int = 10) -> list[dict[str, float | int]]:
    scores = [
        {"home_goals": home, "away_goals": away, "probability": float(matrix[home, away])}
        for home in range(matrix.shape[0])
        for away in range(matrix.shape[1])
    ]
    return sorted(scores, key=lambda item: item["probability"], reverse=True)[:limit]


def implied_probability(odds: float, odds_format: OddsFormat) -> float:
    if odds_format == "decimal":
        if odds <= 1:
            raise ValueError("Decimal odds must be greater than 1")
        return 1 / odds
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def remove_vig(*probabilities: float) -> list[float]:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("At least one implied probability is required")
    return [probability / total for probability in probabilities]


def edge_report(model_probs: dict[str, float], odds: OddsInput, odds_format: OddsFormat) -> dict[str, dict[str, float | str]]:
    output: dict[str, dict[str, float | str]] = {}

    if odds.home is not None and odds.draw is not None and odds.away is not None:
        no_vig = remove_vig(
            implied_probability(odds.home, odds_format),
            implied_probability(odds.draw, odds_format),
            implied_probability(odds.away, odds_format),
        )
        for market, market_prob in zip(("home", "draw", "away"), no_vig, strict=True):
            output[market] = _edge_item(model_probs[market], market_prob)

    two_way_markets = [
        ("over_25", "under_25", odds.over_25, odds.under_25),
        ("btts_yes", "btts_no", odds.btts_yes, odds.btts_no),
    ]
    for first, second, first_odds, second_odds in two_way_markets:
        if first_odds is None or second_odds is None:
            continue
        first_market, second_market = remove_vig(
            implied_probability(first_odds, odds_format),
            implied_probability(second_odds, odds_format),
        )
        output[first] = _edge_item(model_probs[first], first_market)
        output[second] = _edge_item(model_probs[second], second_market)

    return output


def _edge_item(model_probability: float, market_probability: float) -> dict[str, float | str]:
    edge = model_probability - market_probability
    if edge >= 0.04:
        pick = "BET"
    elif edge <= -0.05:
        pick = "FADE"
    else:
        pick = "SKIP"
    return {
        "model_probability": round(model_probability, 6),
        "market_probability_no_vig": round(market_probability, 6),
        "edge": round(edge, 6),
        "pick": pick,
    }


def predict_match(
    home_team: str,
    away_team: str,
    *,
    neutral: bool = False,
    gamma: float | None = None,
    odds: OddsInput | None = None,
    odds_format: OddsFormat = "american",
) -> dict[str, object]:
    get_team(home_team)
    get_team(away_team)

    effective_gamma = gamma if gamma is not None else get_trained_gamma()
    matrix, xg = score_matrix(
        home_team,
        away_team,
        neutral=neutral,
        gamma=effective_gamma,
    )
    markets = market_probabilities(matrix)
    response: dict[str, object] = {
        "home_team": normalize_team_name(home_team),
        "away_team": normalize_team_name(away_team),
        "expected_goals": {
            "home": round(xg.home, 4),
            "away": round(xg.away, 4),
            "home_attack_multiplier": xg.home_attack_multiplier,
            "home_defense_multiplier": xg.home_defense_multiplier,
        },
        "markets": {key: round(value, 6) for key, value in markets.items()},
        "exact_scores": top_exact_scores(matrix),
    }
    if odds:
        response["edges"] = edge_report(markets, odds, odds_format)
    return response


def time_weight(
    match_date: date | None,
    reference_date: date,
    half_life_days: float,
) -> float:
    """
    Peso exponencial decreciente. Un partido a half_life_days de distancia
    pesa la mitad. Si no hay fecha, devuelve 1.0.
    """
    if match_date is None:
        return 1.0
    days_diff = (reference_date - match_date).days
    if days_diff < 0:
        return 1.0
    return 0.5 ** (days_diff / half_life_days)


def negative_log_likelihood(
    params: np.ndarray,
    matches: list[MatchData],
    team_indices: dict[str, int],
    lambda_reg: float = 0.5,
    half_life_days: float = 730.0,
    reference_date: date | None = None,
) -> float:
    n_teams = len(team_indices)
    alpha = params[:n_teams]
    beta = params[n_teams : 2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]
    log_likelihood = 0.0

    ref_date = reference_date or date.today()

    for match in matches:
        home_idx = team_indices[normalize_team_name(match.home_team)]
        away_idx = team_indices[normalize_team_name(match.away_team)]
        home_advantage = 1.0 if match.is_neutral else gamma
        lmbda = alpha[home_idx] * beta[away_idx] * home_advantage
        mu = alpha[away_idx] * beta[home_idx]

        if lmbda <= 0 or mu <= 0:
            return 1e10

        correction = tau_correction(match.home_goals, match.away_goals, lmbda, mu, rho)
        if correction <= 0:
            return 1e10

        temporal_weight = time_weight(match.match_date, ref_date, half_life_days)
        effective_weight = match.weight * temporal_weight

        log_likelihood += effective_weight * (
            log(correction)
            - lmbda
            + match.home_goals * log(lmbda)
            - mu
            + match.away_goals * log(mu)
        )

    reg_term = lambda_reg * np.sum((alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(-log_likelihood + reg_term)


def train_ratings(
    matches: list[MatchData],
    lambda_reg: float = 0.5,
    half_life_days: float = 730.0,
    reference_date: date | None = None,
) -> dict[str, object]:
    teams = sorted(
        {
            normalize_team_name(match.home_team)
            for match in matches
        }
        | {
            normalize_team_name(match.away_team)
            for match in matches
        }
    )
    if len(teams) < 2:
        raise ValueError("At least two teams are required")

    team_indices = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)
    initial_params = np.concatenate(
        [
            np.ones(n_teams),
            np.ones(n_teams),
            [1.15],
            [-0.10],
        ]
    )
    bounds = (
        [(0.05, 5.0)] * n_teams
        + [(0.05, 5.0)] * n_teams
        + [(0.5, 2.5)]
        + [(-0.25, 0.25)]
    )

    constraints = {"type": "eq", "fun": lambda params: np.sum(params[:n_teams]) - n_teams}
    result = minimize(
        negative_log_likelihood,
        initial_params,
        args=(matches, team_indices, lambda_reg, half_life_days, reference_date),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "disp": False},
    )

    if not result.success:
        raise RuntimeError(str(result.message))

    fitted_alpha = result.x[:n_teams]
    fitted_beta = result.x[n_teams : 2 * n_teams]
    gamma = float(result.x[2 * n_teams])
    rho = float(result.x[2 * n_teams + 1])

    ratings = {
        team: {
            "attack": round(float(fitted_alpha[index]), 4),
            "defense": round(float(fitted_beta[index]), 4),
        }
        for team, index in team_indices.items()
    }
    return {
        "global_parameters": {
            "home_advantage_gamma": round(gamma, 4),
            "rho_correction": round(rho, 4),
            "negative_log_likelihood": round(float(result.fun), 4),
            "lambda_reg": lambda_reg,
            "half_life_days": half_life_days,
            "reference_date": str(reference_date) if reference_date else str(date.today()),
        },
        "teams": ratings,
    }