from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, factorial, log

import numpy as np
from scipy.optimize import minimize

from mundial_betting.data import TEAMS, TeamRating, get_team, normalize_team_name
from mundial_betting.models import MatchContext, MatchData, OddsFormat, OddsInput

MAX_GOALS = 15
DEFAULT_RHO = -0.13
DEFAULT_GAMMA = 1.15

# Estados globales para parámetros entrenados
_TRAINED_GAMMA: float = DEFAULT_GAMMA
_TRAINED_RHO: float = DEFAULT_RHO

# H7: Multiplicadores por tipo de competición para ajustar expected goals
# Los torneos de élite tienen menos goles debido a mayor presión defensiva
TOURNAMENT_MULTIPLIERS: dict[str, float] = {
    "FIFA World Cup": 0.90,
    "FIFA World Cup qualification": 0.95,
    "Copa America": 0.92,
    "UEFA Euro": 0.92,
    "UEFA Euro qualification": 0.94,
    "Friendly": 1.10,
    "International Friendly": 1.10,
    "Nations League": 0.96,
    "Gold Cup": 0.93,
    "AFC Asian Cup": 0.91,
    "Africa Cup of Nations": 0.92,
}


@dataclass(frozen=True)
class ExpectedGoals:
    home: float
    away: float
    home_attack_multiplier: float
    home_defense_multiplier: float


@dataclass(frozen=True)
class ContextAdjustments:
    home_attack: float
    away_attack: float
    home_defense: float
    away_defense: float
    h2h_boost_applied: bool
    btts_boost_applied: bool
    key_players_boost_applied: bool
    clean_sheet_boost_applied: bool


def poisson_pmf(goals: int, expected_goals: float) -> float:
    return exp(-expected_goals) * expected_goals**goals / factorial(goals)


def get_trained_gamma() -> float:
    return _TRAINED_GAMMA


def set_trained_gamma(gamma: float) -> None:
    global _TRAINED_GAMMA
    _TRAINED_GAMMA = gamma


def get_trained_rho() -> float:
    return _TRAINED_RHO


def set_trained_rho(rho: float) -> None:
    global _TRAINED_RHO
    _TRAINED_RHO = rho


def tau_correction(
    home_goals: int, away_goals: int, lmbda: float, mu: float, rho: float
) -> float:
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


def _dynamic_max_goals(xg_home: float, xg_away: float, min_goals: int = 15) -> int:
    from scipy.stats import poisson

    max_h = min_goals
    max_a = min_goals
    for k in range(min_goals, 50):
        if poisson.cdf(k, xg_home) > 0.9995:
            max_h = k
            break
    for k in range(min_goals, 50):
        if poisson.cdf(k, xg_away) > 0.9995:
            max_a = k
            break
    return max(max_h, max_a)


def expected_goals(
    home_team: str,
    away_team: str,
    *,
    neutral: bool = False,
    gamma: float = DEFAULT_GAMMA,
    teams: dict[str, TeamRating] | None = None,
    tournament: str | None = None,
) -> ExpectedGoals:
    """Calcula expected goals con ajuste por competición.
    
    Args:
        home_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
        neutral: Si es partido en campo neutral
        gamma: Ventaja de localía
        teams: Diccionario de ratings (opcional, usa TEAMS global si None)
        tournament: Nombre del torneo para aplicar multiplicador de competición
    
    Returns:
        ExpectedGoals con los goles esperados ajustados
    """
    ratings = teams or TEAMS
    home_key = normalize_team_name(home_team)
    away_key = normalize_team_name(away_team)

    if home_key not in ratings:
        raise ValueError(f"Equipo no encontrado: {home_team} (normalizado: {home_key})")
    if away_key not in ratings:
        raise ValueError(f"Equipo no encontrado: {away_team} (normalizado: {away_key})")

    home = ratings[home_key]
    away = ratings[away_key]

    home_advantage = 1.0 if neutral else gamma
    
    # H7: Aplicar multiplicador por tipo de competición
    comp_mult = 1.0
    if tournament:
        comp_mult = TOURNAMENT_MULTIPLIERS.get(tournament, 1.0)

    raw_home = home.attack * away.defense * home_advantage * comp_mult
    raw_away = away.attack * home.defense * comp_mult

    return ExpectedGoals(
        home=min(4.5, max(0.01, raw_home)),
        away=min(4.5, max(0.01, raw_away)),
        home_attack_multiplier=home_advantage * comp_mult,
        home_defense_multiplier=1.0,
    )


def _score_matrix_from_xg(
    xg: ExpectedGoals, rho: float = DEFAULT_RHO, max_goals: int | None = None
) -> np.ndarray:
    """Construye matriz de probabilidades a partir de expected goals ya ajustados."""
    effective_max = (
        max_goals if max_goals is not None else _dynamic_max_goals(xg.home, xg.away)
    )
    matrix = np.zeros((effective_max + 1, effective_max + 1), dtype=float)

    for home_goals in range(effective_max + 1):
        for away_goals in range(effective_max + 1):
            correction = tau_correction(home_goals, away_goals, xg.home, xg.away, rho)
            matrix[home_goals, away_goals] = (
                correction
                * poisson_pmf(home_goals, xg.home)
                * poisson_pmf(away_goals, xg.away)
            )

    total = float(matrix.sum())
    if total <= 0:
        raise ValueError(
            f"Score matrix has zero total probability for xG=({xg.home:.3f}, {xg.away:.3f})"
        )
    if total < 0.95:
        import logging

        logging.warning(
            "Score matrix truncation: sum=%.4f for xG=(%.3f, %.3f). "
            "Increase MAX_GOALS if this occurs frequently.",
            total,
            xg.home,
            xg.away,
        )
    matrix /= total
    return matrix


def score_matrix(
    home_team: str,
    away_team: str,
    *,
    neutral: bool = False,
    rho: float | None = None,
    max_goals: int | None = None,
    gamma: float = DEFAULT_GAMMA,
    teams: dict[str, TeamRating] | None = None,
    tournament: str | None = None,
) -> tuple[np.ndarray, ExpectedGoals]:
    """Construye matriz de probabilidades con ajuste por competición.
    
    Args:
        home_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
        neutral: Si es partido en campo neutral
        rho: Parámetro de corrección tau (usa el entrenado si None)
        max_goals: Máximo de goles a considerar
        gamma: Ventaja de localía
        teams: Diccionario de ratings (opcional)
        tournament: Nombre del torneo para aplicar multiplicador
    
    Returns:
        Tupla con la matriz de probabilidades y ExpectedGoals
    """
    xg = expected_goals(
        home_team,
        away_team,
        neutral=neutral,
        gamma=gamma,
        teams=teams,
        tournament=tournament,
    )
    target_rho = rho if rho is not None else get_trained_rho()
    matrix = _score_matrix_from_xg(xg, rho=target_rho, max_goals=max_goals)
    return matrix, xg


def apply_context_adjustments(
    xg: ExpectedGoals,
    ctx: MatchContext,
) -> tuple[ExpectedGoals, ContextAdjustments]:
    adj = {
        "home_attack": 1.0,
        "away_attack": 1.0,
        "home_defense": 1.0,
        "away_defense": 1.0,
    }
    flags = {
        "h2h_boost_applied": False,
        "btts_boost_applied": False,
        "key_players_boost_applied": False,
        "clean_sheet_boost_applied": False,
    }

    if ctx.h2h_total >= 3:
        h2h_home_wr = ctx.h2h_home_win_rate()
        h2h_boost = (h2h_home_wr - 0.5) * 0.2
        adj["home_attack"] *= 1.0 + h2h_boost
        adj["away_attack"] *= 1.0 - h2h_boost
        flags["h2h_boost_applied"] = True

    if ctx.home_btts_streak >= 3 and ctx.away_btts_streak >= 3:
        adj["home_defense"] *= 0.85
        adj["away_defense"] *= 0.85
        flags["btts_boost_applied"] = True

    if ctx.home_clean_sheets_last5 >= 3:
        adj["home_defense"] *= 1.15
        flags["clean_sheet_boost_applied"] = True
    if ctx.away_clean_sheets_last5 >= 3:
        adj["away_defense"] *= 1.15
        flags["clean_sheet_boost_applied"] = True

    if ctx.home_key_players_available < 1.0 or ctx.away_key_players_available < 1.0:
        adj["home_attack"] *= ctx.home_key_players_available
        adj["away_attack"] *= ctx.away_key_players_available
        flags["key_players_boost_applied"] = True

    adjusted = ExpectedGoals(
        home=xg.home * adj["home_attack"] / adj["away_defense"],
        away=xg.away * adj["away_attack"] / adj["home_defense"],
        home_attack_multiplier=xg.home_attack_multiplier * adj["home_attack"],
        home_defense_multiplier=xg.home_defense_multiplier * adj["home_defense"],
    )

    adjustments = ContextAdjustments(
        home_attack=adj["home_attack"],
        away_attack=adj["away_attack"],
        home_defense=adj["home_defense"],
        away_defense=adj["away_defense"],
        **flags,
    )

    return adjusted, adjustments


def market_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Cálculo de mercados optimizado y vectorizado al 100% con NumPy."""
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())

    n_rows, n_cols = matrix.shape
    r, c = np.indices((n_rows, n_cols))

    over_25 = float(matrix[r + c > 2.5].sum())
    btts_yes = float(matrix[1:, 1:].sum())

    return {
        "home": home_win,
        "draw": draw,
        "away": away_win,
        "over_25": over_25,
        "under_25": float(1.0 - over_25),
        "btts_yes": btts_yes,
        "btts_no": float(1.0 - btts_yes),
    }


def top_exact_scores(
    matrix: np.ndarray, limit: int = 10
) -> list[dict[str, float | int]]:
    scores = [
        {
            "home_goals": home,
            "away_goals": away,
            "probability": float(matrix[home, away]),
        }
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


def _to_decimal_odds(odds: float, odds_format: OddsFormat) -> float:
    if odds_format == "decimal":
        return odds
    if odds > 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


# FIX: edge_report ahora acepta OddsInput o dict
def edge_report(
    model_probs: dict[str, float],
    odds,
    odds_format: OddsFormat,
) -> dict[str, dict[str, float | str | None]]:
    """Acepta odds como OddsInput, dict o SimpleNamespace para compatibilidad."""
    # Normalizar a dict
    if hasattr(odds, "model_dump"):
        odds_dict = odds.model_dump()
    elif hasattr(odds, "__dict__"):
        odds_dict = odds.__dict__
    else:
        odds_dict = odds  # ya es dict

    output: dict[str, dict[str, float | str | None]] = {}

    if (
        odds_dict.get("home") is not None
        and odds_dict.get("draw") is not None
        and odds_dict.get("away") is not None
    ):
        raw = [odds_dict["home"], odds_dict["draw"], odds_dict["away"]]
        no_vig = remove_vig(*[implied_probability(o, odds_format) for o in raw])
        decimals = [_to_decimal_odds(o, odds_format) for o in raw]
        for market, market_prob, dec in zip(
            ("home", "draw", "away"), no_vig, decimals, strict=True
        ):
            output[market] = _edge_item(model_probs[market], market_prob, dec)

    two_way_markets = [
        ("over_25", "under_25", odds_dict.get("over_25"), odds_dict.get("under_25")),
        ("btts_yes", "btts_no", odds_dict.get("btts_yes"), odds_dict.get("btts_no")),
    ]
    for first, second, first_odds, second_odds in two_way_markets:
        if first_odds is None or second_odds is None:
            continue
        first_prob, second_prob = remove_vig(
            implied_probability(first_odds, odds_format),
            implied_probability(second_odds, odds_format),
        )
        output[first] = _edge_item(
            model_probs[first], first_prob, _to_decimal_odds(first_odds, odds_format)
        )
        output[second] = _edge_item(
            model_probs[second], second_prob, _to_decimal_odds(second_odds, odds_format)
        )

    return output


def _edge_item(
    model_probability: float,
    market_probability: float,
    decimal_odds: float | None = None,
) -> dict[str, float | str | None]:
    edge = model_probability - market_probability
    if edge >= 0.04:
        pick = "BET"
    elif edge <= -0.05:
        pick = "FADE"
    else:
        pick = "SKIP"

    kelly: float | None = None
    if decimal_odds is not None and decimal_odds > 1.0:
        b = decimal_odds - 1.0
        k = (b * model_probability - (1.0 - model_probability)) / b
        kelly = round(max(0.0, k), 4)

    return {
        "model_probability": round(model_probability, 6),
        "market_probability_no_vig": round(market_probability, 6),
        "edge": round(edge, 6),
        "kelly_fraction": kelly,
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
    context: MatchContext | None = None,
    tournament: str | None = None,
    tournament_phase: str | None = None,
) -> dict[str, object]:
    """Predice resultado de partido con ajustes por competición y fase.
    
    Args:
        home_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
        neutral: Si es partido en campo neutral
        gamma: Ventaja de localía (usa el entrenado si None)
        odds: Cuotas del mercado para cálculo de edges
        odds_format: Formato de las cuotas
        context: Contexto adicional del partido (H2H, lesiones, etc.)
        tournament: Nombre del torneo para ajuste de expected goals
        tournament_phase: Fase del torneo ('group', 'round_of_16', 'quarter', 
                         'semi', 'final') para ajuste elite
    
    Returns:
        Diccionario con predicciones completas del partido
    """
    get_team(home_team)
    get_team(away_team)

    effective_gamma = gamma if gamma is not None else get_trained_gamma()
    effective_rho = get_trained_rho()

    xg_base = expected_goals(
        home_team,
        away_team,
        neutral=neutral,
        gamma=effective_gamma,
        tournament=tournament,
    )

    if context is not None:
        xg, adjustments = apply_context_adjustments(xg_base, context)
        context_meta = {
            "adjustments": {
                "home_attack_multiplier": round(adjustments.home_attack, 4),
                "away_attack_multiplier": round(adjustments.away_attack, 4),
                "home_defense_multiplier": round(adjustments.home_defense, 4),
                "away_defense_multiplier": round(adjustments.away_defense, 4),
                "h2h_boost_applied": adjustments.h2h_boost_applied,
                "btts_boost_applied": adjustments.btts_boost_applied,
                "key_players_boost_applied": adjustments.key_players_boost_applied,
                "clean_sheet_boost_applied": adjustments.clean_sheet_boost_applied,
            },
            "h2h_home_win_rate": round(context.h2h_home_win_rate(), 4)
            if context.h2h_total > 0
            else None,
            "h2h_btts_rate": round(context.h2h_btts_rate(), 4)
            if context.h2h_total > 0
            else None,
        }
        matrix = _score_matrix_from_xg(xg, rho=effective_rho)
    else:
        xg = xg_base
        context_meta = None
        matrix, _ = score_matrix(
            home_team,
            away_team,
            neutral=neutral,
            rho=effective_rho,
            gamma=effective_gamma,
            tournament=tournament,
        )

    markets = market_probabilities(matrix)
    
    # H8: Ajuste para fases eliminatorias de élite
    if tournament_phase and tournament_phase in {"round_of_16", "quarter", "semi", "final"}:
        markets = _apply_elite_adjustment(markets, tournament_phase)

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
        "probabilities": {key: round(value, 6) for key, value in markets.items()},
        "exact_scores": top_exact_scores(matrix),
    }
    if context_meta is not None:
        response["context"] = context_meta
    if odds:
        response["edges"] = edge_report(markets, odds, odds_format)
    return response


def time_weight(
    match_date: date | None,
    reference_date: date,
    half_life_days: float,
) -> float:
    if match_date is None:
        return 1.0
    days_diff = (reference_date - match_date).days
    if days_diff < 0:
        return 1.0
    return 0.5 ** (days_diff / half_life_days)


def _apply_elite_adjustment(
    markets: dict[str, float],
    tournament_phase: str,
) -> dict[str, float]:
    """Ajusta probabilidades para fases eliminatorias de élite.
    
    En octavos de final en adelante, los equipos juegan más conservador,
    reduciendo la probabilidad de Over 2.5 y BTTS.
    
    Args:
        markets: Diccionario de probabilidades del mercado
        tournament_phase: Fase del torneo ('round_of_16', 'quarter', 'semi', 'final')
    
    Returns:
        Diccionario con probabilidades ajustadas y re-normalizadas
    """
    adjustment_factors = {
        "round_of_16": {"over_25": 0.92, "btts_yes": 0.94},
        "quarter": {"over_25": 0.88, "btts_yes": 0.90},
        "semi": {"over_25": 0.85, "btts_yes": 0.87},
        "final": {"over_25": 0.80, "btts_yes": 0.82},
    }
    
    if tournament_phase not in adjustment_factors:
        return markets
    
    factors = adjustment_factors[tournament_phase]
    adjusted = dict(markets)
    
    for market, factor in factors.items():
        if market in adjusted:
            adjusted[market] *= factor
            # Re-normalizar el complemento
            if market == "over_25":
                adjusted["under_25"] = 1.0 - adjusted["over_25"]
            elif market == "btts_yes":
                adjusted["btts_no"] = 1.0 - adjusted["btts_yes"]
    
    return adjusted


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

        raw_lmbda = alpha[home_idx] * beta[away_idx] * home_advantage
        raw_mu = alpha[away_idx] * beta[home_idx]

        lmbda = min(4.5, max(0.01, raw_lmbda))
        mu = min(4.5, max(0.01, raw_mu))

        correction = tau_correction(match.home_goals, match.away_goals, lmbda, mu, rho)
        if correction <= 0:
            return 1e10

        temporal_weight = time_weight(match.match_date, ref_date, half_life_days)
        effective_weight = match.weight * temporal_weight

        log_likelihood += effective_weight * (
            log(correction)
            - lmbda
            + match.home_goals * log(lmbda)
            - log(factorial(match.home_goals))
            - mu
            + match.away_goals * log(mu)
            - log(factorial(match.away_goals))
        )

    reg_term = lambda_reg * np.sum((alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(-log_likelihood + reg_term)


def train_ratings(
    matches: list[MatchData],
    lambda_reg: float = 0.5,
    half_life_days: float = 730.0,
    reference_date: date | None = None,
    previous_ratings: dict[str, dict] | None = None,
    rho_bounds: tuple[float, float] = (-0.20, -0.05),
    ftol: float = 1e-6,
) -> dict[str, object]:
    """Entrena los ratings Dixon-Coles con optimización en dos fases y warm-start.
    
    Implementa la Recomendación Prioritaria #1: optimización por bloques que:
    (a) carga parámetros previos como warm-start, 
    (b) ejecuta SLSQP en dos fases separando la optimización de rho, 
    (c) relaja ftol a 1e-6 para mayor velocidad.
    
    Args:
        matches: Lista de partidos para entrenar
        lambda_reg: Regularización L2 (se aplica adaptativamente según H7)
        half_life_days: Vida media para ponderación temporal
        reference_date: Fecha de referencia para ponderación
        previous_ratings: Diccionario de ratings previos para warm-start.
                         Formato: {"team_name": {"attack": x, "defense": y}, 
                                   "__gamma__": z, "__rho__": w}
        rho_bounds: Tupla (min, max) para el parámetro rho. Default (-0.20, -0.05)
        ftol: Tolerancia de convergencia. Default 1e-6
    
    Returns:
        Diccionario con ratings entrenados y parámetros globales
    """
    teams = sorted(
        {normalize_team_name(match.home_team) for match in matches}
        | {normalize_team_name(match.away_team) for match in matches}
    )
    if len(teams) < 2:
        raise ValueError("At least two teams are required")

    team_indices = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)
    
    # === WARM-START: Usar parámetros previos si existen ===
    if previous_ratings:
        alpha_init = np.array([
            previous_ratings.get(t, {}).get("attack", 1.0)
            for t in teams
        ])
        beta_init = np.array([
            previous_ratings.get(t, {}).get("defense", 1.0)
            for t in teams
        ])
        gamma_init = previous_ratings.get("__gamma__", 1.15)
        rho_init = previous_ratings.get("__rho__", -0.10)
    else:
        alpha_init = np.ones(n_teams)
        beta_init = np.ones(n_teams)
        gamma_init = 1.15
        rho_init = -0.10
    
    # === FASE 1: Optimizar alpha, beta, gamma con rho fijo ===
    params_phase1 = np.concatenate([alpha_init, beta_init, [gamma_init]])
    bounds_phase1 = (
        [(0.05, 5.0)] * n_teams
        + [(0.05, 5.0)] * n_teams
        + [(0.5, 2.5)]
    )
    
    def nll_phase1(params: np.ndarray) -> float:
        """NLL con rho fijo en fase 1."""
        full_params = np.append(params, rho_init)
        return negative_log_likelihood(
            full_params, matches, team_indices,
            lambda_reg, half_life_days, reference_date
        )
    
    result1 = minimize(
        nll_phase1,
        params_phase1,
        method="SLSQP",
        bounds=bounds_phase1,
        constraints=[
            {"type": "eq", "fun": lambda p: np.sum(p[:n_teams]) - n_teams},
            {"type": "eq", "fun": lambda p: np.sum(p[n_teams:2*n_teams]) - n_teams},
        ],
        options={"maxiter": 100, "ftol": ftol, "disp": False},
    )
    
    # === FASE 2: Optimizar rho con warm-start de Fase 1 ===
    params_phase2 = np.append(result1.x, rho_init)
    bounds_phase2 = (
        [(0.05, 5.0)] * n_teams
        + [(0.05, 5.0)] * n_teams
        + [(0.5, 2.5)]
        + [rho_bounds]  # Bound ampliado para rho
    )
    
    result2 = minimize(
        negative_log_likelihood,
        params_phase2,
        args=(matches, team_indices, lambda_reg, half_life_days, reference_date),
        method="SLSQP",
        bounds=bounds_phase2,
        constraints=[
            {"type": "eq", "fun": lambda p: np.sum(p[:n_teams]) - n_teams},
            {"type": "eq", "fun": lambda p: np.sum(p[n_teams:2*n_teams]) - n_teams},
        ],
        options={"maxiter": 100, "ftol": ftol, "disp": False},
    )
    
    # Fallback: usar resultado de Fase 1 si Fase 2 no converge
    if not result2.success:
        fitted = np.append(result1.x, rho_init)
        nll_final = result1.fun
    else:
        fitted = result2.x
        nll_final = result2.fun

    fitted_alpha = fitted[:n_teams]
    fitted_beta = fitted[n_teams : 2 * n_teams]
    gamma = float(fitted[2 * n_teams])
    rho = float(fitted[2 * n_teams + 1])

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
            "negative_log_likelihood": round(float(nll_final), 4),
            "lambda_reg": lambda_reg,
            "half_life_days": half_life_days,
            "reference_date": str(reference_date)
            if reference_date
            else str(date.today()),
        },
        "teams": ratings,
    }


def get_weighted_h2h(
    h2h_record,
    home_team: str,
    away_team: str,
    reference_date=None,
    half_life_days: float = 730.0,
):
    """Calcula estadísticas H2H con control estricto de excepciones de tipado."""
    if not h2h_record or not hasattr(h2h_record, "matches") or not h2h_record.matches:
        return {
            "home_wins": 0.0,
            "away_wins": 0.0,
            "btts": 0.0,
            "total_weight": 0.0,
            "count": 0,
        }

    ref_date = reference_date or date.today()

    try:
        home_is_a = normalize_team_name(h2h_record.team_a) == normalize_team_name(
            home_team
        )
    except Exception:
        home_is_a = str(h2h_record.team_a).lower() in str(home_team).lower()

    home_wins = away_wins = btts = total_weight = 0.0
    count = 0

    for m in h2h_record.matches:
        try:
            m_date = (
                date.fromisoformat(m["date"])
                if isinstance(m["date"], str)
                else m.get("date")
            )
            weight = time_weight(m_date, ref_date, half_life_days)
        except Exception:
            weight = 1.0

        total_weight += weight
        count += 1

        g_home = int(m.get("goals_a" if home_is_a else "goals_b", 0))
        g_away = int(m.get("goals_b" if home_is_a else "goals_a", 0))

        if g_home > g_away:
            home_wins += weight
        elif g_away > g_home:
            away_wins += weight

        if int(m.get("goals_a", 0)) > 0 and int(m.get("goals_b", 0)) > 0:
            btts += weight

    return {
        "home_wins": home_wins,
        "away_wins": away_wins,
        "btts": btts,
        "total_weight": total_weight,
        "count": count,
    }
