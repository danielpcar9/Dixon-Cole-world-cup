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
# NOTA: FIFA World Cup = 1.0 porque el descuento ya está incorporado en los ratings entrenados
TOURNAMENT_MULTIPLIERS: dict[str, float] = {
    "FIFA World Cup": 1.0,
    "FIFA World Cup qualification": 1.0,
    "Copa America": 0.95,
    "UEFA Euro": 0.95,
    "UEFA Euro qualification": 0.97,
    "Friendly": 1.08,
    "International Friendly": 1.08,
    "Nations League": 0.98,
    "Gold Cup": 0.96,
    "AFC Asian Cup": 0.95,
    "Africa Cup of Nations": 0.95,
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
    home_goals: np.ndarray | int,
    away_goals: np.ndarray | int,
    lmbda: np.ndarray | float,
    mu: np.ndarray | float,
    rho: float,
) -> np.ndarray | float:
    """Corrección tau de Dixon-Coles vectorizada.
    
    Args:
        home_goals: Goles locales (array o escalar)
        away_goals: Goles visitantes (array o escalar)
        lmbda: Expected goals local (array o escalar)
        mu: Expected goals visitante (array o escalar)
        rho: Parámetro de correlación
        
    Returns:
        Factor de corrección tau (array o escalar)
    """
    # Caso escalar (para compatibilidad hacia atrás)
    if not isinstance(home_goals, np.ndarray):
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
    
    # Caso vectorizado: usar np.where anidados
    correction = np.where(
        (home_goals == 0) & (away_goals == 0),
        1 - lmbda * mu * rho,
        np.where(
            (home_goals == 0) & (away_goals == 1),
            1 + lmbda * rho,
            np.where(
                (home_goals == 1) & (away_goals == 0),
                1 + mu * rho,
                np.where(
                    (home_goals == 1) & (away_goals == 1),
                    1 - rho,
                    1.0
                )
            )
        )
    )
    return np.clip(correction, 0.01, 2.0)


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
    
    # Vectorización con meshgrid de NumPy
    home_goals, away_goals = np.meshgrid(
        np.arange(effective_max + 1),
        np.arange(effective_max + 1),
        indexing='ij'
    )
    
    # Calcular corrección tau vectorizada
    corrections = tau_correction(home_goals, away_goals, xg.home, xg.away, rho)
    
    # Calcular PMF de Poisson vectorizada
    home_pmf = np.exp(-xg.home) * np.power(xg.home, home_goals) / np.vectorize(factorial)(home_goals)
    away_pmf = np.exp(-xg.away) * np.power(xg.away, away_goals) / np.vectorize(factorial)(away_goals)
    
    # Matriz completa vectorizada
    matrix = corrections * home_pmf * away_pmf

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


def estimate_corners(xg_home: float, xg_away: float) -> dict[str, float]:
    """Estima corners esperados usando fórmula heurística basada en xG.
    
    Args:
        xg_home: Expected goals del equipo local
        xg_away: Expected goals del equipo visitante
        
    Returns:
        Diccionario con estimación de corners (home, away, total) y 
        probabilidades over 9.5 y over 10.5
    """
    from scipy.stats import poisson
    
    # Base internacional: ~10.5 corners por partido
    BASE_CORNERS = 10.5
    CORNER_PER_XG = 2.5  # empírico en fútbol internacional
    
    expected_home = 3.5 + xg_home * CORNER_PER_XG
    expected_away = 3.5 + xg_away * CORNER_PER_XG
    expected_total = expected_home + expected_away
    
    # Mercados típicos: over/under 9.5, 10.5
    over_95 = 1 - poisson.cdf(9, expected_total)
    over_105 = 1 - poisson.cdf(10, expected_total)
    
    return {
        "expected_home": round(expected_home, 1),
        "expected_away": round(expected_away, 1),
        "expected_total": round(expected_total, 1),
        "over_9_5": round(over_95, 4),
        "over_10_5": round(over_105, 4),
    }


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
    
    # H9: Warning para gap confederativo alto
    from mundial_betting.data import TEAMS as GLOBAL_TEAMS, normalize_team_name
    home_key = normalize_team_name(home_team)
    away_key = normalize_team_name(away_team)
    warning_msg: str | None = None
    if home_key in GLOBAL_TEAMS and away_key in GLOBAL_TEAMS:
        home = GLOBAL_TEAMS[home_key]
        away = GLOBAL_TEAMS[away_key]
        rating_gap = abs(home.attack - away.attack) / max(home.attack, away.attack)
        if rating_gap > 0.4:
            warning_msg = "Gap confederativo alto — divergencia modelo/mercado esperada en 1X2"

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
        "exact_scores": top_exact_scores(matrix),
    }
    if warning_msg:
        response["warning"] = warning_msg
    if context_meta is not None:
        response["context"] = context_meta
    if odds:
        response["edges"] = edge_report(markets, odds, odds_format)
    
    # Añadir estimación de corners basada en xG
    response["corners"] = estimate_corners(xg.home, xg.away)
    
    return response


def time_weight(
    match_date: date | np.ndarray | None,
    reference_date: date,
    half_life_days: float,
) -> float | np.ndarray:
    """Ponderación temporal exponencial vectorizada.
    
    Args:
        match_date: Fecha del partido (date, array de dates, o None)
        reference_date: Fecha de referencia
        half_life_days: Vida media en días
        
    Returns:
        Peso temporal (float o array)
    """
    # Caso escalar (para compatibilidad hacia atrás)
    if not isinstance(match_date, np.ndarray):
        if match_date is None:
            return 1.0
        days_diff = (reference_date - match_date).days
        if days_diff < 0:
            return 1.0
        return 0.5 ** (days_diff / half_life_days)
    
    # Caso vectorizado: calcular para todo el array de una vez
    # Convertir fechas a días desde epoch
    ref_ordinal = reference_date.toordinal()
    match_ordinals = np.array([d.toordinal() if d is not None else ref_ordinal for d in match_date])
    days_diff = ref_ordinal - match_ordinals
    
    # Weight = 0.5^(days_diff / half_life_days)
    weights = np.where(days_diff < 0, 1.0, 0.5 ** (days_diff / half_life_days))
    return weights


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
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    temporal_weights: np.ndarray,
    log_fact_home: np.ndarray,
    log_fact_away: np.ndarray,
    is_neutral: np.ndarray,
    n_teams: int,
    lambda_reg: float = 0.5,
) -> float:
    """Función de verosimilitud negativa completamente vectorizada.
    
    Args:
        params: Vector de parámetros [alpha, beta, gamma, rho]
        home_idx: Índices de equipos locales (precalculado)
        away_idx: Índices de equipos visitantes (precalculado)
        home_goals: Goles locales (precalculado)
        away_goals: Goles visitantes (precalculado)
        weights: Pesos por torneo (precalculado)
        temporal_weights: Pesos temporales (precalculado)
        log_fact_home: Log(factorial) goles locales (precalculado)
        log_fact_away: Log(factorial) goles visitantes (precalculado)
        is_neutral: Array booleano indicando partidos en campo neutral (precalculado)
        n_teams: Número de equipos
        lambda_reg: Regularización L2
        
    Returns:
        Verosimilitud negativa
    """
    alpha = params[:n_teams]
    beta = params[n_teams : 2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    # Compute lambda and mu for all matches at once
    home_advantage = np.where(is_neutral, 1.0, gamma)  # 1.0 para neutrales, gamma para locales
    raw_lmbda = alpha[home_idx] * beta[away_idx] * home_advantage
    raw_mu = alpha[away_idx] * beta[home_idx]

    lmbda = np.clip(raw_lmbda, 0.01, 4.5)
    mu = np.clip(raw_mu, 0.01, 4.5)

    # Vectorized tau correction (ahora opera directamente sobre arrays)
    corrections = tau_correction(home_goals, away_goals, lmbda, mu, rho)

    if np.any(corrections <= 0):
        return 1e10

    effective_weights = weights * temporal_weights

    # Vectorized log-likelihood computation
    log_likelihood = np.sum(effective_weights * (
        np.log(corrections)
        - lmbda
        + home_goals * np.log(lmbda)
        - log_fact_home
        - mu
        + away_goals * np.log(mu)
        - log_fact_away
    ))

    reg_term = lambda_reg * np.sum((alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(-log_likelihood + reg_term)


def train_ratings(
    matches: list[MatchData],
    lambda_reg: float = 0.5,
    half_life_days: float = 730.0,
    reference_date: date | None = None,
    previous_ratings: dict[str, dict] | None = None,
    rho_bounds: tuple[float, float] = (-0.20, 0.05),
    ftol: float = 1e-6,
    maxiter: int = 1000,
) -> dict[str, object]:
    """Entrena los ratings Dixon-Coles con optimización en dos fases y warm-start.
    
    Implementa la Recomendación Prioritaria #1: optimización por bloques que:
    (a) carga parámetros previos como warm-start, 
    (b) ejecuta SLSQP en dos fases separando la optimización de rho, 
    (c) relaja ftol a 1e-6 para mayor velocidad,
    (d) pasa arrays precalculados a la función de verosimilitud.
    
    Args:
        matches: Lista de partidos para entrenar
        lambda_reg: Regularización L2 (se aplica adaptativamente según H7)
        half_life_days: Vida media para ponderación temporal
        reference_date: Fecha de referencia para ponderación
        previous_ratings: Diccionario de ratings previos para warm-start.
                         Formato: {"team_name": {"attack": x, "defense": y}, 
                                   "__gamma__": z, "__rho__": w}
        rho_bounds: Tupla (min, max) para el parámetro rho. Default (-0.20, 0.05) según modelo Dixon-Coles original
        ftol: Tolerancia de convergencia. Default 1e-6
        maxiter: Máximo de iteraciones. Default 1000
    
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
    
    # === PRECALCULAR ARRAYS CONSTANTES (fuera del bucle de optimización) ===
    home_teams = [normalize_team_name(m.home_team) for m in matches]
    away_teams = [normalize_team_name(m.away_team) for m in matches]
    home_goals = np.array([m.home_goals for m in matches])
    away_goals = np.array([m.away_goals for m in matches])
    weights = np.array([m.weight for m in matches])
    
    # Map teams to indices
    home_idx = np.array([team_indices[t] for t in home_teams])
    away_idx = np.array([team_indices[t] for t in away_teams])
    
    # Precalcular log(factorial) para evitar recálculo en cada iteración
    from math import factorial as _factorial
    log_fact_home = np.array([log(_factorial(g)) for g in home_goals])
    log_fact_away = np.array([log(_factorial(g)) for g in away_goals])
    
    # Precalcular pesos temporales
    ref_date = reference_date or date.today()
    match_dates = np.array([m.match_date for m in matches])
    temporal_weights = time_weight(match_dates, ref_date, half_life_days)
    
    # Precalcular array de neutralidad
    is_neutral = np.array([m.is_neutral for m in matches])
    
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
            full_params, home_idx, away_idx, home_goals, away_goals,
            weights, temporal_weights, log_fact_home, log_fact_away,
            is_neutral, n_teams, lambda_reg
        )
    
    result1 = minimize(
        nll_phase1,
        params_phase1,
        method="L-BFGS-B",
        bounds=bounds_phase1,
        options={"maxiter": maxiter, "ftol": ftol},
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
        args=(home_idx, away_idx, home_goals, away_goals, weights, temporal_weights,
              log_fact_home, log_fact_away, is_neutral, n_teams, lambda_reg),
        method="L-BFGS-B",
        bounds=bounds_phase2,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    
    # Fallback: usar resultado de Fase 1 si Fase 2 no converge
    if not result2.success:
        import logging
        logging.warning("Phase 2 did not converge: %s — using Phase 1 result", result2.message)
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
