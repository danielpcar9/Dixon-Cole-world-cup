from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from mundial_betting.api import app
from mundial_betting.data import TEAMS, TeamRating
from mundial_betting.dixon_coles import (
    apply_context_adjustments,
    expected_goals,
    market_probabilities,
    predict_match,
    score_matrix,
    set_trained_gamma,
    tau_correction,
    time_weight,
    train_ratings,
)
from mundial_betting.models import MatchContext, MatchData


def test_tau_correction_is_bounded_for_extreme_values() -> None:
    correction = tau_correction(0, 0, 100.0, 100.0, -0.13)

    assert 0.01 <= correction <= 2.0


def test_expected_goals_uses_gamma_parameter() -> None:
    xg_low = expected_goals("Estados Unidos", "Mexico", gamma=1.05)
    xg_high = expected_goals("Estados Unidos", "Mexico", gamma=1.50)

    assert xg_high.home > xg_low.home
    assert xg_high.home_attack_multiplier == 1.50
    assert xg_low.home_attack_multiplier == 1.05
    assert xg_high.home_defense_multiplier == 1.0


def test_expected_goals_neutral_ignores_gamma() -> None:
    xg = expected_goals("Estados Unidos", "Mexico", neutral=True, gamma=1.50)

    assert xg.home_attack_multiplier == 1.0
    assert xg.home == pytest.approx(
        TEAMS["Estados Unidos"].attack * TEAMS["Mexico"].defense,
        rel=1e-6,
    )


def test_predict_match_uses_trained_gamma_by_default() -> None:
    matches = [
        MatchData(
            home_team="Mexico",
            away_team="Canada",
            home_goals=5,
            away_goals=0,
        ),
    ]
    result = train_ratings(matches)
    trained_gamma = result["global_parameters"]["home_advantage_gamma"]
    set_trained_gamma(trained_gamma)

    pred = predict_match("Mexico", "Canada")
    assert pred["expected_goals"]["home_attack_multiplier"] == pytest.approx(
        trained_gamma,
        rel=1e-4,
    )


def test_score_matrix_is_normalized() -> None:
    matrix, xg = score_matrix("Estados Unidos", "Mexico")

    assert round(float(matrix.sum()), 8) == 1.0
    assert xg.home > 0
    assert xg.away > 0


def test_market_probabilities_add_up() -> None:
    matrix, _ = score_matrix("Francia", "Argentina", neutral=True)
    markets = market_probabilities(matrix)

    assert round(markets["home"] + markets["draw"] + markets["away"], 8) == 1.0
    assert round(markets["over_25"] + markets["under_25"], 8) == 1.0
    assert round(markets["btts_yes"] + markets["btts_no"], 8) == 1.0


def test_predict_endpoint_returns_edges() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "home_team": "Estados Unidos",
            "away_team": "México",
            "odds_format": "american",
            "odds": {
                "home": -120,
                "draw": 280,
                "away": 350,
                "over_25": -110,
                "under_25": -110,
                "btts_yes": -105,
                "btts_no": -115,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "markets" in body
    assert "edges" in body
    assert body["home_team"] == "Estados Unidos"
    assert body["away_team"] == "Mexico"


def test_train_ratings_returns_team_parameters() -> None:
    result = train_ratings(
        [
            MatchData(home_team="Mexico", away_team="Canada", home_goals=2, away_goals=1),
            MatchData(home_team="Canada", away_team="Estados Unidos", home_goals=1, away_goals=1),
            MatchData(home_team="Estados Unidos", away_team="Mexico", home_goals=1, away_goals=0),
            MatchData(home_team="Mexico", away_team="Estados Unidos", home_goals=0, away_goals=0),
        ]
    )

    assert "global_parameters" in result
    assert set(result["teams"]) == {"Canada", "Estados Unidos", "Mexico"}


def test_train_endpoint_persists_ratings() -> None:
    import os
    from mundial_betting.data import TRAINED_RATINGS_PATH, TEAMS, load_trained_ratings

    old_file_exists = TRAINED_RATINGS_PATH.exists()
    old_file_content = None
    if old_file_exists:
        with open(TRAINED_RATINGS_PATH, "r", encoding="utf-8") as f:
            old_file_content = f.read()
        os.remove(TRAINED_RATINGS_PATH)

    client = TestClient(app)
    try:
        response = client.post(
            "/train",
            json={
                "matches": [
                    {"home_team": "Mexico", "away_team": "Canada", "home_goals": 2, "away_goals": 1},
                    {"home_team": "Canada", "away_team": "Estados Unidos", "home_goals": 1, "away_goals": 1},
                    {"home_team": "Estados Unidos", "away_team": "Mexico", "home_goals": 1, "away_goals": 0},
                    {"home_team": "Mexico", "away_team": "Estados Unidos", "home_goals": 0, "away_goals": 0},
                ]
            }
        )
        assert response.status_code == 200
        assert TRAINED_RATINGS_PATH.exists()
        assert "Mexico" in TEAMS
    finally:
        if TRAINED_RATINGS_PATH.exists():
            os.remove(TRAINED_RATINGS_PATH)
        if old_file_exists and old_file_content is not None:
            with open(TRAINED_RATINGS_PATH, "w", encoding="utf-8") as f:
                f.write(old_file_content)
        load_trained_ratings()


def test_training_and_prediction_share_same_model() -> None:
    matches = [
        MatchData(home_team="Mexico", away_team="Canada", home_goals=2, away_goals=1),
        MatchData(home_team="Canada", away_team="Mexico", home_goals=1, away_goals=0),
    ]
    result = train_ratings(matches)
    trained_gamma = result["global_parameters"]["home_advantage_gamma"]

    alpha_mex = result["teams"]["Mexico"]["attack"]
    beta_can = result["teams"]["Canada"]["defense"]
    expected_lambda = alpha_mex * beta_can * trained_gamma

    xg = expected_goals(
        "Mexico",
        "Canada",
        gamma=trained_gamma,
        teams={
            "Mexico": TeamRating(attack=alpha_mex, defense=1.0, flag="MX"),
            "Canada": TeamRating(attack=1.0, defense=beta_can, flag="CA"),
        },
    )
    assert round(xg.home, 4) == round(expected_lambda, 4)


def test_regularization_shrinks_extreme_ratings() -> None:
    matches = [
        MatchData(home_team="Mexico", away_team="Canada", home_goals=5, away_goals=0),
        MatchData(home_team="Canada", away_team="Mexico", home_goals=0, away_goals=5),
    ]

    result_low_reg = train_ratings(matches, lambda_reg=0.01)
    result_high_reg = train_ratings(matches, lambda_reg=2.0)

    mex_attack_low = result_low_reg["teams"]["Mexico"]["attack"]
    mex_attack_high = result_high_reg["teams"]["Mexico"]["attack"]

    assert mex_attack_high < mex_attack_low
    assert mex_attack_high > 1.0
    assert abs(mex_attack_high - 1.0) < abs(mex_attack_low - 1.0)


def test_regularization_does_not_prevent_learning_clear_signal() -> None:
    matches = (
        [MatchData(home_team="Mexico", away_team="Canada", home_goals=3, away_goals=0) for _ in range(20)]
        + [MatchData(home_team="Canada", away_team="Mexico", home_goals=0, away_goals=3) for _ in range(20)]
    )

    result = train_ratings(matches, lambda_reg=1.0)

    assert result["teams"]["Mexico"]["attack"] > result["teams"]["Canada"]["attack"]
    assert result["teams"]["Mexico"]["defense"] < result["teams"]["Canada"]["defense"]


def test_train_endpoint_accepts_lambda_reg() -> None:
    client = TestClient(app)
    response = client.post(
        "/train",
        json={
            "matches": [
                {"home_team": "Mexico", "away_team": "Canada", "home_goals": 2, "away_goals": 1},
            ],
            "lambda_reg": 1.5,
        },
    )
    assert response.status_code == 200
    assert response.json()["global_parameters"]["lambda_reg"] == 1.5


def test_time_weight_no_date_returns_one() -> None:
    assert time_weight(None, date.today(), 730) == 1.0


def test_time_weight_recent_match_high_weight() -> None:
    recent = date.today() - timedelta(days=30)
    weight = time_weight(recent, date.today(), 730)
    assert weight > 0.9


def test_time_weight_old_match_low_weight() -> None:
    old = date.today() - timedelta(days=1460)
    weight = time_weight(old, date.today(), 730)
    assert weight == pytest.approx(0.25, rel=1e-3)


def test_time_weight_future_match_no_penalty() -> None:
    future = date.today() + timedelta(days=30)
    weight = time_weight(future, date.today(), 730)
    assert weight == 1.0


def test_training_with_time_decay_changes_ratings() -> None:
    today = date.today()
    old_date = today - timedelta(days=2000)
    recent_date = today - timedelta(days=30)

    matches_no_decay = [
        MatchData(home_team="Mexico", away_team="Canada", home_goals=5, away_goals=0, match_date=old_date),
        MatchData(home_team="Canada", away_team="Mexico", home_goals=0, away_goals=1, match_date=recent_date),
    ]
    result_no_decay = train_ratings(matches_no_decay, half_life_days=99999)

    matches_with_decay = [
        MatchData(home_team="Mexico", away_team="Canada", home_goals=5, away_goals=0, match_date=old_date),
        MatchData(home_team="Canada", away_team="Mexico", home_goals=0, away_goals=1, match_date=recent_date),
    ]
    result_with_decay = train_ratings(matches_with_decay, half_life_days=730)

    mex_attack_no_decay = result_no_decay["teams"]["Mexico"]["attack"]
    mex_attack_decay = result_with_decay["teams"]["Mexico"]["attack"]

    assert mex_attack_decay < mex_attack_no_decay


def test_train_endpoint_accepts_time_params() -> None:
    client = TestClient(app)
    today = date.today().isoformat()
    old = (date.today() - timedelta(days=1000)).isoformat()

    response = client.post(
        "/train",
        json={
            "matches": [
                {
                    "home_team": "Mexico",
                    "away_team": "Canada",
                    "home_goals": 2,
                    "away_goals": 1,
                    "match_date": old,
                },
            ],
            "lambda_reg": 0.5,
            "half_life_days": 500,
            "reference_date": today,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["global_parameters"]["half_life_days"] == 500.0
    assert body["global_parameters"]["reference_date"] == today


def test_train_ratings_with_weights() -> None:
    # Baseline with uniform weights
    matches_uniform = [
        MatchData(home_team="Mexico", away_team="Canada", home_goals=3, away_goals=0, weight=1.0),
        MatchData(home_team="Canada", away_team="Estados Unidos", home_goals=0, away_goals=3, weight=1.0),
        MatchData(home_team="Estados Unidos", away_team="Mexico", home_goals=0, away_goals=3, weight=1.0),
    ]
    res_uniform = train_ratings(matches_uniform)

    # Heavily weight the second match (Canada vs Estados Unidos) which was a 0-3 win for Estados Unidos
    matches_weighted = [
        MatchData(home_team="Mexico", away_team="Canada", home_goals=3, away_goals=0, weight=1.0),
        MatchData(home_team="Canada", away_team="Estados Unidos", home_goals=0, away_goals=3, weight=100.0),
        MatchData(home_team="Estados Unidos", away_team="Mexico", home_goals=0, away_goals=3, weight=1.0),
    ]
    res_weighted = train_ratings(matches_weighted)

    # Ratings should differ because of the weights
    assert res_uniform["teams"] != res_weighted["teams"]


def test_apply_context_adjustments_h2h_boost() -> None:
    ctx = MatchContext(
        h2h_home_wins=1,
        h2h_away_wins=5,
        h2h_total=6,
        h2h_btts_count=0,
    )
    xg_base = expected_goals("Estados Unidos", "Mexico", gamma=1.0)
    xg_adj, adj = apply_context_adjustments(xg_base, ctx)

    assert adj.h2h_boost_applied is True
    assert xg_adj.home < xg_base.home
    assert xg_adj.away > xg_base.away


def test_apply_context_adjustments_btts_boost() -> None:
    ctx = MatchContext(
        home_btts_streak=5,
        away_btts_streak=4,
    )
    xg_base = expected_goals("Estados Unidos", "Mexico", gamma=1.0)
    xg_adj, adj = apply_context_adjustments(xg_base, ctx)

    assert adj.btts_boost_applied is True
    assert xg_adj.home > xg_base.home
    assert xg_adj.away > xg_base.away


def test_apply_context_adjustments_key_players() -> None:
    ctx = MatchContext(
        home_key_players_available=0.7,
        away_key_players_available=1.0,
    )
    xg_base = expected_goals("Estados Unidos", "Mexico", gamma=1.0)
    xg_adj, adj = apply_context_adjustments(xg_base, ctx)

    assert adj.key_players_boost_applied is True
    assert xg_adj.home < xg_base.home
    assert xg_adj.away == xg_base.away


def test_predict_with_context_returns_adjustments() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "home_team": "Estados Unidos",
            "away_team": "México",
            "context": {
                "h2h_home_wins": 1,
                "h2h_away_wins": 5,
                "h2h_total": 6,
                "h2h_btts_count": 0,
                "home_btts_streak": 5,
                "away_key_players_available": 1.0,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "context" in body
    assert body["context"]["adjustments"]["h2h_boost_applied"] is True
    assert body["context"]["h2h_home_win_rate"] == pytest.approx(0.1667, rel=1e-3)


def test_predict_without_context_no_context_field() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "home_team": "Estados Unidos",
            "away_team": "México",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "context" not in body


