import pytest
from fastapi.testclient import TestClient

from mundial_betting.api import app
from mundial_betting.data import TEAMS
from mundial_betting.dixon_coles import (
    expected_goals,
    market_probabilities,
    predict_match,
    score_matrix,
    set_trained_gamma,
    tau_correction,
    train_ratings,
)
from mundial_betting.models import MatchData


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


