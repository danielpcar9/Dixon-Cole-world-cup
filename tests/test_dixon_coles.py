from fastapi.testclient import TestClient

from mundial_betting.api import app
from mundial_betting.dixon_coles import market_probabilities, score_matrix, train_ratings
from mundial_betting.models import MatchData


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
