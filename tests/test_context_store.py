import pytest
from fastapi.testclient import TestClient

import mundial_betting.context_store as context_store
from mundial_betting.api import app
from mundial_betting.context_store import (
    add_h2h_match,
    build_match_context,
    get_h2h,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_context_files(tmp_path, monkeypatch):
    monkeypatch.setattr(context_store, "TEAM_CONTEXT_PATH", tmp_path / "team_contexts.json")
    monkeypatch.setattr(context_store, "H2H_PATH", tmp_path / "h2h_records.json")
    context_store._team_contexts = {}
    context_store._h2h_records = {}
    context_store.load_team_contexts()
    context_store.load_h2h_records()
    yield
    context_store._team_contexts = {}
    context_store._h2h_records = {}


def test_build_match_context_from_empty() -> None:
    ctx = build_match_context("EquipoInventadoA", "EquipoInventadoB")
    assert ctx is None


def test_h2h_persistence() -> None:
    add_h2h_match("Mexico", "Canada", 2, 1, "2024-06-01", "Amistoso")
    add_h2h_match("Mexico", "Canada", 3, 0, "2024-06-15", "Nations League")

    record = get_h2h("Mexico", "Canada")
    assert record is not None
    assert record.total_matches() == 2
    assert record.win_rate_for("Mexico") == 1.0
    assert record.btts_rate() == 0.5


def test_endpoint_get_team_context_404() -> None:
    response = client.get("/context/EquipoQueNoExiste")
    assert response.status_code == 404


def test_endpoint_post_and_get_team_context() -> None:
    payload = {
        "team_name": "Mexico",
        "form": {
            "last_results": ["W", "W", "D", "W", "L"],
            "goals_scored": 8,
            "goals_conceded": 3,
            "btts_count": 2,
            "clean_sheets": 2,
        },
        "key_players": [
            {"name": "Lozano", "status": "available", "impact": 1.0},
            {"name": "Jimenez", "status": "doubtful", "impact": 0.8},
        ],
    }

    post_resp = client.post("/context/Mexico", json=payload)
    assert post_resp.status_code == 200

    get_resp = client.get("/context/Mexico")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["form"]["win_rate"] == 0.6


def test_predict_auto_context_uses_persisted_data() -> None:
    client.post(
        "/context/Mexico",
        json={
            "team_name": "Mexico",
            "form": {
                "last_results": ["W", "W", "W"],
                "goals_scored": 6,
                "goals_conceded": 1,
                "btts_count": 1,
                "clean_sheets": 2,
            },
            "key_players": [{"name": "Lozano", "status": "available", "impact": 1.0}],
        },
    )

    client.post(
        "/h2h",
        json={
            "team_a": "Mexico",
            "team_b": "Canada",
            "goals_a": 3,
            "goals_b": 0,
            "date": "2024-06-01",
        },
    )

    response = client.post(
        "/predict-auto-context",
        json={
            "home_team": "Mexico",
            "away_team": "Canada",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "context" in body
    assert body["context"]["h2h_home_win_rate"] == 1.0
