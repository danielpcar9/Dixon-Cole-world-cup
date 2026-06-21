from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from datetime import date

from mundial_betting.models import H2HRecord, MatchContext, TeamContext

TEAM_CONTEXT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "team_contexts.json"
)
H2H_PATH = Path(__file__).resolve().parent.parent / "data" / "h2h_records.json"

_team_contexts: dict[str, TeamContext] = {}
_h2h_records: dict[str, H2HRecord] = {}


def _h2h_key(team_a: str, team_b: str) -> str:
    """Clave normalizada para H2H, ordenada alfabéticamente."""
    from mundial_betting.data import normalize_team_name

    a = normalize_team_name(team_a)
    b = normalize_team_name(team_b)
    return f"{min(a, b)}__vs__{max(a, b)}"


def load_team_contexts() -> None:
    global _team_contexts
    if not TEAM_CONTEXT_PATH.exists():
        return
    try:
        with open(TEAM_CONTEXT_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _team_contexts = {name: TeamContext(**data) for name, data in raw.items()}
    except Exception as exc:
        print(f"Warning: Failed to load team contexts: {exc}")


def save_team_contexts() -> None:
    TEAM_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEAM_CONTEXT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {name: ctx.model_dump() for name, ctx in _team_contexts.items()},
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def load_h2h_records() -> None:
    global _h2h_records
    if not H2H_PATH.exists():
        return
    try:
        with open(H2H_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _h2h_records = {key: H2HRecord(**data) for key, data in raw.items()}
    except Exception as exc:
        print(f"Warning: Failed to load H2H records: {exc}")


def save_h2h_records() -> None:
    H2H_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(H2H_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {key: rec.model_dump() for key, rec in _h2h_records.items()},
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def get_team_context(team_name: str) -> Optional[TeamContext]:
    from mundial_betting.data import normalize_team_name

    norm = normalize_team_name(team_name)
    return _team_contexts.get(norm)


def set_team_context(ctx: TeamContext) -> None:
    from mundial_betting.data import normalize_team_name

    norm = normalize_team_name(ctx.team_name)
    ctx.team_name = norm
    _team_contexts[norm] = ctx
    save_team_contexts()


def get_h2h(team_a: str, team_b: str) -> Optional[H2HRecord]:
    return _h2h_records.get(_h2h_key(team_a, team_b))


def set_h2h(record: H2HRecord) -> None:
    key = _h2h_key(record.team_a, record.team_b)
    _h2h_records[key] = record
    save_h2h_records()


def add_h2h_match(
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
    match_date: str,
    tournament: str = "",
) -> None:
    """Añade un partido al historial H2H entre dos equipos."""
    key = _h2h_key(team_a, team_b)
    record = _h2h_records.get(key)
    if record is None:
        record = H2HRecord(team_a=team_a, team_b=team_b)
        _h2h_records[key] = record

    record.matches.append(
        {
            "date": match_date,
            "home_team": team_a,  # who was actually the home side for this match
            "goals_a": goals_a,
            "goals_b": goals_b,
            "tournament": tournament,
        }
    )
    save_h2h_records()


def build_match_context(home_team: str, away_team: str) -> Optional[MatchContext]:
    """
    Construye MatchContext automáticamente a partir de los datos persistidos.
    Si no hay datos suficientes, devuelve None.
    """
    home_ctx = get_team_context(home_team)
    away_ctx = get_team_context(away_team)
    h2h = get_h2h(home_team, away_team)

    if not home_ctx and not away_ctx and not h2h:
        return None

    h2h_home_wins: float = 0.0
    h2h_away_wins: float = 0.0
    h2h_total: float = 0.0
    h2h_btts_count: float = 0.0

    if h2h:
        from datetime import date as _date
        from mundial_betting.data import normalize_team_name
        from mundial_betting.dixon_coles import time_weight

        # Determine whether today's home_team corresponds to team_a or team_b in the
        # stored H2HRecord. The key is sorted alphabetically so team_a may not be the
        # side that was 'home' in any individual match — we only care about which slot
        # maps to the team that is HOME today.
        home_is_team_a = normalize_team_name(h2h.team_a) == normalize_team_name(
            home_team
        )
        ref_date = _date.today()

        for match in h2h.matches:
            try:
                match_date_val = _date.fromisoformat(match["date"])
            except (KeyError, ValueError):
                match_date_val = None

            w = time_weight(match_date_val, ref_date, half_life_days=730.0)

            h2h_total += w
            goals_home = match["goals_a"] if home_is_team_a else match["goals_b"]
            goals_away = match["goals_b"] if home_is_team_a else match["goals_a"]

            if goals_home > goals_away:
                h2h_home_wins += w
            elif goals_away > goals_home:
                h2h_away_wins += w

            if match["goals_a"] > 0 and match["goals_b"] > 0:
                h2h_btts_count += w

    home_form = home_ctx.form if home_ctx else None
    away_form = away_ctx.form if away_ctx else None

    return MatchContext(
        h2h_home_wins=h2h_home_wins,
        h2h_away_wins=h2h_away_wins,
        h2h_total=h2h_total,
        h2h_btts_count=h2h_btts_count,
        home_btts_streak=home_form.btts_count if home_form else 0,
        away_btts_streak=away_form.btts_count if away_form else 0,
        home_clean_sheets_last5=home_form.clean_sheets if home_form else 0,
        away_clean_sheets_last5=away_form.clean_sheets if away_form else 0,
        home_key_players_available=home_ctx.availability_factor() if home_ctx else 1.0,
        away_key_players_available=away_ctx.availability_factor() if away_ctx else 1.0,
    )
