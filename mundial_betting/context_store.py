from __future__ import annotations

import json
import threading
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
_init_lock = threading.Lock()
_initialized = False


def _h2h_key(team_a: str, team_b: str) -> str:
    """Clave normalizada para H2H, ordenada alfabéticamente."""
    from mundial_betting.data import normalize_team_name

    a = normalize_team_name(team_a)
    b = normalize_team_name(team_b)
    return f"{min(a, b)}__vs__{max(a, b)}"


def load_team_contexts() -> None:
    global _team_contexts
    if not TEAM_CONTEXT_PATH.exists():
        print(f"ℹ️  No existe {TEAM_CONTEXT_PATH} — contextos de equipo vacíos")
        return
    try:
        with open(TEAM_CONTEXT_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _team_contexts = {name: TeamContext(**data) for name, data in raw.items()}
        print(f"✅ Contextos de equipo cargados: {len(_team_contexts)} equipos")
    except Exception as exc:
        print(f"⚠️  Error cargando contextos de equipo: {exc}")


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
        print(f"ℹ️  No existe {H2H_PATH} — registros H2H vacíos")
        return
    try:
        with open(H2H_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _h2h_records = {key: H2HRecord(**data) for key, data in raw.items()}
        print(f"✅ Registros H2H cargados: {len(_h2h_records)} series")
    except Exception as exc:
        print(f"⚠️  Error cargando H2H: {exc}")


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
            "home_team": team_a,
            "goals_a": goals_a,
            "goals_b": goals_b,
            "tournament": tournament,
        }
    )
    save_h2h_records()


def build_match_context(home_team: str, away_team: str) -> Optional[MatchContext]:
    """
    Construye MatchContext automáticamente a partir de datos persistidos.
    Devuelve None si no hay datos suficientes para afectar el modelo.
    """
    home_ctx = get_team_context(home_team)
    away_ctx = get_team_context(away_team)
    h2h = get_h2h(home_team, away_team)

    # Si no hay NADA de datos, devolver None para que el modelo use predicción base
    if not home_ctx and not away_ctx and not h2h:
        print(f"ℹ️  Sin datos de contexto para {home_team} vs {away_team}")
        return None

    # Si hay contextos de equipo pero están vacíos (form=None), igual son útiles
    has_meaningful_data = False

    h2h_home_wins: float = 0.0
    h2h_away_wins: float = 0.0
    h2h_total: float = 0.0
    h2h_btts_count: float = 0.0

    if h2h and h2h.matches:
        from datetime import date as _date
        from mundial_betting.data import normalize_team_name
        from mundial_betting.dixon_coles import time_weight

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

        if h2h_total > 0:
            has_meaningful_data = True
            print(f"   📊 H2H: {len(h2h.matches)} partidos, peso total={h2h_total:.2f}")

    home_form = home_ctx.form if home_ctx else None
    away_form = away_ctx.form if away_ctx else None

    # Detectar si los formularios tienen datos reales
    if home_form and (home_form.btts_count > 0 or home_form.clean_sheets > 0):
        has_meaningful_data = True
    if away_form and (away_form.btts_count > 0 or away_form.clean_sheets > 0):
        has_meaningful_data = True

    if not has_meaningful_data:
        print(
            f"ℹ️  Datos de contexto existen pero están vacíos para {home_team} vs {away_team}"
        )
        return None

    ctx = MatchContext(
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

    # Log de qué boosts se aplicarán
    boosts = []
    if h2h_total >= 3:
        boosts.append(f"H2H({h2h_total:.1f})")
    if (
        home_form
        and away_form
        and home_form.btts_count >= 3
        and away_form.btts_count >= 3
    ):
        boosts.append("BTTS")
    if home_form and home_form.clean_sheets >= 3:
        boosts.append(f"CS-home({home_form.clean_sheets})")
    if away_form and away_form.clean_sheets >= 3:
        boosts.append(f"CS-away({away_form.clean_sheets})")
    if home_ctx and home_ctx.availability_factor() < 1.0:
        boosts.append(f"Lesiones-home({home_ctx.availability_factor():.2f})")
    if away_ctx and away_ctx.availability_factor() < 1.0:
        boosts.append(f"Lesiones-away({away_ctx.availability_factor():.2f})")

    print(
        f"   🎯 Contexto aplicado: {', '.join(boosts) if boosts else 'sin boosts activos'}"
    )

    return ctx


def get_context_for_teams(home_team: str, away_team: str) -> Optional[MatchContext]:
    """Entry point usado por la API. Carga datos si no están en memoria con thread-safe lazy init."""
    global _initialized
    if not _initialized:
        with _init_lock:
            if not _initialized:  # Double-check pattern
                if not _team_contexts:
                    load_team_contexts()
                if not _h2h_records:
                    load_h2h_records()
                _initialized = True
    return build_match_context(home_team, away_team)
