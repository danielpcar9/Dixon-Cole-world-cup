#!/usr/bin/env python3
"""
Script de inicialización de contexto para Mundial Betting.
Pobla datos de forma, lesiones y H2H para pruebas del frontend.
"""

import json
import sys
from typing import Any
from urllib import error, request

BASE_URL = "http://127.0.0.1:8000"


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any] | str]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=5) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return exc.code, body
    except error.URLError:
        return 0, ""


def check_server() -> bool:
    """Verifica que el servidor esté corriendo."""
    status, _ = _request_json("GET", f"{BASE_URL}/health")
    return status == 200


def init_team_context(team_name: str, form_data: dict[str, Any], players: list[dict[str, Any]]) -> bool:
    """Registra o actualiza el contexto de un equipo."""
    payload = {
        "team_name": team_name,
        "form": form_data,
        "key_players": players,
    }

    status, body = _request_json("POST", f"{BASE_URL}/context/{team_name}", payload=payload)
    if status == 200:
        print(f"  ✅ Contexto guardado: {team_name}")
    else:
        print(f"  ❌ Error guardando {team_name}: {status} - {body}")
        return False
    return True


def init_h2h(
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
    match_date: str,
    tournament: str = "",
) -> bool:
    """Registra un partido H2H."""
    payload = {
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "date": match_date,
        "tournament": tournament,
    }

    status, body = _request_json("POST", f"{BASE_URL}/h2h", payload=payload)
    if status == 200:
        print(f"  ✅ H2H: {team_a} {goals_a}-{goals_b} {team_b} ({match_date})")
    else:
        print(f"  ❌ Error H2H {team_a} vs {team_b}: {status} - {body}")
        return False
    return True


def main() -> None:
    print("=" * 60)
    print("Inicialización de contexto - Mundial Betting")
    print("=" * 60)

    print("\n🔍 Verificando servidor...")
    if not check_server():
        print("❌ El servidor no responde en", BASE_URL)
        print("   Asegúrate de ejecutar: uvicorn mundial_betting.api:app --reload")
        sys.exit(1)
    print("✅ Servidor activo")

    print("\n📋 Registrando contexto de JAPÓN...")
    init_team_context(
        team_name="Japon",
        form_data={
            "last_results": ["W", "W", "D", "W", "W"],
            "goals_scored": 12,
            "goals_conceded": 3,
            "btts_count": 2,
            "clean_sheets": 3,
        },
        players=[
            {"name": "Koki Ogawa", "status": "available", "impact": 1.0},
            {"name": "Kaoru Mitoma", "status": "injured", "impact": 0.9},
            {"name": "Takefusa Kubo", "status": "available", "impact": 0.8},
            {"name": "Wataru Endo", "status": "available", "impact": 0.7},
        ],
    )

    print("\n📋 Registrando contexto de TÚNEZ...")
    init_team_context(
        team_name="Tunez",
        form_data={
            "last_results": ["L", "D", "W", "L", "D"],
            "goals_scored": 5,
            "goals_conceded": 8,
            "btts_count": 4,
            "clean_sheets": 1,
        },
        players=[
            {"name": "Elias Achouri", "status": "available", "impact": 0.7},
            {"name": "Youssef Msakni", "status": "doubtful", "impact": 0.8},
            {"name": "Aissa Laidouni", "status": "available", "impact": 0.6},
        ],
    )

    print("\n📋 Registrando contexto de MÉXICO...")
    init_team_context(
        team_name="Mexico",
        form_data={
            "last_results": ["W", "W", "D", "L", "W"],
            "goals_scored": 8,
            "goals_conceded": 5,
            "btts_count": 3,
            "clean_sheets": 2,
        },
        players=[
            {"name": "Hirving Lozano", "status": "available", "impact": 1.0},
            {"name": "Raul Jimenez", "status": "available", "impact": 0.8},
            {"name": "Edson Alvarez", "status": "suspended", "impact": 0.7},
        ],
    )

    print("\n📋 Registrando contexto de CANADÁ...")
    init_team_context(
        team_name="Canada",
        form_data={
            "last_results": ["L", "L", "W", "D", "L"],
            "goals_scored": 4,
            "goals_conceded": 9,
            "btts_count": 2,
            "clean_sheets": 1,
        },
        players=[
            {"name": "Jonathan David", "status": "available", "impact": 0.9},
            {"name": "Alphonso Davies", "status": "injured", "impact": 1.0},
            {"name": "Cyle Larin", "status": "available", "impact": 0.7},
        ],
    )

    print("\n⚽ Registrando historiales H2H...")
    init_h2h("Tunez", "Japon", 0, 2, "2002-06-14", "Mundial 2002")
    init_h2h("Mexico", "Canada", 3, 0, "2023-07-09", "Gold Cup 2023")
    init_h2h("Mexico", "Canada", 2, 1, "2022-11-15", "Amistoso")
    init_h2h("Canada", "Mexico", 0, 2, "2021-07-29", "Gold Cup 2021")
    init_h2h("Mexico", "Japon", 1, 2, "2016-06-16", "Kirin Cup 2016")

    print("\n" + "=" * 60)
    print("✅ Inicialización completa")
    print("=" * 60)
    print(f"\nAbre el frontend en: {BASE_URL}")
    print("Selecciona dos equipos y haz clic en 'Predecir con contexto automático'")
    print("\nPara verificar los datos guardados:")
    print(f"  curl {BASE_URL}/context/Japon")
    print(f"  curl {BASE_URL}/h2h/Tunez/Japon")


if __name__ == "__main__":
    main()
