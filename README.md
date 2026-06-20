# Mundial Betting

Backend y frontend minimo para predecir partidos del Mundial con un modelo Dixon-Coles.

## Que incluye

- API FastAPI con `/predict`, `/train`, `/teams` y `/health`.
- Motor Dixon-Coles separado en `mundial_betting/dixon_coles.py`.
- Quita de vig para mercados 1X2, Over/Under 2.5 y BTTS.
- Entrenamiento con `scipy.optimize.minimize`.
- Frontend estatico servido desde FastAPI.
- Tests basicos con `pytest`.

## Ejecutar en local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn mundial_betting.api:app --reload
```

Abre `http://127.0.0.1:8000`.

Tambien puedes usar `uv` sin crear entorno manual:

```bash
uv run --with fastapi,uvicorn,scipy,numpy,pydantic,pytest uvicorn mundial_betting.api:app --reload
```

## Probar

```bash
pytest
```

## Ejemplo de prediccion

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Estados Unidos",
    "away_team": "Mexico",
    "odds_format": "american",
    "odds": {
      "home": -120,
      "draw": 280,
      "away": 350,
      "over_25": -110,
      "under_25": -110,
      "btts_yes": -105,
      "btts_no": -115
    }
  }'
```

## Ejemplo de entrenamiento

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  --data-binary @data/sample_matches.json
```

El archivo de ejemplo es pequeno y solo valida el pipeline. Para producir ratings utiles necesitas un historico amplio de partidos internacionales, idealmente ponderado por fecha, torneo y relevancia.

## Subir a GitHub

```bash
git add .
git commit -m "Initial Mundial Betting backend"
git branch -M main
git remote add origin git@github.com:TU_USUARIO/TU_REPO.git
git push -u origin main
```

## Nota responsable

Este proyecto estima probabilidades; no garantiza resultados. Usalo como herramienta analitica, no como promesa de ganancia.
