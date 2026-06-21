Claro. Voy a revisar el README actual para entender qué información ya tiene y qué necesita actualizarse con los nuevos features
Aquí tienes el README actualizado. He reestructurado por completo: más claro, informativo, y con ejemplos que reflejan los nuevos features (gamma unificado, regularización, time decay).

---

```markdown
# Mundial Betting

Modelo Dixon-Coles para predecir partidos de fútbol internacional. Backend en FastAPI con frontend estático.

## Qué hace

Predice probabilidades de mercado (1X2, Over/Under 2.5, BTTS) a partir de ratings de ataque/defensa por equipo, ajustados con ventaja de localía y corrección de correlación Dixon-Coles.

### Características

| Feature | Descripción |
|---------|-------------|
| **Modelo Dixon-Coles** | Corrección de correlación para empates a bajo scoring |
| **Ventaja de localía unificada** | Parámetro `gamma` entrenado, coherente entre predicción y entrenamiento |
| **Regularización** | `lambda_reg` para evitar ratings extremos en equipos con pocos partidos |
| **Ponderación temporal** | `half_life_days` para dar más peso a partidos recientes |
| **Quita de vig** | Método proporcional para mercados 1X2, O/U 2.5 y BTTS |
| **Edge detection** | Comparación modelo vs. mercado con señales BET/FADE/SKIP |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

O con `uv` (sin crear entorno manual):

```bash
uv run --with fastapi,uvicorn,scipy,numpy,pydantic,pytest uvicorn mundial_betting.api:app --reload
```

## Uso

### 1. Iniciar servidor

```bash
uvicorn mundial_betting.api:app --reload
```

Abre `http://127.0.0.1:8000` para el frontend o usa la API directamente.

### 2. Predecir un partido

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Estados Unidos",
    "away_team": "Mexico",
    "neutral": false,
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

**Respuesta:**

```json
{
  "home_team": "Estados Unidos",
  "away_team": "Mexico",
  "expected_goals": {
    "home": 1.5234,
    "away": 0.8912,
    "home_attack_multiplier": 1.2345
  },
  "markets": {
    "home": 0.456789,
    "draw": 0.256789,
    "away": 0.286422,
    "over_25": 0.423456,
    "under_25": 0.576544,
    "btts_yes": 0.389123,
    "btts_no": 0.610877
  },
  "exact_scores": [
    {"home_goals": 1, "away_goals": 0, "probability": 0.182345},
    ...
  ],
  "edges": {
    "home": {
      "model_probability": 0.456789,
      "market_probability_no_vig": 0.423456,
      "edge": 0.033333,
      "pick": "SKIP"
    },
    ...
  }
}
```

### 3. Entrenar con datos propios

El endpoint `/train` ajusta ratings de ataque/defensa, ventaja de localía (`gamma`), y corrección de correlación (`rho`) a partir de un historial de partidos.

**Parámetros opcionales:**

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `lambda_reg` | `0.5` | Fuerza de regularización L2 (mayor = ratings más cercanos a 1.0) |
| `half_life_days` | `730` | Días para que un partido pierda la mitad de su peso |
| `reference_date` | `hoy` | Fecha de referencia para calcular el decay |

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "matches": [
      {
        "home_team": "Mexico",
        "away_team": "Canada",
        "home_goals": 2,
        "away_goals": 1,
        "match_date": "2024-06-20",
        "weight": 1.0
      },
      {
        "home_team": "Canada",
        "away_team": "Estados Unidos",
        "home_goals": 0,
        "away_goals": 3,
        "match_date": "2024-06-15",
        "weight": 2.0
      }
    ],
    "lambda_reg": 0.5,
    "half_life_days": 730,
    "reference_date": "2026-06-20"
  }'
```

**Respuesta:**

```json
{
  "global_parameters": {
    "home_advantage_gamma": 1.2345,
    "rho_correction": -0.1345,
    "negative_log_likelihood": 123.4567,
    "lambda_reg": 0.5,
    "half_life_days": 730,
    "reference_date": "2026-06-20"
  },
  "teams": {
    "Mexico": {"attack": 1.2345, "defense": 0.8765},
    "Canada": {"attack": 0.7654, "defense": 1.1234},
    "Estados Unidos": {"attack": 1.3456, "defense": 0.9876}
  }
}
```

Los ratings entrenados se persisten automáticamente y el endpoint `/predict` los usa sin necesidad de reiniciar el servidor.

### 4. Ver equipos disponibles

```bash
curl http://127.0.0.1:8000/teams
```

## Formato de datos de entrenamiento

Cada partido en el JSON de entrenamiento debe incluir:

```json
{
  "home_team": "string",
  "away_team": "string",
  "home_goals": 0,
  "away_goals": 0,
  "is_neutral": false,
  "weight": 1.0,
  "match_date": "YYYY-MM-DD"
}
```

| Campo | Requerido | Descripción |
|-------|-----------|-------------|
| `home_team` / `away_team` | Sí | Nombres normalizados (ver `/teams`) |
| `home_goals` / `away_goals` | Sí | Goles anotados |
| `is_neutral` | No | `true` si no hay ventaja de localía |
| `weight` | No | Peso manual del partido (default: 1.0) |
| `match_date` | No | Fecha para time decay (default: sin decay) |

## Tests

```bash
pytest -q
```

## Arquitectura

```
mundial_betting/
├── api.py           # FastAPI: endpoints /predict, /train, /teams
├── dixon_coles.py   # Motor: NLL, optimización, time decay, regularización
├── models.py        # Pydantic: validación de requests/responses
├── data.py          # Carga/persistencia de ratings y metadatos de equipos
└── frontend/        # UI estática servida desde /
```

## Nota responsable

Este proyecto estima probabilidades; no garantiza resultados. Úsalo como herramienta analítica, no como promesa de ganancia.
```

---

## Cambios clave respecto al anterior

| Aspecto | Antes | Ahora |
|---------|-------|-------|
| Estructura | Lista plana | Secciones claras con tabla de contenido implícito |
| Features | Lista genérica | Tabla con descripción técnica |
| Ejemplo `/train` | Solo `curl` básico | Con todos los parámetros nuevos explicados |
| Parámetros | No documentados | Tabla con defaults y descripción |
| Respuestas | No mostradas | JSON de ejemplo para `/predict` y `/train` |
| Formato de datos | No documentado | Tabla campo por campo |
| Arquitectura | No mencionada | Diagrama de archivos con responsabilidades |

