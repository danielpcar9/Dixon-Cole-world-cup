"""World Cup prediction backend.

Módulo principal para predicciones de fútbol usando el modelo Dixon-Coles,
con calibración de probabilidades y backtesting avanzado.
"""

__all__ = [
    "__version__",
    "dixon_coles",
    "calibration",
    "data",
    "models",
]

__version__ = "0.2.0"

# Importar componentes principales para acceso directo
from mundial_betting import dixon_coles, calibration
