"""
Modulos de Simulacion CFD
=========================
Paquete que contiene los modulos principales para simulacion de flujo de fluidos.
"""

# Importar desde campo_velocidadesV4
from .campo_velocidadesV4 import (
    NewtonRaphsonFlowSolver,
    interpolate_cubic_natural_manual,
    NY, NX, V0_INITIAL, VY_TEST,
    VIGA_INF_X_MIN, VIGA_INF_X_MAX, VIGA_INF_Y_MIN, VIGA_INF_Y_MAX,
    VIGA_SUP_X_MIN, VIGA_SUP_X_MAX, VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX
)

# Importar desde campo_velocidadesV4.1
from .campo_velocidadesV4_1 import (
    interpolate_bicubic
)

# Alias for compatibility
interpolate_bicubic_natural = interpolate_bicubic


__all__ = [
    'NewtonRaphsonFlowSolver',
    'interpolate_cubic_natural_manual',
    'interpolate_bicubic_natural',
    'NY', 'NX', 'V0_INITIAL', 'VY_TEST',
    'VIGA_INF_X_MIN', 'VIGA_INF_X_MAX', 'VIGA_INF_Y_MIN', 'VIGA_INF_Y_MAX',
    'VIGA_SUP_X_MIN', 'VIGA_SUP_X_MAX', 'VIGA_SUP_Y_MIN', 'VIGA_SUP_Y_MAX'
]
