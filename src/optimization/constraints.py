"""
Construcción de restricciones para el problema de optimización de carteras.

Todas las funciones devuelven listas de restricciones compatibles con cvxpy.
"""

import cvxpy as cp
import numpy as np


def long_only(w: cp.Variable) -> cp.Constraint:
    r"""
    Restricción de no-negatividad (long-only): w_i \ge 0 para todo i.

    Parameters
    ----------
    w : cp.Variable
        Vector de pesos (n_activos,).

    Returns
    -------
    cp.Constraint
        w >= 0
    """
    return w >= 0


def presupuesto(w: cp.Variable) -> cp.Constraint:
    r"""
    Restricción de presupuesto: \sum_i w_i = 1 (inversión total).

    Parameters
    ----------
    w : cp.Variable
        Vector de pesos (n_activos,).

    Returns
    -------
    cp.Constraint
        sum(w) == 1
    """
    return cp.sum(w) == 1


def tope_por_activo(w: cp.Variable, max_peso: float = 0.20) -> cp.Constraint:
    r"""
    Restricción de cota superior por activo: w_i \le max_peso para todo i.

    Evita concentración excesiva en un solo título.

    Parameters
    ----------
    w : cp.Variable
        Vector de pesos (n_activos,).
    max_peso : float
        Peso máximo permitido por activo (por defecto 0.20 = 20%).

    Returns
    -------
    cp.Constraint
        w <= max_peso
    """
    return w <= max_peso


def construir_restricciones(
    w: cp.Variable,
    long_only_flag: bool = True,
    max_peso: float | None = 0.20,
) -> list[cp.Constraint]:
    """
    Construye la lista completa de restricciones para el QP.

    Parameters
    ----------
    w : cp.Variable
        Vector de pesos.
    long_only_flag : bool
        Si True, añade w >= 0.
    max_peso : float o None
        Si es un número, añade w_i <= max_peso. Si es None, no se aplica.

    Returns
    -------
    list[cp.Constraint]
        Lista de restricciones lista para pasar a cp.Problem.
    """
    restricciones = [presupuesto(w)]
    if long_only_flag:
        restricciones.append(long_only(w))
    if max_peso is not None:
        restricciones.append(tope_por_activo(w, max_peso))
    return restricciones
