"""
Tests para el módulo de optimización.

Verifica:
  1. La solución numérica del QP (cvxpy) coincide con la solución analítica
     en un caso de juguete sin restricciones de desigualdad.
  2. Las condiciones KKT se satisfacen en la solución numérica.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import inv

from src.optimization.markowitz import cartera_minima_varianza
from src.optimization.covariance import covarianza_muestral


# --- Datos de juguete (3 activos) ---

MU = pd.Series([0.001, 0.002, 0.0005], index=["A", "B", "C"])
SIGMA = pd.DataFrame(
    [[0.010, 0.002, 0.001],
     [0.002, 0.020, 0.003],
     [0.001, 0.003, 0.015]],
    index=["A", "B", "C"],
    columns=["A", "B", "C"],
)


def test_gmv_coincide_con_analitica_sin_restricciones() -> None:
    """
    Verifica que la GMV numérica (cvxpy) coincide con la solución analítica
    cuando solo hay restricción de presupuesto (sin long-only ni max_peso).

    Solución analítica (sin restricciones de desigualdad):
        w* = Σ^{-1} 1 / (1^T Σ^{-1} 1)

    donde 1 es el vector de unos.
    """
    # Solución numérica: sin long-only ni max_peso
    w_numerica = cartera_minima_varianza(
        SIGMA, long_only=False, max_peso=None
    )

    # Solución analítica
    n = len(SIGMA)
    unos = np.ones(n)
    Sigma_inv = inv(SIGMA.values)
    w_analitica = Sigma_inv @ unos / (unos @ Sigma_inv @ unos)
    w_analitica = pd.Series(w_analitica, index=SIGMA.columns)

    # Comparar: deben ser iguales salvo error numérico
    np.testing.assert_allclose(w_numerica.values, w_analitica.values, rtol=1e-5)
    # Verificar que suman 1
    assert abs(w_numerica.sum() - 1.0) < 1e-10


def test_gmv_suma_uno() -> None:
    """Los pesos de la GMV deben sumar 1 (con y sin restricciones adicionales)."""
    # Con long_only + max_peso
    w = cartera_minima_varianza(SIGMA, long_only=True, max_peso=0.50)
    assert abs(w.sum() - 1.0) < 1e-10

    # Solo presupuesto
    w2 = cartera_minima_varianza(SIGMA, long_only=False, max_peso=None)
    assert abs(w2.sum() - 1.0) < 1e-10


def test_gmv_respeta_long_only() -> None:
    """Con long_only=True, todos los pesos deben ser >= 0."""
    w = cartera_minima_varianza(SIGMA, long_only=True, max_peso=None)
    assert (w >= -1e-10).all()


def test_gmv_respeta_max_peso() -> None:
    """Los pesos no deben superar max_peso."""
    max_peso = 0.50
    w = cartera_minima_varianza(SIGMA, long_only=True, max_peso=max_peso)
    assert (w <= max_peso + 1e-10).all()


def test_condiciones_kkt() -> None:
    r"""
    Verifica las condiciones KKT para el problema con restricciones.

    Problema:
        min  w^T Σ w
        s.a. 1^T w = 1           (λ, sin signo)
             w >= 0              (ν_i >= 0)
             w <= w_max          (η_i >= 0)

    Lagrangiano:
        L = w^T Σ w - λ(1^T w - 1) - ν^T w + η^T(w - w_max)

    KKT:
        1. Estacionariedad: 2 Σ w* - λ 1 - ν + η = 0
        2. Factibilidad primal: 1^T w* = 1, w* >= 0, w* <= w_max
        3. Factibilidad dual: ν >= 0, η >= 0
        4. Holgura complementaria: ν_i w*_i = 0, η_i (w*_i - w_max) = 0

    Verificamos que existen multiplicadores λ, ν, η que satisfacen KKT
    para la solución numérica obtenida.
    """
    w_max = 0.50
    w = cartera_minima_varianza(SIGMA, long_only=True, max_peso=w_max)
    w_vals = w.values
    n = len(w)
    Sigma_vals = SIGMA.values

    # Calculamos λ a partir de la estacionariedad.
    # Para activos con 0 < w_i < w_max: ν_i = η_i = 0 (holgura complementaria).
    # Entonces: 2 (Σ w)_i - λ = 0 → λ = 2 (Σ w)_i.
    activos_interiores = (w_vals > 1e-8) & (w_vals < w_max - 1e-8)
    if activos_interiores.any():
        i = np.where(activos_interiores)[0][0]
        lam = 2 * (Sigma_vals @ w_vals)[i]
    else:
        # Si no hay interiores, estimamos λ de la restricción de presupuesto.
        # Usamos el valor medio de 2 Σ w para activos activos.
        activos_activos = w_vals > 1e-8
        lam = 2 * np.mean((Sigma_vals @ w_vals)[activos_activos])

    # Calculamos ν y η de la estacionariedad: 2 Σ w - λ 1 - ν + η = 0
    grad = 2 * Sigma_vals @ w_vals
    # ν_i = max(0, -(2 Σ w - λ 1)_i) para activos donde w_i = 0
    # η_i = max(0, (2 Σ w - λ 1)_i) para activos donde w_i = w_max
    nu = np.zeros(n)
    eta = np.zeros(n)

    for i in range(n):
        residuo = grad[i] - lam
        if w_vals[i] < 1e-10:
            # Activo en cota inferior: w_i = 0. ν_i >= 0, η_i = 0.
            nu[i] = max(0.0, -residuo)
        elif w_vals[i] > w_max - 1e-10:
            # Activo en cota superior: w_i = w_max. ν_i = 0, η_i >= 0.
            eta[i] = max(0.0, residuo)
        else:
            # Interior: ν_i = η_i = 0, el grad debe coincidir con λ.
            pass

    # Verificar estacionariedad
    estacionariedad = grad - lam * np.ones(n) - nu + eta
    np.testing.assert_allclose(estacionariedad, np.zeros(n), atol=1e-4)

    # Verificar factibilidad dual
    assert (nu >= -1e-10).all(), f"ν debe ser >= 0: {nu}"
    assert (eta >= -1e-10).all(), f"η debe ser >= 0: {eta}"

    # Verificar holgura complementaria
    for i in range(n):
        assert abs(nu[i] * w_vals[i]) < 1e-8, (
            f"Holgura ν_{i}*w_{i} debe ser 0: ν={nu[i]}, w={w_vals[i]}"
        )
        assert abs(eta[i] * (w_vals[i] - w_max)) < 1e-8, (
            f"Holgura η_{i}*(w_{i}-w_max) debe ser 0: η={eta[i]}, w={w_vals[i]}"
        )
