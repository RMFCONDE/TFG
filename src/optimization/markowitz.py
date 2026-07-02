"""
Optimización de carteras según el modelo media-varianza de Markowitz.

Todas las variantes se resuelven como problemas cuadráticos convexos (QP)
con cvxpy. Las restricciones se construyen desde constraints.py.

Variantes:
  - Cartera de mínima varianza global (GMV).
  - Cartera de máximo ratio de Sharpe (tangente).
  - Frontera eficiente (barrido de target returns).
"""

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints import construir_restricciones


def cartera_minima_varianza(
    Sigma: pd.DataFrame,
    long_only: bool = True,
    max_peso: float | None = 0.20,
) -> pd.Series:
    r"""
    Cartera de mínima varianza global (GMV).

    Resuelve:
        min_w   w^T \Sigma w
        s.a.    \sum w_i = 1
                w_i \ge 0               (si long_only)
                w_i \le max_peso        (si se especifica)

    Parameters
    ----------
    Sigma : pd.DataFrame
        Matriz de covarianza (n x n).
    long_only : bool
        Si True, impone w >= 0.
    max_peso : float o None
        Peso máximo por activo.

    Returns
    -------
    pd.Series
        Pesos óptimos, con índice = tickers de Sigma.
    """
    n = len(Sigma.columns)
    w = cp.Variable(n)
    restricciones = construir_restricciones(w, long_only, max_peso)

    # Función objetivo: varianza de la cartera = w^T \Sigma w.
    # cp.quad_form(w, Sigma) calcula exactamente w^T * Sigma * w.
    riesgo = cp.quad_form(w, Sigma.values)
    objetivo = cp.Minimize(riesgo)
    problema = cp.Problem(objetivo, restricciones)
    problema.solve()

    if problema.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"GMV: el solver no encontró óptimo. Status: {problema.status}")

    pesos = pd.Series(w.value, index=Sigma.columns)
    return pesos


def cartera_max_sharpe(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    rf: float = 0.0,
    long_only: bool = True,
    max_peso: float | None = 0.20,
) -> pd.Series:
    r"""
    Cartera de máximo ratio de Sharpe (cartera tangente).

    El ratio de Sharpe es (mu^T w - rf) / sqrt(w^T \Sigma w).
    Maximizar esta razón es un problema cuasiconvexo. Lo resolvemos
    construyendo la frontera eficiente y seleccionando el punto con
    mayor Sharpe.

    Estrategia:
      1. Obtenemos la GMV (extremo izquierdo de la frontera).
      2. Barrimos N targets de rentabilidad entre mu_gmv y mu_max.
      3. Para cada target, resolvemos el QP de mínima varianza con
         restricción de rentabilidad mínima.
      4. Seleccionamos la cartera con mayor ratio de Sharpe.

    Parameters
    ----------
    mu : pd.Series
        Rentabilidades esperadas por activo (índice = tickers).
    Sigma : pd.DataFrame
        Matriz de covarianza.
    rf : float
        Tasa libre de riesgo (por defecto 0).
    long_only : bool
        Si True, w >= 0.
    max_peso : float o None
        Peso máximo por activo.

    Returns
    -------
    pd.Series
        Pesos de la cartera tangente.
    """
    # Cartera de mínima varianza (sin target de retorno)
    w_gmv = cartera_minima_varianza(Sigma, long_only, max_peso)
    mu_gmv = mu @ w_gmv  # rentabilidad esperada de la GMV
    mu_max = mu.max()     # rentabilidad del activo con mayor mu

    if mu_max <= mu_gmv:
        # Si todos los mu son iguales (o peores que GMV), la GMV es la tangente
        return w_gmv

    # Barrido de targets entre mu_gmv y mu_max
    n_targets = 50
    targets = np.linspace(mu_gmv, mu_max, n_targets)
    n = len(mu)
    mejor_sharpe = -np.inf
    mejor_w = w_gmv

    for target in targets:
        w = cp.Variable(n)
        restricciones = construir_restricciones(w, long_only, max_peso)
        # Restricción adicional: rentabilidad esperada >= target
        restricciones.append(mu.values @ w >= target)

        riesgo = cp.quad_form(w, Sigma.values)
        problema = cp.Problem(cp.Minimize(riesgo), restricciones)
        problema.solve()

        if problema.status not in ("optimal", "optimal_inaccurate"):
            continue

        w_opt = w.value
        r_p = mu.values @ w_opt
        sigma_p = np.sqrt(w_opt @ Sigma.values @ w_opt)
        sharpe = (r_p - rf) / sigma_p if sigma_p > 0 else -np.inf

        if sharpe > mejor_sharpe:
            mejor_sharpe = sharpe
            mejor_w = w_opt

    return pd.Series(mejor_w, index=mu.index)


def frontera_eficiente(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    long_only: bool = True,
    max_peso: float | None = 0.20,
    n_puntos: int = 50,
) -> pd.DataFrame:
    r"""
    Construye la frontera eficiente: mínimo riesgo para cada nivel
    de rentabilidad esperada.

    Para cada target r_obj \in [mu_gmv, mu_max], resuelve:
        min_w   w^T \Sigma w
        s.a.    \sum w_i = 1
                w_i \ge 0
                w_i \le max_peso
                \mu^T w \ge r_obj

    Devuelve un DataFrame con las columnas:
      - rentabilidad: rentabilidad esperada de la cartera.
      - volatilidad: desviación típica de la cartera.
      - sharpe: ratio de Sharpe = rentabilidad / volatilidad (suponiendo rf=0).

    Parameters
    ----------
    mu : pd.Series
        Rentabilidades esperadas.
    Sigma : pd.DataFrame
        Matriz de covarianza.
    long_only : bool
        Si True, w >= 0.
    max_peso : float o None
        Peso máximo por activo.
    n_puntos : int
        Número de puntos en la frontera.

    Returns
    -------
    pd.DataFrame
        Frontera eficiente con columnas [rentabilidad, volatilidad, sharpe].
    """
    w_gmv = cartera_minima_varianza(Sigma, long_only, max_peso)
    mu_gmv = mu @ w_gmv
    mu_max = mu.max()

    targets = np.linspace(mu_gmv, mu_max, n_puntos)
    n = len(mu)
    resultados = []

    for target in targets:
        w = cp.Variable(n)
        restricciones = construir_restricciones(w, long_only, max_peso)
        restricciones.append(mu.values @ w >= target)

        riesgo = cp.quad_form(w, Sigma.values)
        problema = cp.Problem(cp.Minimize(riesgo), restricciones)
        problema.solve()

        if problema.status not in ("optimal", "optimal_inaccurate"):
            continue

        w_opt = w.value
        r_p = mu.values @ w_opt
        sigma_p = np.sqrt(w_opt @ Sigma.values @ w_opt)
        sharpe = r_p / sigma_p if sigma_p > 0 else 0.0
        resultados.append(
            {"rentabilidad": r_p, "volatilidad": sigma_p, "sharpe": sharpe}
        )

    return pd.DataFrame(resultados)
