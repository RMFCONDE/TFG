"""
Motor de backtesting walk-forward.

Esquema:
  - Ventana de estimación de train_window meses (típicamente 36).
  - Rebalanceo mensual: en cada mes t, se estiman μ y Σ con datos hasta t,
    se calculan los pesos óptimos, y se mantiene la cartera durante t+1.
  - Evaluación out-of-sample: la rentabilidad del mes t+1 NO se usa
    para calcular los pesos que la generan.

Principio fundamental: PROHIBIDO el look-ahead. En el mes t solo se
conoce la información hasta t.
"""

import numpy as np
import pandas as pd

from src.optimization.covariance import estimar_covarianza
from src.optimization.markowitz import (
    cartera_minima_varianza,
    cartera_max_sharpe,
)
from src.ml.predict import predict_mu, historical_mean_mu


def run_walk_forward(
    retornos: pd.DataFrame,
    bench_retornos: pd.Series,
    train_window: int = 36,
    long_only: bool = True,
    max_peso: float = 0.20,
    metodo_cov: str = "muestral",
    metodo_mu: str = "historico",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta el backtesting walk-forward para una configuración dada.

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades mensuales de los activos del universo (sin benchmark).
    bench_retornos : pd.Series
        Rentabilidades mensuales del benchmark (SPY).
    train_window : int
        Meses en la ventana de estimación (default 36).
    long_only : bool
        Si True, pesos >= 0.
    max_peso : float
        Peso máximo por activo.
    metodo_cov : str
        Estimador de covarianza: "muestral", "ledoit_wolf" o "pca".
    metodo_mu : str
        Estimador de μ: "historico", "ridge", "lasso", "rf".
    seed : int
        Semilla aleatoria.
    verbose : bool
        Si True, imprime progreso cada 12 meses.

    Returns
    -------
    dict
        Claves:
        - "rentabilidades": pd.Series con rentabilidad mensual de la cartera.
        - "pesos": pd.DataFrame con pesos en cada rebalanceo (índice=fecha, columnas=tickers).
        - "config": dict con los parámetros usados.
    """
    from src.ml.features import compute_features

    fechas = retornos.index
    n = len(fechas)

    # Primer mes con predicción: necesitamos train_window meses de historia
    # y predecimos el mes train_window+1
    inicio = train_window
    fin = n - 1  # último mes que podemos evaluar (necesitamos r_{t+1})

    rentabilidades_cartera = []
    pesos_historicos = {}

    # Para ML: precomputar el panel de features una sola vez
    panel = None
    if metodo_mu != "historico":
        panel = compute_features(retornos, bench_retornos, min_periods=train_window)

    for i in range(inicio, fin):
        t = fechas[i]  # mes de decisión

        # Datos disponibles hasta t (inclusive)
        train_retornos = retornos.loc[:t].iloc[-train_window:]

        # --- Estimar Σ ---
        Sigma = estimar_covarianza(train_retornos, metodo=metodo_cov)

        # --- Estimar μ ---
        if metodo_mu == "historico":
            mu = historical_mean_mu(retornos, t, train_window)
        else:
            mu = predict_mu(retornos, bench_retornos, t, metodo=metodo_mu,
                            train_window=train_window, seed=seed, panel=panel)

        # --- Calcular pesos ---
        pesos = cartera_max_sharpe(mu, Sigma, long_only=long_only, max_peso=max_peso)

        # Guardar pesos (para análisis de turnover, concentración, etc.)
        pesos_historicos[t] = pesos

        # --- Rentabilidad out-of-sample ---
        # La cartera se mantiene durante el mes t+1
        r_next = retornos.loc[fechas[i + 1]]
        r_cartera = pesos @ r_next
        rentabilidades_cartera.append(r_cartera)

        if verbose and (i - inicio) % 12 == 0:
            print(
                f"  {metodo_mu}/{metodo_cov}: mes {t.date()} -> "
                f"r_cartera={r_cartera:.4f}, n_activos={(pesos > 0.001).sum()}"
            )

    # Construir Series/DataFrames de salida
    rent_idx = fechas[inicio + 1 : fin + 1]  # fechas de las rentabilidades
    rentabilidades = pd.Series(rentabilidades_cartera, index=rent_idx, name="rentabilidad")

    pesos_df = pd.DataFrame(pesos_historicos).T
    pesos_df.index.name = "fecha"

    return {
        "rentabilidades": rentabilidades,
        "pesos": pesos_df,
        "config": {
            "train_window": train_window,
            "long_only": long_only,
            "max_peso": max_peso,
            "metodo_cov": metodo_cov,
            "metodo_mu": metodo_mu,
        },
    }


def run_all_strategies(
    retornos: pd.DataFrame,
    bench_retornos: pd.Series,
    train_window: int = 36,
    long_only: bool = True,
    max_peso: float = 0.20,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Ejecuta todas las combinaciones de estrategias:
      - GMV con cov muestral + μ histórico (Markowitz clásico)
      - GMV con cov Ledoit-Wolf + μ histórico
      - GMV con cov PCA + μ histórico
      - Max Sharpe con Ridge
      - Max Sharpe con Lasso
      - Max Sharpe con RF

    También calcula los baselines 1/N y SPY.

    Parameters
    ----------
    retornos, bench_retornos, train_window, ... : igual que run_walk_forward.

    Returns
    -------
    dict[str, dict]
        Clave = nombre de estrategia, valor = resultado de run_walk_forward.
    """
    estrategias = {}

    # --- Markowitz con μ histórico ---
    for cov_name, cov_method in [
        ("Markowitz (muestral)", "muestral"),
        ("Markowitz (Ledoit-Wolf)", "ledoit_wolf"),
        ("Markowitz (PCA)", "pca"),
    ]:
        if verbose:
            print(f"\n--- {cov_name} ---")
        estrategias[cov_name] = run_walk_forward(
            retornos, bench_retornos, train_window,
            long_only, max_peso,
            metodo_cov=cov_method,
            metodo_mu="historico",
            seed=seed,
            verbose=verbose,
        )

    # --- Markowitz + ML ---
    # Usamos Ledoit-Wolf como estimador de Σ (el más fiable)
    for ml_name, ml_method in [
        ("Markowitz + Ridge", "ridge"),
        ("Markowitz + Lasso", "lasso"),
        ("Markowitz + RF", "rf"),
    ]:
        if verbose:
            print(f"\n--- {ml_name} ---")
        estrategias[ml_name] = run_walk_forward(
            retornos, bench_retornos, train_window,
            long_only, max_peso,
            metodo_cov="ledoit_wolf",
            metodo_mu=ml_method,
            seed=seed,
            verbose=verbose,
        )

    return estrategias
