"""
Pipeline de predicción de rentabilidades esperadas (μ) con ML.

Para cada mes t en el periodo out-of-sample, y para cada activo:
  1. Extrae features usando solo datos hasta t.
  2. Entrena el modelo sobre la ventana de entrenamiento [t-36, t-1].
  3. Predice μ para el mes t+1.
  4. Devuelve el vector completo de μ predichos, listo para el QP.

La predict_mu() es la función principal que orquesta el proceso.
"""

import numpy as np
import pandas as pd

from .features import compute_features, get_pooled_train_pred
from .models import train_predict_ridge, train_predict_lasso, train_predict_rf


# Diccionario que mapea nombres de método a su función de entrenamiento
_MODEL_FUNCTIONS = {
    "ridge": train_predict_ridge,
    "lasso": train_predict_lasso,
    "rf": train_predict_rf,
}


def predict_mu(
    retornos: pd.DataFrame,
    bench_retornos: pd.Series,
    t: pd.Timestamp,
    metodo: str = "ridge",
    train_window: int = 36,
    seed: int = 42,
    panel: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Predice μ (rentabilidad esperada) para todos los activos en el mes t+1.

    Enfoque AGRUPADO (panel): se entrena un único modelo con las observaciones
    de todos los activos en la ventana [t - train_window, t - 1] y luego se
    predice cada activo con sus features del mes t. Esto multiplica el tamaño
    efectivo de la muestra de entrenamiento (~n_activos x train_window) y reduce
    la varianza del estimador, a diferencia de ajustar un modelo por activo.

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades mensuales históricas (índice = fecha, columnas = tickers).
    bench_retornos : pd.Series
        Rentabilidades mensuales del benchmark (mismo índice).
    t : pd.Timestamp
        Mes de predicción (se usan datos hasta t inclusive).
    metodo : str
        Modelo a usar: "ridge", "lasso" o "rf".
    train_window : int
        Número de meses de entrenamiento (default 36 = 3 años).
    seed : int
        Semilla aleatoria para reproducibilidad.
    panel : pd.DataFrame o None
        Panel de features precomputado. Si None, se calcula aquí (más lento).

    Returns
    -------
    pd.Series
        μ predichos, índice = tickers (sin SPY).
    """
    train_fn = _MODEL_FUNCTIONS.get(metodo)
    if train_fn is None:
        raise ValueError(
            f"Método desconocido: {metodo}. Opciones: {list(_MODEL_FUNCTIONS.keys())}"
        )

    # Precomputar panel si no se proporciona (solo la primera vez)
    if panel is None:
        panel = compute_features(retornos, bench_retornos, min_periods=train_window)

    # Solo activos del universo (excluir benchmark si está en las columnas)
    bench_name = getattr(bench_retornos, "name", None)
    tickers = [c for c in retornos.columns if c != bench_name]

    # Conjunto de entrenamiento agrupado y filas de predicción (una por activo)
    X_train, y_train, X_pred, pred_tickers = get_pooled_train_pred(
        panel, tickers, t, train_window
    )

    # Media histórica como red de seguridad para activos sin predicción
    fallback = historical_mean_mu(retornos, t, train_window)

    if len(X_train) == 0 or len(X_pred) == 0:
        # Sin datos suficientes: μ = media histórica de cada activo
        return fallback.reindex(tickers)

    # Un único ajuste del modelo sobre todos los activos; predicción vectorizada
    preds = train_fn(X_train, y_train, X_pred, seed=seed)
    predicciones = pd.Series(preds, index=pred_tickers)

    # Completar activos sin fila de predicción con la media histórica
    return predicciones.reindex(tickers).fillna(fallback)


def predict_mu_all_methods(
    retornos: pd.DataFrame,
    bench_retornos: pd.Series,
    t: pd.Timestamp,
    train_window: int = 36,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """
    Predice μ con los tres métodos ML.

    Útil para comparar predicciones en el backtesting.

    Parameters
    ----------
    retornos, bench_retornos, t, train_window, seed : igual que predict_mu.

    Returns
    -------
    dict[str, pd.Series]
        Diccionario con claves "ridge", "lasso", "rf" y valores = μ predichos.
    """
    return {
        metodo: predict_mu(retornos, bench_retornos, t, metodo, train_window, seed)
        for metodo in _MODEL_FUNCTIONS
    }


def historical_mean_mu(
    retornos: pd.DataFrame,
    t: pd.Timestamp,
    train_window: int = 36,
) -> pd.Series:
    """
    Estimación naive de μ como la media histórica (baseline no-ML).

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades mensuales.
    t : pd.Timestamp
        Mes de predicción.
    train_window : int
        Ventana para la media.

    Returns
    -------
    pd.Series
        μ = media de los últimos train_window meses.
    """
    tickers = retornos.columns
    medias = {}
    for ticker in tickers:
        hist = retornos[ticker].loc[:t]
        medias[ticker] = hist.iloc[-train_window:].mean()
    return pd.Series(medias)
