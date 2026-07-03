"""
Modelos de Machine Learning para la predicción de rentabilidades esperadas (μ).

Implementa tres modelos:
  1. Ridge (regresión lineal con regularización L2)
  2. Lasso (regresión lineal con regularización L1)
  3. Random Forest (ensemble de árboles)

Cada modelo se entrena de forma AGRUPADA (panel): un único modelo aprende la
relación features -> rentabilidad a partir de las observaciones de todos los
activos dentro de la ventana de estimación, y luego predice cada activo con
sus propias features. El emparejamiento (features de t-1, target de t) impide
cualquier fuga de información futura.

El parámetro de regularización α de Ridge/Lasso se selecciona por validación
cruzada temporal (TimeSeriesSplit) para respetar el orden cronológico.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _get_cv_splits(n_samples: int, max_splits: int = 3) -> int:
    """Determina el número de cortes para CV temporal (máx 3 por velocidad)."""
    if n_samples < 12:
        return 0  # no CV, usar alpha fijo
    return min(max_splits, n_samples // 8)


def train_predict_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """
    Ridge: regresión lineal con penalización L2.

    El hiperparámetro α se selecciona por validación cruzada temporal
    (TimeSeriesSplit, 5 cortes) entre un rango logarítmico de 20 valores.
    Si no hay suficientes datos para CV, se usa α=1.0 (default razonable).

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target (rentabilidad del mes siguiente).
    X_pred : pd.DataFrame
        Features del mes actual para predecir el siguiente.
    seed : int
        Semilla aleatoria.

    Returns
    -------
    np.ndarray
        Predicción de rentabilidad esperada para X_pred.
    """
    if len(X_train) < 10:
        return np.full(len(X_pred), y_train.mean())

    n_splits = _get_cv_splits(len(X_train))

    if n_splits >= 2:
        alphas = np.logspace(-3, 2, 10)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        reg = RidgeCV(alphas=alphas, cv=tscv, scoring="neg_mean_squared_error")
    else:
        reg = Ridge(alpha=1.0, random_state=seed)

    # Estandarizar las features (media 0, varianza 1) antes de la regresión:
    # imprescindible al agrupar activos, pues sus escalas son muy distintas
    # (p. ej. beta ~1 frente a rentabilidades ~0,02) y la penalización lo exige.
    model = make_pipeline(StandardScaler(), reg)
    model.fit(X_train, y_train)
    return model.predict(X_pred)


def train_predict_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """
    Lasso: regresión lineal con penalización L1 (selección automática de features).

    El hiperparámetro α se selecciona por validación cruzada temporal.
    Lasso tiende a llevar coeficientes a cero, lo que lo hace más interpretable.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target.
    X_pred : pd.DataFrame
        Features de predicción.
    seed : int
        Semilla aleatoria.

    Returns
    -------
    np.ndarray
        Predicción.
    """
    if len(X_train) < 10:
        return np.full(len(X_pred), y_train.mean())

    n_splits = _get_cv_splits(len(X_train))

    if n_splits >= 2:
        alphas = np.logspace(-3, 2, 10)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        reg = LassoCV(
            alphas=alphas, cv=tscv, max_iter=5000, random_state=seed
        )
    else:
        reg = Lasso(alpha=1.0, max_iter=5000, random_state=seed)

    # Estandarización previa (ver Ridge): clave para que la penalización L1
    # no elimine features solo por su escala al agrupar activos.
    model = make_pipeline(StandardScaler(), reg)
    model.fit(X_train, y_train)
    return model.predict(X_pred)


def train_predict_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """
    Random Forest: ensemble de árboles de regresión.

    Usa 100 árboles con profundidad máxima 3 y un mínimo de 5 muestras por
    hoja para evitar sobreajuste. El resto de hiperparámetros se mantienen en
    los defaults de scikit-learn.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target.
    X_pred : pd.DataFrame
        Features de predicción.
    seed : int
        Semilla aleatoria.

    Returns
    -------
    np.ndarray
        Predicción.
    """
    if len(X_train) < 10:
        return np.full(len(X_pred), y_train.mean())

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_pred)
