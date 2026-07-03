"""
Feature engineering para la predicción de rentabilidades esperadas (μ).

Features calculadas para cada activo i en cada mes t:
  - Rezagos cortos: r_{i,t}, r_{i,t-1}, r_{i,t-2}
  - Momentum medio plazo: r acumulada de t-3 a t (3m) y de t-6 a t (6m)
  - Momentum largo plazo: r acumulada de t-12 a t (12m)
  - Volatilidad realizada: sigma_{i,12m} (12 meses)
  - Exposición al mercado (beta): beta_{i,36m} rolling al SPY

Target: rentabilidad del mes siguiente r_{i, t+1}.

Principio fundamental: en cada instante t solo se usa información hasta t.
"""

import numpy as np
import pandas as pd


def compute_features(
    retornos: pd.DataFrame,
    bench_retornos: pd.Series,
    min_periods: int = 36,
) -> pd.DataFrame:
    """
    Calcula todas las features para todos los activos en todos los periodos.

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades mensuales, índice = fecha, columnas = tickers.
    bench_retornos : pd.Series
        Rentabilidad del benchmark (SPY), mismo índice que retornos.
    min_periods : int
        Mínimo de observaciones para calcular features (36 meses = 3 años).

    Returns
    -------
    pd.DataFrame
        Panel con MultiIndex (ticker, fecha). Cada fila contiene las features
        y el target (r_{t+1}) para ese activo en ese mes.
    """
    tickers = retornos.columns.tolist()
    fechas = retornos.index

    # DataFrames para almacenar features. Clave: (ticker, fecha)
    filas = []

    for ticker in tickers:
        r = retornos[ticker]  # Serie con índice = fecha

        # --- Features basadas en rezagos ---
        r_lag1 = r.shift(1)   # r_t  (usamos shift(1) porque el target es t+1)
        r_lag2 = r.shift(2)   # r_{t-1}
        r_lag3 = r.shift(3)   # r_{t-2}

        # Momentum: rentabilidad acumulada entre t-k y t
        # Usamos .rolling(k).sum() que da r_{t-k+1} + ... + r_t
        # Pero para "momento hasta t", shift(1) para alinear con el target en t+1
        mom3 = r.rolling(3).sum().shift(1)    # suma de r_{t-2}, r_{t-1}, r_t
        mom6 = r.rolling(6).sum().shift(1)    # suma de los últimos 6 meses
        mom12 = r.rolling(12).sum().shift(1)  # suma de los últimos 12 meses

        # Volatilidad realizada: desviación típica rodante de 12 meses
        vol12 = r.rolling(12).std().shift(1)

        # Beta al SPY: cov(r_i, r_m) / var(r_m) sobre ventana de 36 meses
        beta_36m = _rolling_beta(r, bench_retornos, window=36).shift(1)

        # --- Features del mercado (SPY) ---
        r_mkt_lag1 = bench_retornos.shift(1)   # r_{m,t}
        r_mkt_lag2 = bench_retornos.shift(2)   # r_{m,t-1}
        r_mkt_vol12 = bench_retornos.rolling(12).std().shift(1)

        # --- Ensamblar en un DataFrame temporal ---
        df_ticker = pd.DataFrame(
            {
                "ticker": ticker,
                "r_lag1": r_lag1,
                "r_lag2": r_lag2,
                "r_lag3": r_lag3,
                "mom3": mom3,
                "mom6": mom6,
                "mom12": mom12,
                "vol12": vol12,
                "beta_36m": beta_36m,
                "r_mkt_lag1": r_mkt_lag1,
                "r_mkt_lag2": r_mkt_lag2,
                "r_mkt_vol12": r_mkt_vol12,
                "target": r.shift(-1),   # r_{t+1}, lo que queremos predecir
            },
            index=fechas,
        )
        filas.append(df_ticker)

    # Concatenar todos los tickers en un panel multi-indexado
    panel = pd.concat(filas)
    panel = panel.set_index("ticker", append=True).swaplevel()

    # Eliminar filas donde las FEATURES son NaN (rolling inicial, etc.).
    # NO eliminamos filas sin target: la última fila de cada ticker
    # (que no tiene r_{t+1}) se necesita para predicción.
    feature_cols = get_feature_names()
    panel = panel.dropna(subset=feature_cols)

    return panel


def _rolling_beta(
    serie: pd.Series, mercado: pd.Series, window: int = 36
) -> pd.Series:
    """
    Calcula la beta rolling de una serie respecto al mercado.

    beta_i = cov(r_i, r_m) / var(r_m) sobre una ventana móvil.

    Parameters
    ----------
    serie : pd.Series
        Rentabilidades del activo.
    mercado : pd.Series
        Rentabilidades del benchmark (mismo índice).
    window : int
        Número de periodos de la ventana.

    Returns
    -------
    pd.Series
        Beta rolling, misma longitud que serie.
    """
    cov = serie.rolling(window).cov(mercado)
    var = mercado.rolling(window).var()
    return cov / var


def get_train_pred_split(
    panel: pd.DataFrame, ticker: str, t: pd.Timestamp, train_window: int = 36
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Para un ticker y un instante t, separa train y prediction.

    Train: meses [t - train_window, t - 1] (ambos inclusive).
           Solo se usan filas con target no nulo.
    Predict: features del mes t (para predecir t+1).

    Parameters
    ----------
    panel : pd.DataFrame
        Panel de features con MultiIndex (ticker, fecha).
    ticker : str
        Ticker del activo.
    t : pd.Timestamp
        Mes de predicción (se usan features de t para predecir t+1).
    train_window : int
        Número de meses de entrenamiento.

    Returns
    -------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento (r_{s+1} para s en train).
    X_pred : pd.DataFrame
        Features del mes t para predecir r_{t+1}.
    """
    df_ticker = panel.loc[ticker]
    fechas = df_ticker.index.sort_values()

    # Encontrar la posición de t (o la fecha más cercana anterior)
    idx_t = fechas.get_indexer([t], method="ffill")[0]
    if idx_t == -1:
        raise KeyError(f"No hay datos hasta la fecha {t} para {ticker}")

    actual_t = fechas[idx_t]

    # Ventana de entrenamiento: desde train_window meses antes de t hasta t-1
    idx_start = max(0, idx_t - train_window)
    train_dates = fechas[idx_start:idx_t]

    # Filtrar filas de entrenamiento con target válido
    train_data = df_ticker.loc[train_dates].dropna(subset=["target"])
    X_train = train_data.drop(columns="target")
    y_train = train_data["target"]

    # Predicción: features del mes t (target puede ser NaN, no importa)
    X_pred = df_ticker.loc[[actual_t]].drop(columns="target")

    return X_train, y_train, X_pred


def get_pooled_train_pred(
    panel: pd.DataFrame,
    tickers: list[str],
    t: pd.Timestamp,
    train_window: int = 36,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """
    Construye el conjunto de entrenamiento AGRUPADO (panel) para el instante t.

    A diferencia de get_train_pred_split (un activo), aquí se apilan las
    observaciones (features, target) de TODOS los activos dentro de la ventana
    [t - train_window, t - 1]. Así un único modelo aprende la relación
    features -> rentabilidad compartida entre activos, con ~30 x train_window
    observaciones en lugar de train_window por activo (reduce la varianza del
    estimador). El control de look-ahead es idéntico al del caso por activo:
    cada activo aporta solo filas con información hasta t-1, y la predicción usa
    las features del mes t.

    Returns
    -------
    X_train : pd.DataFrame
        Features de entrenamiento apiladas y ordenadas cronológicamente
        (el orden temporal permite usar TimeSeriesSplit en la CV).
    y_train : pd.Series
        Target correspondiente, alineado con X_train.
    X_pred : pd.DataFrame
        Una fila de features por activo (mes t), indexada por ticker.
    pred_tickers : list[str]
        Tickers para los que hay fila de predicción, en el orden de X_pred.
    """
    X_tr_list, y_tr_list, X_pred_rows, pred_tickers = [], [], [], []

    for ticker in tickers:
        try:
            X_tr, y_tr, X_pr = get_train_pred_split(panel, ticker, t, train_window)
        except (KeyError, IndexError):
            continue
        if len(X_tr) > 0:
            X_tr_list.append(X_tr)
            y_tr_list.append(y_tr)
        if len(X_pr) > 0:
            X_pred_rows.append(X_pr)
            pred_tickers.append(ticker)

    if not X_tr_list or not X_pred_rows:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), []

    X_train = pd.concat(X_tr_list)
    y_train = pd.concat(y_tr_list)

    # Ordenar cronológicamente (el índice es la fecha) para que la validación
    # cruzada temporal de Ridge/Lasso respete el orden de los datos.
    orden = np.argsort(X_train.index.values, kind="stable")
    X_train = X_train.iloc[orden]
    y_train = y_train.iloc[orden]

    # Una fila de predicción por activo, identificada por su ticker.
    X_pred = pd.concat(X_pred_rows)
    X_pred.index = pred_tickers

    return X_train, y_train, X_pred, pred_tickers


def get_feature_names() -> list[str]:
    """Devuelve los nombres de las features (sin target)."""
    return [
        "r_lag1", "r_lag2", "r_lag3",
        "mom3", "mom6", "mom12",
        "vol12", "beta_36m",
        "r_mkt_lag1", "r_mkt_lag2", "r_mkt_vol12",
    ]
