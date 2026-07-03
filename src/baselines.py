"""
Carteras de referencia (baselines).

Implementa dos estrategias pasivas que sirven como punto de comparación:
  1. 1/N equiponderada: mismo peso en todos los activos.
  2. SPY: comprar y mantener el benchmark del mercado.

Ambas se evalúan sobre la misma ventana out-of-sample que el resto
de estrategias para que la comparación sea justa.
"""

import numpy as np
import pandas as pd


def cartera_1n(
    retornos: pd.DataFrame,
    train_window: int = 36,
) -> dict:
    """
    Cartera equiponderada 1/N.

    En cada rebalanceo mensual, invierte 1/N en cada uno de los N activos.
    Es el baseline innocuo por excelencia (DeMiguel, Garlappi y Uppal, 2009).

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades mensuales de los activos del universo.
    train_window : int
        Meses iniciales a descartar (para alinear con el backtesting).

    Returns
    -------
    dict
        Claves "rentabilidades", "pesos", "config".
    """
    n = retornos.shape[1]
    fechas = retornos.index
    inicio = train_window
    fin = len(fechas) - 1

    w = pd.Series(1.0 / n, index=retornos.columns)
    rentabilidades = []

    for i in range(inicio, fin):
        r_next = retornos.iloc[i + 1]
        r_cartera = w @ r_next
        rentabilidades.append(r_cartera)

    rent_idx = fechas[inicio + 1 : fin + 1]
    rents = pd.Series(rentabilidades, index=rent_idx, name="rentabilidad")

    # DataFrame de pesos (constante en el tiempo)
    pesos = pd.DataFrame(
        [w.values] * len(rents),
        index=rent_idx,
        columns=w.index,
    )

    return {
        "rentabilidades": rents,
        "pesos": pesos,
        "config": {"metodo": "1/N", "train_window": train_window},
    }


def cartera_spy(
    bench_retornos: pd.Series,
    train_window: int = 36,
) -> dict:
    """
    Cartera SPY (comprar y mantener el benchmark).

    Equivale a invertir el 100% en el SPY. Sirve para responder a la pregunta:
    ¿supera la estrategia activa al mercado?

    Parameters
    ----------
    bench_retornos : pd.Series
        Rentabilidades mensuales del SPY.
    train_window : int
        Meses iniciales a descartar.

    Returns
    -------
    dict
        Claves "rentabilidades", "pesos", "config".
    """
    inicio = train_window
    fin = len(bench_retornos) - 1

    rents = bench_retornos.iloc[inicio + 1 : fin + 1].copy()
    rents.name = "rentabilidad"

    # Peso: 100% SPY
    pesos = pd.DataFrame(
        {"SPY": 1.0},
        index=rents.index,
    )

    return {
        "rentabilidades": rents,
        "pesos": pesos,
        "config": {"metodo": "SPY", "train_window": train_window},
    }
