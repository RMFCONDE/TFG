"""
Métricas de evaluación de carteras.

Todas las métricas se calculan sobre la serie de rentabilidades out-of-sample
generada por el backtesting walk-forward.

Métricas:
  - Rentabilidad anualizada (compuesta geométricamente)
  - Volatilidad anualizada
  - Ratio de Sharpe (anualizado, suponiendo rf=0)
  - Máximo drawdown
  - Turnover medio (cambio en pesos entre rebalanceos)
  - Herfindahl-Hirschman (HHI) medio
"""

import numpy as np
import pandas as pd


def rentabilidad_anualizada(rentabilidades: pd.Series) -> float:
    """
    Rentabilidad anualizada compuesta geométricamente.

    r_anual = (∏(1 + r_t))^{12/T} - 1
    """
    T = len(rentabilidades)
    if T == 0:
        return 0.0
    acumulada = (1 + rentabilidades).prod()
    return acumulada ** (12 / T) - 1


def volatilidad_anualizada(rentabilidades: pd.Series) -> float:
    """Volatilidad anualizada: std * sqrt(12)."""
    return rentabilidades.std() * np.sqrt(12)


def sharpe_ratio(rentabilidades: pd.Series) -> float:
    """
    Ratio de Sharpe anualizado (rf = 0).

    S = r_anual / σ_anual
    """
    r_ann = rentabilidad_anualizada(rentabilidades)
    vol_ann = volatilidad_anualizada(rentabilidades)
    return r_ann / vol_ann if vol_ann > 0 else 0.0


def max_drawdown(rentabilidades: pd.Series) -> float:
    """
    Máximo drawdown: máxima caída desde un pico anterior.

    Se expresa en valor positivo (ej. 0.35 significa caída del 35%).
    """
    acumulada = (1 + rentabilidades).cumprod()
    pico = acumulada.cummax()
    drawdown = (acumulada - pico) / pico
    return abs(drawdown.min())


def turnover_medio(pesos: pd.DataFrame) -> float:
    """
    Turnover medio entre rebalanceos.

    TO_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|
    (el 0.5 normaliza: si todos los pesos cambian al 100%, TO=1)

    Devuelve la media de TO_t a lo largo del tiempo.
    """
    if len(pesos) <= 1:
        return 0.0
    cambios = pesos.diff().iloc[1:].abs().sum(axis=1)
    return (0.5 * cambios).mean()


def hhi_medio(pesos: pd.DataFrame) -> float:
    """
    Herfindahl-Hirschman medio de los pesos.

    HHI_t = sum_i w_{i,t}^2
    HHI = 1/N para equiponderada, 1 para un solo activo.
    """
    return (pesos ** 2).sum(axis=1).mean()


def n_activos_medio(pesos: pd.DataFrame, umbral: float = 0.001) -> float:
    """Número medio de activos con peso > umbral."""
    return (pesos.abs() > umbral).sum(axis=1).mean()


def tabla_resumen(
    resultados: dict[str, dict],
    bench_rentabilidades: pd.Series = None,
) -> pd.DataFrame:
    """
    Genera una tabla resumen con todas las métricas para cada estrategia.

    Parameters
    ----------
    resultados : dict[str, dict]
        Diccionario con resultados de run_walk_forward para cada estrategia.
    bench_rentabilidades : pd.Series o None
        Rentabilidades del benchmark (para calcular métricas del SPY).

    Returns
    -------
    pd.DataFrame
        Columnas: rent_anual, vol_anual, sharpe, max_dd, turnover, hhi, n_activos.
        Índice: nombre de la estrategia.
    """
    filas = []

    for nombre, res in resultados.items():
        rents = res["rentabilidades"]
        pesos = res["pesos"]

        filas.append(
            {
                "Estrategia": nombre,
                "Rent. anual (%)": rentabilidad_anualizada(rents) * 100,
                "Vol. anual (%)": volatilidad_anualizada(rents) * 100,
                "Sharpe": sharpe_ratio(rents),
                "Max DD (%)": max_drawdown(rents) * 100,
                "Turnover": turnover_medio(pesos),
                "HHI": hhi_medio(pesos),
                "Nº activos": n_activos_medio(pesos),
            }
        )

    return pd.DataFrame(filas).set_index("Estrategia")
