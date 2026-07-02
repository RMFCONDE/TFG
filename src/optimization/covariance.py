"""
Estimación de la matriz de covarianza.

Implementa tres estimadores:
  1. Muestral (insesgado): S = (1/(T-1)) * (R - \bar{R})^T (R - \bar{R})
  2. Shrinkage de Ledoit-Wolf: contrae la matriz muestral hacia una matriz
     estructurada (target), reduciendo el error cuadrático medio esperado.
     Usamos la implementación de scikit-learn (LedoitWolf).
  3. PCA / modelos de factores: descomposición espectral, retención de los k
     factores principales que explican una fracción dada de la varianza total.
     Conecta con APT (Arbitrage Pricing Theory).
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def covarianza_muestral(retornos: pd.DataFrame) -> pd.DataFrame:
    r"""
    Matriz de covarianza muestral (estimador insesgado).

    Fórmula:
        \Sigma_{ij} = \frac{1}{T-1} \sum_{t=1}^T (r_{ti} - \bar{r}_i)(r_{tj} - \bar{r}_j)

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades (T observaciones x n activos).

    Returns
    -------
    pd.DataFrame
        Matriz de covarianza (n x n), con índices y columnas = tickers.
    """
    # pandas.DataFrame.cov() ya usa el denominador T-1 (insesgado).
    return retornos.cov()


def covarianza_ledoit_wolf(retornos: pd.DataFrame) -> pd.DataFrame:
    r"""
    Matriz de covarianza con shrinkage de Ledoit-Wolf.

    Idea:
      \Sigma_{LW} = (1 - \delta) \Sigma_{muestral} + \delta \cdot T
    donde T es una matriz target estructurada (scikit-learn usa un target
    de correlación constante) y \delta \in [0,1] es la intensidad de
    shrinkage calculada analíticamente para minimizar el MSE esperado.

    Ventaja: reduce el error de estimación en muestras finitas, lo que
    estabiliza los pesos de la cartera (menos concentración, menos turnover).

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades (T observaciones x n activos).

    Returns
    -------
    pd.DataFrame
        Matriz de covarianza shrinkage (n x n).
    """
    lw = LedoitWolf()
    lw.fit(retornos)
    cov = lw.covariance_  # ndarray
    return pd.DataFrame(cov, index=retornos.columns, columns=retornos.columns)


def varianza_explicada(retornos: pd.DataFrame) -> pd.DataFrame:
    r"""
    Descomposición espectral de la matriz de covarianza muestral.

    Devuelve los autovalores ordenados de mayor a menor junto con la
    varianza explicada acumulada. Útil para elegir el número k de factores
    mediante el criterio del codo o un umbral de varianza explicada.

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades (T observaciones x n activos).

    Returns
    -------
    pd.DataFrame
        Columnas: autovalor, varianza_explicada, varianza_acumulada.
        Índice = 1..n (número de componente).
    """
    Sigma = retornos.cov().values
    autovalores = np.linalg.eigvalsh(Sigma)  # simétrica → eigvalsh más estable
    autovalores = autovalores[::-1]  # orden descendente

    total = autovalores.sum()
    explicada = autovalores / total
    acumulada = np.cumsum(explicada)

    return pd.DataFrame(
        {
            "autovalor": autovalores,
            "varianza_explicada": explicada,
            "varianza_acumulada": acumulada,
        },
        index=range(1, len(autovalores) + 1),
    )


def covarianza_pca(
    retornos: pd.DataFrame,
    varianza_objetivo: float = 0.80,
    conservar_riesgo_especifico: bool = True,
) -> pd.DataFrame:
    r"""
    Matriz de covarianza basada en PCA / modelo de factores.

    Descompone la covarianza muestral \Sigma en:
        \Sigma = V \Lambda V^T = \sum_{i=1}^n \lambda_i v_i v_i^T

    Retiene solo los k factores principales que explican al menos
    varianza_objetivo de la varianza total:
        \Sigma_{k} = \sum_{i=1}^k \lambda_i v_i v_i^T

    Si conservar_riesgo_especifico es True, se añade la diagonal del
    residuo (varianza específica de cada activo no explicada por los factores):
        \Sigma_{pca} = \Sigma_{k} + \text{diag}(\Sigma - \Sigma_{k})

    Esto equivale a un modelo de factores con factores = componentes principales
    y riesgo específico = varianza residual por activo. Conecta con APT.

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades (T observaciones x n activos).
    varianza_objetivo : float
        Fracción de varianza total a retener (entre 0 y 1).
    conservar_riesgo_especifico : bool
        Si True, añade la diagonal del residuo a la matriz reconstruida.

    Returns
    -------
    pd.DataFrame
        Matriz de covarianza PCA (n x n).
    """
    Sigma = retornos.cov().values
    n = Sigma.shape[0]

    # Descomposición espectral
    autovalores, autovectores = np.linalg.eigh(Sigma)
    # eigh devuelve en orden ascendente; invertimos para descendente
    autovalores = autovalores[::-1]
    autovectores = autovectores[:, ::-1]

    # Determinar k: cuántos factores para alcanzar varianza_objetivo
    total_var = autovalores.sum()
    acumulada = np.cumsum(autovalores) / total_var
    k = int(np.searchsorted(acumulada, varianza_objetivo) + 1)
    k = min(k, n)  # no más de n factores

    # Reconstruir con k factores
    V_k = autovectores[:, :k]       # (n, k)
    Lambda_k = np.diag(autovalores[:k])  # (k, k)
    Sigma_k = V_k @ Lambda_k @ V_k.T     # (n, n)

    if conservar_riesgo_especifico:
        # Añadir varianza específica (diagonal del residuo)
        residuo = Sigma - Sigma_k
        riesgo_especifico = np.diag(np.diag(residuo))  # solo diagonal
        Sigma_pca = Sigma_k + riesgo_especifico
    else:
        Sigma_pca = Sigma_k

    return pd.DataFrame(Sigma_pca, index=retornos.columns, columns=retornos.columns)


def estimar_covarianza(
    retornos: pd.DataFrame, metodo: str = "muestral", **kwargs
) -> pd.DataFrame:
    """
    Interfaz unificada para estimar la covarianza.

    Parameters
    ----------
    retornos : pd.DataFrame
        Rentabilidades.
    metodo : str
        "muestral", "ledoit_wolf" o "pca".
    **kwargs
        Argumentos adicionales para el método (ej. varianza_objetivo para pca).

    Returns
    -------
    pd.DataFrame
        Matriz de covarianza (n x n).
    """
    if metodo == "muestral":
        return covarianza_muestral(retornos)
    elif metodo == "ledoit_wolf":
        return covarianza_ledoit_wolf(retornos)
    elif metodo == "pca":
        return covarianza_pca(retornos, **kwargs)
    else:
        raise ValueError(
            f"Método desconocido: {metodo}. Opciones válidas: 'muestral', 'ledoit_wolf', 'pca'."
        )
