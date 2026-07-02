"""
Preprocesado de datos: precios → rentabilidades logarítmicas.

Pipeline:
  1. Cargar precios ajustados desde data/raw/precios.csv.
  2. Calcular rentabilidades logarítmicas diarias: r_t = ln(P_t / P_{t-1}).
  3. Limpiar valores faltantes (forward-fill + drop de filas residuales).
  4. (Opcional) Agregar a frecuencia mensual para el backtesting.
  5. Guardar en data/processed/.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def cargar_precios() -> pd.DataFrame:
    """
    Carga los precios crudos desde data/raw/precios.csv.

    Returns
    -------
    pd.DataFrame
        Índice = DatetimeIndex, columnas = tickers, valores = precio ajustado.
    """
    raiz = Path(__file__).resolve().parents[2]
    ruta = raiz / "data" / "raw" / "precios.csv"
    precios = pd.read_csv(ruta, index_col=0, parse_dates=True)
    return precios


def calcular_retornos(precios: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rentabilidades logarítmicas diarias.

    Fórmula:
        r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Usamos np.log() sobre los precios y luego .diff().
    Esto es más eficiente y numéricamente estable que dividir y luego aplicar log.

    Parameters
    ----------
    precios : pd.DataFrame
        Precios ajustados (filas = fechas, columnas = tickers).

    Returns
    -------
    pd.DataFrame
        Rentabilidades logarítmicas diarias. La primera fila de cada ticker es NaN
        (no hay P_{t-1} para el primer día).
    """
    # np.log(): logaritmo natural elemento a elemento sobre todo el DataFrame.
    # .diff(): resta cada fila menos la fila anterior (por defecto axis=0).
    # Esto equivale a ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1}).
    retornos = np.log(precios).diff()
    return retornos


def limpiar_retornos(retornos: pd.DataFrame) -> pd.DataFrame:
    """
    Gestiona valores faltantes en las rentabilidades.

    Estrategia:
      1. Eliminar la primera fila (toda NaN porque no hay P_{t-1} para el primer día).
      2. Si quedara algún NaN en el interior (días sin cotización), se rellena
         hacia adelante (ffill). Son pocos y no deberían distorsionar.
      3. Si tras eso persisten NaN (al inicio de algún ticker), se eliminan esas filas.

    Parameters
    ----------
    retornos : pd.DataFrame
        DataFrame de rentabilidades con posibles NaN.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio, sin NaN.
    """
    limpio = retornos.dropna(how="all")  # quita filas donde TODOS los tickers son NaN
    limpio = limpio.ffill()              # rellena huecos interiores hacia adelante
    # Si algún ticker empieza más tarde, las primeras filas pueden ser NaN.
    # Las eliminamos (son muy pocas).
    limpio = limpio.dropna()
    return limpio


def agregar_mensual(retornos_diarios: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega rentabilidades diarias a frecuencia mensual.

    Para rentabilidades logarítmicas, la rentabilidad mensual es la suma
    de las rentabilidades diarias dentro de cada mes:
        r_mensual = Σ r_diaria

    Propiedad de los log-returns: ln(P_fin / P_ini) = Σ ln(P_t / P_{t-1})
    donde la suma es sobre todos los días del mes.

    Parameters
    ----------
    retornos_diarios : pd.DataFrame
        Rentabilidades logarítmicas diarias.

    Returns
    -------
    pd.DataFrame
        Rentabilidades logarítmicas mensuales (índice = fin de mes).
    """
    # .resample('ME'): agrupa por fin de mes ('Month End').
    # .sum(): al ser log-returns, la suma telescópica da el retorno mensual.
    mensual = retornos_diarios.resample("ME").sum()
    return mensual


def guardar_retornos(retornos: pd.DataFrame, nombre: str) -> None:
    """Guarda un DataFrame de retornos en data/processed/."""
    raiz = Path(__file__).resolve().parents[2]
    ruta = raiz / "data" / "processed" / f"{nombre}.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    retornos.to_csv(ruta)
    print(f"Guardado: {ruta} ({len(retornos)} filas x {len(retornos.columns)} columnas)")


def main() -> None:
    """Pipeline de preprocesado: carga → retornos → limpieza → guardado."""
    print("Cargando precios...")
    precios = cargar_precios()
    print(f"  Dimensiones: {precios.shape}")

    # Rentabilidades diarias
    print("\nCalculando rentabilidades logarítmicas diarias...")
    retornos_diarios = calcular_retornos(precios)
    retornos_diarios = limpiar_retornos(retornos_diarios)
    print(f"  Limpio: {retornos_diarios.shape}")

    # Rentabilidades mensuales (para rebalanceo)
    print("\nAgregando a frecuencia mensual...")
    retornos_mensuales = agregar_mensual(retornos_diarios)
    print(f"  Mensual: {retornos_mensuales.shape}")

    # Guardar
    guardar_retornos(retornos_diarios, "retornos_diarios")
    guardar_retornos(retornos_mensuales, "retornos_mensuales")

    # Resumen para inspección
    print("\n--- Primeras filas: retornos diarios ---")
    print(retornos_diarios[["AAPL", "SPY", "MSFT"]].head())
    print("\n--- Estadísticos descriptivos: SPY diario ---")
    print(retornos_diarios["SPY"].describe())


if __name__ == "__main__":
    main()
