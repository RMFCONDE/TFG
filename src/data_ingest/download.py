"""
Descarga de precios ajustados desde Yahoo Finance (yfinance).

Fuente: precios diarios ajustados (columna 'Adj Close' de yfinance).
Universo: ~30 acciones del S&P 500 + SPY como benchmark.
Periodo: 2010-01-01 a 2024-12-31 (configurable en config.yaml).
"""

import yaml
from pathlib import Path
import pandas as pd
import yfinance as yf


def cargar_config() -> dict:
    """Carga config.yaml desde la raíz del proyecto y devuelve el diccionario."""
    raiz = Path(__file__).resolve().parents[2]  # src/data_ingest -> src -> raíz
    with open(raiz / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def descargar_precios(tickers: list[str], inicio: str, fin: str) -> pd.DataFrame:
    """
    Descarga precios ajustados diarios para una lista de tickers.

    Parameters
    ----------
    tickers : list[str]
        Lista de símbolos (ej. ['AAPL', 'MSFT', ...]).
    inicio : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    fin : str
        Fecha de fin en formato 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Columnas = tickers, filas = fechas (índice DatetimeIndex).
        Cada celda es el precio de cierre ajustado (Adj Close) de ese día.
        Incluye dividendos y splits en el ajuste.
    """
    # yfinance descarga OHLCV + Adj Close para todos los tickers de golpe.
    # auto_adjust=False: conservamos solo 'Adj Close' para tener control explícito.
    # group_by='ticker': las columnas quedan como MultiIndex (Ticker, OHLCV).
    datos = yf.download(
        tickers,
        start=inicio,
        end=fin,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    # Extraemos solo la columna 'Adj Close' de cada ticker.
    # Con group_by='ticker', el MultiIndex tiene nivel 0 = ticker, nivel 1 = OHLCV.
    # .xs('Adj Close', axis=1, level=1) selecciona el nivel 1='Adj Close' para todos los tickers.
    precios = datos.xs("Adj Close", axis=1, level=1)

    # El índice ya es DatetimeIndex. Lo ordenamos por si acaso.
    precios = precios.sort_index()

    return precios


def guardar_precios(precios: pd.DataFrame, ruta: Path) -> None:
    """Guarda el DataFrame de precios en CSV dentro de data/raw/."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    precios.to_csv(ruta)
    print(f"Precios guardados en {ruta} ({len(precios)} filas x {len(precios.columns)} columnas)")


def main() -> None:
    """Pipeline principal: carga config, descarga, guarda."""
    cfg = cargar_config()
    tickers = cfg["data"]["tickers"]
    benchmark = cfg["data"]["benchmark"]
    inicio = cfg["data"]["start"]
    fin = cfg["data"]["end"]

    # Unimos tickers + benchmark en una sola lista para una descarga conjunta
    todos = tickers + [benchmark]

    print(f"Descargando {len(todos)} tickers ({len(tickers)} acciones + {benchmark})")
    print(f"Periodo: {inicio} → {fin}")

    precios = descargar_precios(todos, inicio, fin)

    # Guardamos
    raiz = Path(__file__).resolve().parents[2]
    ruta_raw = raiz / "data" / "raw" / "precios.csv"
    guardar_precios(precios, ruta_raw)

    # Resumen rápido
    print("\n--- Resumen de la descarga ---")
    print(f"Primera fecha: {precios.index[0].strftime('%Y-%m-%d')}")
    print(f"Última fecha:  {precios.index[-1].strftime('%Y-%m-%d')}")
    nulos = precios.isnull().sum()
    nulos_con_faltantes = nulos[nulos > 0]
    if len(nulos_con_faltantes) > 0:
        print(f"Tickers con datos faltantes:\n{nulos_con_faltantes}")
    else:
        print("Sin datos faltantes.")
    print(f"Columnas: {list(precios.columns)}")


if __name__ == "__main__":
    main()
