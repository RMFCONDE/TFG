# TFG — Optimización de carteras: Markowitz + Machine Learning

Código y memoria del Trabajo Fin de Grado en Ingeniería Matemática (CEU
Universidad San Pablo). Implementa el modelo media-varianza de Markowitz como
programa cuadrático convexo, tres estimadores de covarianza (muestral,
Ledoit-Wolf, PCA) y la predicción de rentabilidades esperadas con Machine
Learning (Ridge, Lasso, Random Forest), evaluados mediante backtesting
walk-forward.

Este README contiene lo mínimo para **reproducir los resultados y la memoria**
desde cero.

## Requisitos

- **Python ≥ 3.10** (desarrollado con 3.13).
- Para recompilar la memoria: una distribución **LaTeX** con `xelatex` y
  `biber` (p. ej. TeX Live o MacTeX) y las fuentes Lato y TeX Gyre Pagella.

## Instalación

Desde la raíz del proyecto:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducir los resultados

Ejecutar desde la raíz del proyecto, en orden. Todos los parámetros (tickers,
fechas, ventanas, semilla) están en `config.yaml`; la semilla está fija
(`seed: 42`) para que los resultados sean reproducibles.

```bash
# 1. Descargar precios diarios ajustados (yfinance, requiere internet)
#    -> data/raw/precios.csv   [ya incluido; este paso solo lo regenera]
python -m src.data_ingest.download

# 2. Calcular rentabilidades logarítmicas diarias y mensuales
#    -> data/processed/retornos_diarios.csv, retornos_mensuales.csv
python -m src.data_ingest.preprocess

# 3. Análisis de los estimadores de covarianza (cap. 5)
#    -> data/outputs/: scree_plot.pdf, pesos_por_estimador.pdf,
#       hhi_temporal.pdf, tabla_comparativa_sigma.tex
python scripts/analisis_estabilidad.py

# 4. Backtesting de todas las estrategias (caps. 7-8)
#    -> data/outputs/: tabla_resultados.tex, valor_acumulado.pdf,
#       sharpe_comparacion.pdf
python scripts/run_backtest.py
```

> Los datos crudos (`data/raw/precios.csv`) ya están incluidos. El paso 1 solo
> es necesario si se quieren refrescar; los resultados de la memoria se
> reproducen a partir de esos datos con los pasos 2–4.

## Tests

Verifican el optimizador frente a la solución analítica y las condiciones KKT:

```bash
pytest
```

## Recompilar la memoria

Las tablas y figuras de `data/outputs/` se incluyen automáticamente en el
documento. Para regenerar el PDF:

```bash
cd memoria
latexmk -pdf -xelatex -shell-escape TFG.tex   # -> memoria/TFG.pdf
```

## Estructura

```
config.yaml                # parámetros del experimento: tickers, fechas, ventanas, semilla
requirements.txt           # dependencias de Python (versiones congeladas)
pyproject.toml             # metadatos del proyecto
src/                       # código fuente
  main.py                  # orquesta el pipeline completo
  baselines.py             # estrategias de referencia: 1/N y SPY
  data_ingest/
    download.py            # descarga precios ajustados de Yahoo Finance (yfinance)
    preprocess.py          # retornos logarítmicos diarios y mensuales
  optimization/
    markowitz.py           # QP media-varianza con cvxpy: GMV, máximo Sharpe y frontera
    covariance.py          # estimadores de covarianza: muestral, Ledoit-Wolf y PCA
    constraints.py         # restricciones del problema: long-only, presupuesto, tope
  ml/
    features.py            # las 11 features y la muestra agrupada (pooled)
    models.py              # Ridge, Lasso y Random Forest (con estandarización)
    predict.py             # predicción walk-forward de mu, sin look-ahead
  backtest/
    engine.py              # motor walk-forward: ventana de 36 meses, rebalanceo mensual
    metrics.py             # Sharpe, drawdown, turnover, HHI, nº de activos
    significance.py        # test de Jobson-Korkie/Memmel y bootstrap del Sharpe
scripts/
  analisis_estabilidad.py  # figuras y tabla del cap. 5: HHI, scree plot, pesos
  run_classical.py         # carteras clásicas: frontera, GMV y máximo Sharpe
  run_ml.py                # entrenamiento y diagnóstico de los modelos de ML
  run_backtest.py          # backtesting completo: tablas de resultados y significancia
tests/
  test_optimization.py     # valida el QP: solución analítica y condiciones KKT
data/
  raw/                     # precios diarios descargados
  processed/               # retornos limpios (diarios y mensuales)
  outputs/                 # tablas (.tex) y figuras (.pdf) que usa la memoria
memoria/                   # fuente LaTeX de la memoria -> TFG.pdf
demo/                      # demo web interactiva del optimizador (apoyo a la presentación)
```
