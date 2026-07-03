"""
Análisis de estabilidad: compara los 3 estimadores de covarianza.

Produces figuras para el capítulo 4 de la memoria:
  1. Scree plot: autovalores y varianza explicada acumulada.
  2. Pesos GMV con cada estimador.
  3. Evolución temporal de los pesos (ventanas rodantes).
  4. Concentración (HHI) y turnover a lo largo del tiempo.
"""

import matplotlib
matplotlib.use("Agg")  # sin GUI

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# --- Configuración ---
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.optimization.covariance import varianza_explicada, estimar_covarianza
from src.optimization.markowitz import cartera_minima_varianza
OUTPUT_DIR = RAIZ / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
})

COLORES = {"muestral": "#1f77b4", "ledoit_wolf": "#ff7f0e", "pca": "#2ca02c"}


def cargar_datos() -> tuple[pd.DataFrame, list[str]]:
    """Carga retornos mensuales y devuelve (datos, tickers sin SPY)."""
    ret = pd.read_csv(
        RAIZ / "data" / "processed" / "retornos_mensuales.csv",
        index_col=0, parse_dates=True,
    )
    tickers = [c for c in ret.columns if c != "SPY"]
    return ret[tickers], tickers


def scree_plot(retornos: pd.DataFrame) -> Path:
    """
    Figura 1: autovalores (scree plot) + varianza explicada acumulada.

    Muestra cuántos componentes principales se necesitan para capturar
    distintos niveles de varianza total.
    """
    var_exp = varianza_explicada(retornos)
    n = len(var_exp)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Barras: autovalores
    ax1.bar(range(1, n + 1), var_exp["autovalor"], color=COLORES["pca"], alpha=0.7)
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Autovalor")
    ax1.set_title("Descomposición espectral de la covarianza muestral")

    # Línea: varianza acumulada
    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), var_exp["varianza_acumulada"] * 100,
             "o-", color="black", markersize=3, linewidth=1.2)
    ax2.set_ylabel("Varianza explicada acumulada (%)")
    ax2.axhline(80, color="gray", linestyle="--", alpha=0.5,
                label="80%")
    ax2.axhline(90, color="gray", linestyle="-.", alpha=0.5,
                label="90%")
    ax2.axhline(95, color="gray", linestyle=":", alpha=0.5,
                label="95%")
    ax2.legend(loc="lower right", fontsize=8)

    ruta = OUTPUT_DIR / "scree_plot.pdf"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def tabla_comparativa(retornos: pd.DataFrame) -> Path:
    """
    Tabla: pesos GMV, concentración y volatilidad para los 3 Σ estimadores.
    Se guarda como .tex para inclusión directa en la memoria.
    """
    resultados = []
    for metodo, nombre in [("muestral", "Muestral"),
                            ("ledoit_wolf", "Ledoit-Wolf"),
                            ("pca", "PCA (80\\%)")]:
        Sigma = estimar_covarianza(retornos, metodo=metodo)
        w = cartera_minima_varianza(Sigma)
        resultados.append({
            "Estimador": nombre,
            "Activos $>$ 1\\%": (w > 0.01).sum(),
            "HHI": f"{(w ** 2).sum():.3f}",
            "$\\sigma_p$ (mensual)": f"{(w @ Sigma @ w) ** 0.5 * 100:.2f}\\%",
        })

    df = pd.DataFrame(resultados)
    latex = df.to_latex(index=False, escape=False, column_format="lccc")
    ruta = OUTPUT_DIR / "tabla_comparativa_sigma.tex"
    ruta.write_text(latex)
    return ruta


def pesos_por_estimador(retornos: pd.DataFrame) -> Path:
    """
    Figura: pesos GMV con cada estimador (gráfico de barras agrupadas).
    """
    pesos = {}
    for metodo in ["muestral", "ledoit_wolf", "pca"]:
        Sigma = estimar_covarianza(retornos, metodo=metodo)
        w = cartera_minima_varianza(Sigma)
        pesos[metodo] = w[w > 0.02]  # solo activos con peso > 2%

    # Índice unión de todos los activos que aparecen en algún estimador
    todos = sorted(set().union(*[p.index for p in pesos.values()]))
    n_activos = len(todos)
    x = np.arange(n_activos)
    ancho = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metodo, nombre) in enumerate(
        [("muestral", "Muestral"), ("ledoit_wolf", "Ledoit-Wolf"), ("pca", "PCA")]
    ):
        valores = [pesos[metodo].get(t, 0.0) for t in todos]
        ax.bar(x + i * ancho, valores, ancho, label=nombre, color=COLORES[metodo])

    ax.set_xticks(x + ancho)
    ax.set_xticklabels(todos, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peso")
    ax.set_title("Pesos de la cartera de mínima varianza según estimador de $\\Sigma$")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ruta = OUTPUT_DIR / "pesos_por_estimador.pdf"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def estabilidad_temporal(retornos: pd.DataFrame) -> Path:
    """
    Figura: evolución de la concentración (HHI) con ventanas rodantes
    de 3 años para cada estimador.

    El HHI (Herfindahl-Hirschman Index) mide concentración:
        HHI = Σ w_i^2
    HHI ≈ 1/N → cartera equiponderada (más diversificada)
    HHI ≈ 1   → toda la masa en un solo activo (más concentrada)
    """
    ventana = 36  # meses (3 años)
    fechas = []
    hhi = {"muestral": [], "ledoit_wolf": [], "pca": []}

    for i in range(ventana, len(retornos) + 1):
        ventana_ret = retornos.iloc[i - ventana : i]
        fechas.append(ventana_ret.index[-1])

        for metodo in ["muestral", "ledoit_wolf", "pca"]:
            Sigma = estimar_covarianza(ventana_ret, metodo=metodo)
            w = cartera_minima_varianza(Sigma)
            hhi[metodo].append((w ** 2).sum())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for metodo, color in COLORES.items():
        ax.plot(fechas, hhi[metodo], color=color, linewidth=1.2,
                label=metodo.replace("_", " ").title())

    # Referencia: 1/N (30 activos → HHI = 1/30 ≈ 0.033)
    n_activos = len(retornos.columns)
    ax.axhline(1 / n_activos, color="gray", linestyle="--", alpha=0.6,
               label=f"1/N ({n_activos} activos)")

    ax.set_ylabel("HHI (concentración)")
    ax.set_title("Evolución de la concentración de la GMV (ventana 3 años)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    ruta = OUTPUT_DIR / "hhi_temporal.pdf"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def main() -> None:
    print("Cargando datos...")
    retornos, tickers = cargar_datos()
    # Últimos 3 años para las figuras estáticas
    ret_reciente = retornos.iloc[-36:]
    print(f"  Ventana reciente: {ret_reciente.index[0].strftime('%Y-%m')} → {ret_reciente.index[-1].strftime('%Y-%m')}")

    print("\nGenerando figuras y tablas...")

    r1 = scree_plot(ret_reciente)
    print(f"  ✓ {r1}")

    r2 = tabla_comparativa(ret_reciente)
    print(f"  ✓ {r2}")

    r3 = pesos_por_estimador(ret_reciente)
    print(f"  ✓ {r3}")

    r4 = estabilidad_temporal(retornos)
    print(f"  ✓ {r4}")

    print(f"\nOutputs en {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
