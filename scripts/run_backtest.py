"""
Ejecuta el backtesting completo de todas las estrategias y exporta
los resultados (tabla .tex y figuras .pdf) a data/outputs/.

Uso:
    python scripts/run_backtest.py
"""

import sys
import os

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.backtest.engine import run_all_strategies
from src.backtest.metrics import tabla_resumen
from src.backtest.significance import tabla_significancia
from src.baselines import cartera_1n, cartera_spy

# Configuración de figuras (fuente serif, tamaño para la memoria)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")


def main():
    # --- Cargar datos ---
    print("Cargando datos...")
    retornos = pd.read_csv(
        "data/processed/retornos_mensuales.csv", index_col=0, parse_dates=True
    )
    spy = retornos["SPY"]
    activos = retornos.drop(columns="SPY")

    print(f"  Universo: {activos.shape[1]} activos")
    print(f"  Periodo: {activos.index[0].date()} a {activos.index[-1].date()}")
    print(f"  Meses totales: {activos.shape[0]}")

    # --- Ejecutar todas las estrategias ---
    print("\nEjecutando backtesting walk-forward...")
    resultados = run_all_strategies(activos, spy, verbose=True)

    # Añadir baselines
    print("\n--- Baselines ---")
    bl_1n = cartera_1n(activos)
    bl_spy = cartera_spy(spy)
    resultados["1/N (equiponderada)"] = bl_1n
    resultados["SPY (mercado)"] = bl_spy
    print("  Baselines calculados.")

    # --- Tabla resumen ---
    print("\n--- Tabla Resumen ---")
    tabla = tabla_resumen(resultados)
    print(tabla.to_string())

    # Exportar a LaTeX
    latex_path = os.path.join(OUTPUT_DIR, "tabla_resultados.tex")
    tabla_latex = tabla.copy()
    tabla_latex.columns = [
        "Rent. anual (\\%)",
        "Vol. anual (\\%)",
        "Sharpe",
        "Máx. DD (\\%)",
        "Turnover",
        "HHI",
        "Nº activos",
    ]
    tabla_latex = tabla_latex.round(2)

    # Formato compacto: 2 decimales, coma decimal y una sola fila de cabecera
    # (sin esto, to_latex imprime 6 decimales y la tabla desborda la página).
    fmt2 = lambda x: f"{x:.2f}".replace(".", ",")
    tabla_out = tabla_latex.copy()
    tabla_out.insert(0, "Estrategia", tabla_out.index)
    with open(latex_path, "w") as f:
        f.write(
            tabla_out.to_latex(
                escape=False, index=False, float_format=fmt2,
                column_format="lrrrrrrr",
            )
        )
    print(f"  Tabla exportada: {latex_path}")

    # --- Guardar series de rentabilidad mensual (reproducibilidad) ---
    rents_df = pd.DataFrame(
        {nombre: res["rentabilidades"] for nombre, res in resultados.items()}
    )
    rents_path = os.path.join(OUTPUT_DIR, "rentabilidades_estrategias.csv")
    rents_df.to_csv(rents_path)
    print(f"  Rentabilidades guardadas: {rents_path}")

    # --- Significancia estadística de los ratios de Sharpe ---
    print("\n--- Significancia estadística (IC bootstrap + test de Memmel) ---")
    sig = tabla_significancia(resultados)
    print(sig.to_string())

    sig_latex = sig.copy()
    sig_latex.columns = [
        "Sharpe",
        "IC 95\\% inf.",
        "IC 95\\% sup.",
        "$p$ vs SPY",
        "$p$ vs 1/N",
    ]
    sig_latex = sig_latex.round(3)
    sig_path = os.path.join(OUTPUT_DIR, "tabla_significancia.tex")
    fmt3 = lambda x: f"{x:.3f}".replace(".", ",")
    sig_out = sig_latex.copy()
    sig_out.insert(0, "Estrategia", sig_out.index)
    with open(sig_path, "w") as f:
        f.write(
            sig_out.to_latex(
                escape=False, index=False, na_rep="---", float_format=fmt3,
                column_format="lrrrrr",
            )
        )
    print(f"  Tabla de significancia exportada: {sig_path}")

    # --- Figura: Curva de valor acumulado ---
    print("\nGenerando figuras...")
    fig, ax = plt.subplots(figsize=(10, 5))

    colores = {
        "Markowitz (muestral)": "#2196F3",
        "Markowitz (Ledoit-Wolf)": "#FF9800",
        "Markowitz (PCA)": "#4CAF50",
        "Markowitz + Ridge": "#F44336",
        "Markowitz + Lasso": "#9C27B0",
        "Markowitz + RF": "#795548",
        "1/N (equiponderada)": "#607D8B",
        "SPY (mercado)": "#000000",
    }

    for nombre, res in resultados.items():
        rents = res["rentabilidades"]
        acum = (1 + rents).cumprod()
        color = colores.get(nombre, "#999999")
        lw = 2.0 if "SPY" in nombre or "1/N" in nombre else 1.2
        ls = "--" if "SPY" in nombre or "1/N" in nombre else "-"
        ax.plot(
            acum.index,
            acum.values,
            label=nombre,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=0.9,
        )

    ax.set_title("Evolución de 1 € invertido — Carteras optimizadas vs baselines")
    ax.set_ylabel("Valor acumulado (€)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "valor_acumulado.pdf"))
    plt.close(fig)
    print("  Figura guardada: valor_acumulado.pdf")

    # --- Figura: Sharpe comparado ---
    fig, ax = plt.subplots(figsize=(8, 4))
    nombres_list = list(tabla.index)
    sharpes = tabla["Sharpe"].values
    bars = ax.barh(
        range(len(nombres_list)),
        sharpes,
        color=[colores.get(n, "#999") for n in nombres_list],
    )
    ax.set_yticks(range(len(nombres_list)))
    ax.set_yticklabels(nombres_list, fontsize=9)
    ax.set_xlabel("Ratio de Sharpe (anualizado)")
    ax.set_title("Ratio de Sharpe por estrategia")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sharpe_comparacion.pdf"))
    plt.close(fig)
    print("  Figura guardada: sharpe_comparacion.pdf")

    print("\n--- Backtesting completo ---")


if __name__ == "__main__":
    main()
