"""
Significancia estadística de las diferencias de rendimiento entre estrategias.

Comparar ratios de Sharpe a partir de una sola muestra (143 meses) sin medir la
incertidumbre es engañoso: una diferencia de 0,05 en el Sharpe puede ser puro
ruido muestral. Este módulo aporta dos herramientas:

  1. Test de Jobson-Korkie con la corrección de Memmel (2003) para contrastar si
     la diferencia entre dos ratios de Sharpe es estadísticamente distinta de
     cero (H0: Sharpe iguales). Es el test estándar en la literatura de carteras
     (lo usan, p. ej., DeMiguel, Garlappi y Uppal, 2009).

  2. Intervalos de confianza del ratio de Sharpe por bootstrap de bloques, que
     no presupone normalidad de las rentabilidades.

Convención: el test trabaja con el ratio de Sharpe por periodo (mensual,
media/desviación típica). Para presentarlo, se anualiza multiplicando por
sqrt(12).
"""

import numpy as np
import pandas as pd
from scipy import stats

PERIODOS_ANIO = 12


def sharpe_mensual(r: np.ndarray) -> float:
    """Ratio de Sharpe por periodo (media / desviación típica muestral, rf=0)."""
    sd = r.std(ddof=1)
    return r.mean() / sd if sd > 0 else 0.0


def sharpe_anualizado(rentabilidades: pd.Series) -> float:
    """Sharpe anualizado a partir de rentabilidades mensuales (media/σ · sqrt(12))."""
    r = np.asarray(rentabilidades, dtype=float)
    return sharpe_mensual(r) * np.sqrt(PERIODOS_ANIO)


def jobson_korkie_memmel(r1: pd.Series, r2: pd.Series) -> dict:
    """
    Contrasta H0: Sharpe(r1) = Sharpe(r2) (test de Jobson-Korkie, corrección de Memmel).

    Las series se alinean por fechas comunes antes del cálculo.

    Returns
    -------
    dict con:
        diff_sharpe : diferencia de Sharpe anualizado (r1 - r2).
        z           : estadístico de contraste (~N(0,1) bajo H0).
        p_value     : p-valor bilateral.
    """
    df = pd.concat([r1, r2], axis=1, join="inner").dropna()
    a = df.iloc[:, 0].to_numpy(dtype=float)
    b = df.iloc[:, 1].to_numpy(dtype=float)
    T = len(a)

    sh1, sh2 = sharpe_mensual(a), sharpe_mensual(b)
    rho = np.corrcoef(a, b)[0, 1]

    # Varianza asintótica de la diferencia de Sharpe (Memmel, 2003)
    theta = (1.0 / T) * (
        2 - 2 * rho + 0.5 * (sh1**2 + sh2**2 - 2 * sh1 * sh2 * rho**2)
    )
    z = (sh1 - sh2) / np.sqrt(theta) if theta > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "diff_sharpe": (sh1 - sh2) * np.sqrt(PERIODOS_ANIO),
        "z": z,
        "p_value": p,
    }


def bootstrap_sharpe_ci(
    rentabilidades: pd.Series,
    n_boot: int = 10000,
    block: int = 3,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Intervalo de confianza del Sharpe anualizado por bootstrap de bloques circular.

    El bootstrap de bloques (longitud `block` meses) preserva la posible
    autocorrelación de las rentabilidades, a diferencia del bootstrap i.i.d.

    Returns
    -------
    (lo, hi) : extremos del intervalo de confianza al (1 - alpha).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(rentabilidades, dtype=float)
    T = len(r)
    n_blocks = int(np.ceil(T / block))

    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
        muestra = r[idx[:T]]
        sharpes[i] = sharpe_mensual(muestra) * np.sqrt(PERIODOS_ANIO)

    lo, hi = np.percentile(sharpes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def tabla_significancia(
    resultados: dict,
    referencias: tuple[str, ...] = ("SPY (mercado)", "1/N (equiponderada)"),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Construye la tabla de significancia para todas las estrategias.

    Para cada estrategia: Sharpe anualizado, IC 95 % por bootstrap, y el p-valor
    del test de Memmel frente a cada cartera de referencia.

    Parameters
    ----------
    resultados : dict
        Salida de run_all_strategies + baselines (clave -> dict con
        "rentabilidades").
    referencias : tuple[str]
        Nombres de las estrategias de referencia para el contraste pareado.

    Returns
    -------
    pd.DataFrame indexado por estrategia.
    """
    filas = {}
    for nombre, res in resultados.items():
        r = res["rentabilidades"]
        lo, hi = bootstrap_sharpe_ci(r, seed=seed)
        fila = {
            "Sharpe": sharpe_anualizado(r),
            "IC 95% inf": lo,
            "IC 95% sup": hi,
        }
        for ref in referencias:
            if nombre == ref:
                fila[f"p vs {ref.split()[0]}"] = np.nan
            else:
                test = jobson_korkie_memmel(r, resultados[ref]["rentabilidades"])
                fila[f"p vs {ref.split()[0]}"] = test["p_value"]
        filas[nombre] = fila

    return pd.DataFrame(filas).T
