"""Estadística inferencial: intervalos de confianza y pruebas de hipótesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _clean(series) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").dropna()


def _z_critical(conf: float) -> float:
    return float(stats.norm.ppf(1.0 - (1.0 - conf) / 2.0))


def _p_from_z(z: float, alternative: str) -> float:
    if alternative == "two-sided":
        return float(2.0 * stats.norm.sf(abs(z)))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    return float(stats.norm.cdf(z))


def interpretar_p(p: float, alpha: float, h0: str = "", h1: str = "") -> str:
    if p < alpha:
        base = (
            f"p = {p:.4g} < α = {alpha:.2g}. Se RECHAZA H0. "
            "Existe evidencia estadística suficiente para respaldar H1."
        )
    else:
        base = (
            f"p = {p:.4g} ≥ α = {alpha:.2g}. No se rechaza H0. "
            "No hay evidencia suficiente para respaldar H1."
        )
    contexto = ""
    if h0 or h1:
        contexto = f"  (H0: {h0} ; H1: {h1})"
    return base + contexto


# --------------------------------------------------------------------------- #
# Intervalos de confianza
# --------------------------------------------------------------------------- #
def ci_mean_t(x_bar: float, s: float, n: int, conf: float = 0.95) -> dict:
    se = s / np.sqrt(n)
    df = n - 1
    tcrit = float(stats.t.ppf(1.0 - (1.0 - conf) / 2.0, df))
    margin = tcrit * se
    return {
        "Método": "t de Student (σ desconocida)",
        "Estimación": x_bar,
        "Error estándar": se,
        "Valor crítico": tcrit,
        "Grados de libertad": df,
        "Margen de error": margin,
        "Límite inferior": x_bar - margin,
        "Límite superior": x_bar + margin,
    }


def ci_mean_z(x_bar: float, sigma: float, n: int, conf: float = 0.95) -> dict:
    se = sigma / np.sqrt(n)
    zcrit = _z_critical(conf)
    margin = zcrit * se
    return {
        "Método": "Normal Z (σ conocida)",
        "Estimación": x_bar,
        "Error estándar": se,
        "Valor crítico": zcrit,
        "Grados de libertad": np.nan,
        "Margen de error": margin,
        "Límite inferior": x_bar - margin,
        "Límite superior": x_bar + margin,
    }


def ci_mean(series=None, x_bar=None, s=None, n=None, sigma=None,
            conf: float = 0.95) -> dict:
    """IC para la media. Si se pasa 'series', calcula desde la muestra."""
    if series is not None:
        data = _clean(series)
        n = int(data.size)
        x_bar = float(data.mean())
        s = float(data.std(ddof=1))
    if sigma is not None:
        return ci_mean_z(x_bar, sigma, n, conf)
    return ci_mean_t(x_bar, s, n, conf)


def ci_proportion(x: float, n: int, conf: float = 0.95,
                  method: str = "wald") -> dict:
    p = x / n
    zcrit = _z_critical(conf)
    if method == "wilson":
        denom = 1.0 + zcrit**2 / n
        centro = (p + zcrit**2 / (2.0 * n)) / denom
        half = zcrit * np.sqrt(p * (1 - p) / n + zcrit**2 / (4.0 * n**2)) / denom
        lower, upper = centro - half, centro + half
        se = np.sqrt(p * (1 - p) / n)
        metodo = "Wilson (proporción)"
    else:
        se = np.sqrt(p * (1 - p) / n)
        half = zcrit * se
        lower, upper = p - half, p + half
        metodo = "Wald (proporción)"
    return {
        "Método": metodo,
        "Estimación": p,
        "Error estándar": se,
        "Valor crítico": zcrit,
        "Margen de error": half,
        "Límite inferior": max(0.0, lower),
        "Límite superior": min(1.0, upper),
    }


def ci_variance(s: float, n: int, conf: float = 0.95) -> dict:
    alpha = 1.0 - conf
    df = n - 1
    chi_lo = float(stats.chi2.ppf(alpha / 2.0, df))
    chi_hi = float(stats.chi2.ppf(1.0 - alpha / 2.0, df))
    var = s**2
    lower = df * var / chi_hi
    upper = df * var / chi_lo
    return {
        "Método": "Chi-cuadrado (varianza)",
        "Estimación (varianza)": var,
        "Grados de libertad": df,
        "Límite inferior (var)": lower,
        "Límite superior (var)": upper,
        "Límite inferior (σ)": np.sqrt(lower),
        "Límite superior (σ)": np.sqrt(upper),
    }


def ci_diff_means(sample1=None, sample2=None, x1=None, s1=None, n1=None,
                  x2=None, s2=None, n2=None, conf: float = 0.95,
                  equal_var: bool = False) -> dict:
    if sample1 is not None and sample2 is not None:
        a, b = _clean(sample1), _clean(sample2)
        x1, s1, n1 = float(a.mean()), float(a.std(ddof=1)), int(a.size)
        x2, s2, n2 = float(b.mean()), float(b.std(ddof=1)), int(b.size)
    diff = x1 - x2
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        metodo = "t (varianzas iguales)"
    else:
        v1, v2 = s1**2 / n1, s2**2 / n2
        se = np.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
        metodo = "t Welch (varianzas distintas)"
    tcrit = float(stats.t.ppf(1.0 - (1.0 - conf) / 2.0, df))
    margin = tcrit * se
    return {
        "Método": metodo,
        "Diferencia (x̄1 - x̄2)": diff,
        "Error estándar": se,
        "Grados de libertad": df,
        "Margen de error": margin,
        "Límite inferior": diff - margin,
        "Límite superior": diff + margin,
    }


def ci_diff_proportions(x1: float, n1: int, x2: float, n2: int,
                        conf: float = 0.95) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = _z_critical(conf)
    margin = zcrit * se
    return {
        "Método": "Wald (diferencia de proporciones)",
        "Diferencia (p̂1 - p̂2)": diff,
        "Error estándar": se,
        "Valor crítico": zcrit,
        "Margen de error": margin,
        "Límite inferior": diff - margin,
        "Límite superior": diff + margin,
    }


# --------------------------------------------------------------------------- #
# Pruebas de hipótesis
# --------------------------------------------------------------------------- #
def t_test_1samp(series, mu0: float, conf: float = 0.95,
                 alternative: str = "two-sided") -> dict:
    s = _clean(series)
    n = int(s.size)
    res = stats.ttest_1samp(s, mu0, alternative=alternative)
    x_bar, sd = float(s.mean()), float(s.std(ddof=1))
    return {
        "Prueba": "t de Student — una muestra",
        "Estadístico": float(res.statistic),
        "p-valor": float(res.pvalue),
        "Grados de libertad": n - 1,
        "Media muestral": x_bar,
        "Desviación estándar": sd,
        "n": n,
        "Media hipotética": mu0,
    }


def z_test_1samp(x_bar: float, sigma: float, n: int, mu0: float,
                 alternative: str = "two-sided") -> dict:
    z = (x_bar - mu0) / (sigma / np.sqrt(n))
    return {
        "Prueba": "Z — una muestra (σ conocida)",
        "Estadístico": float(z),
        "p-valor": _p_from_z(z, alternative),
        "n": n,
        "Media muestral": x_bar,
        "Sigma conocida": sigma,
        "Media hipotética": mu0,
    }


def t_test_ind(sample1=None, sample2=None, x1=None, s1=None, n1=None,
               x2=None, s2=None, n2=None, equal_var: bool = True,
               alternative: str = "two-sided") -> dict:
    if sample1 is not None and sample2 is not None:
        a, b = _clean(sample1), _clean(sample2)
        x1, s1, n1 = float(a.mean()), float(a.std(ddof=1)), int(a.size)
        x2, s2, n2 = float(b.mean()), float(b.std(ddof=1)), int(b.size)
    res = stats.ttest_ind_from_stats(
        x1, s1, n1, x2, s2, n2,
        equal_var=equal_var, alternative=alternative,
    )
    if equal_var:
        df = n1 + n2 - 2
    else:
        v1, v2 = s1**2 / n1, s2**2 / n2
        df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    return {
        "Prueba": "t — dos muestras independientes"
        + (" (varianzas iguales)" if equal_var else " (Welch)"),
        "Estadístico": float(res.statistic),
        "p-valor": float(res.pvalue),
        "Grados de libertad": df,
        "Media 1": x1, "Media 2": x2,
        "n1": n1, "n2": n2,
    }


def t_test_rel(sample1, sample2, alternative: str = "two-sided") -> dict:
    a, b = _clean(sample1), _clean(sample2)
    n = min(a.size, b.size)
    res = stats.ttest_rel(a.iloc[:n], b.iloc[:n], alternative=alternative)
    diff = a.iloc[:n] - b.iloc[:n]
    return {
        "Prueba": "t — muestras pareadas",
        "Estadístico": float(res.statistic),
        "p-valor": float(res.pvalue),
        "Grados de libertad": n - 1,
        "Diferencia media": float(diff.mean()),
        "n pares": n,
    }


def z_test_prop_1(x: float, n: int, p0: float,
                  alternative: str = "two-sided") -> dict:
    p = x / n
    se = np.sqrt(p0 * (1 - p0) / n)
    z = (p - p0) / se
    return {
        "Prueba": "Z — una proporción",
        "Estadístico": float(z),
        "p-valor": _p_from_z(z, alternative),
        "p̂ muestral": p,
        "p0 hipotética": p0,
        "n": n,
    }


def z_test_prop_2(x1: float, n1: int, x2: float, n2: int,
                  alternative: str = "two-sided") -> dict:
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
    z = (p1 - p2) / se
    return {
        "Prueba": "Z — diferencia de proporciones",
        "Estadístico": float(z),
        "p-valor": _p_from_z(z, alternative),
        "p̂1": p1, "p̂2": p2,
        "n1": n1, "n2": n2,
    }


def chi2_gof(observed, expected=None) -> dict:
    obs = np.asarray(observed, dtype=float)
    if expected is None:
        exp = np.full_like(obs, obs.sum() / obs.size)
    else:
        exp = np.asarray(expected, dtype=float)
    res = stats.chisquare(f_obs=obs, f_exp=exp)
    k = obs.size
    return {
        "Prueba": "Chi-cuadrado — bondad de ajuste",
        "Estadístico": float(res.statistic),
        "p-valor": float(res.pvalue),
        "Grados de libertad": k - 1,
        "Observado": obs.tolist() if hasattr(obs, "tolist") else list(obs),
    }


def chi2_independencia(table: pd.DataFrame) -> dict:
    obs = table.to_numpy(dtype=float)
    chi2, p, dof, expected = stats.chi2_contingency(obs)
    return {
        "Prueba": "Chi-cuadrado — independencia",
        "Estadístico": float(chi2),
        "p-valor": float(p),
        "Grados de libertad": int(dof),
        "Tabla esperada": pd.DataFrame(
            expected, index=table.index, columns=table.columns
        ),
    }


def anova_oneway(groups: list) -> dict:
    samples = [_clean(g) for g in groups]
    samples = [s for s in samples if s.size > 0]
    res = stats.f_oneway(*samples)
    k = len(samples)
    n_total = int(sum(s.size for s in samples))
    return {
        "Prueba": "ANOVA de un factor",
        "Estadístico F": float(res.statistic),
        "p-valor": float(res.pvalue),
        "Grados de libertad (entre)": k - 1,
        "Grados de libertad (dentro)": n_total - k,
        "Grupos": k,
        "N total": n_total,
    }


def shapiro(series) -> dict:
    s = _clean(series)
    res = stats.shapiro(s)
    return {
        "Prueba": "Shapiro-Wilk (normalidad)",
        "Estadístico W": float(res.statistic),
        "p-valor": float(res.pvalue),
        "n": int(s.size),
        "Asimetría": float(stats.skew(s, bias=False)),
        "Curtosis": float(stats.kurtosis(s, bias=False)),
    }


def f_test_var(sample1=None, sample2=None, s1=None, n1=None,
               s2=None, n2=None) -> dict:
    if sample1 is not None and sample2 is not None:
        a, b = _clean(sample1), _clean(sample2)
        s1, n1 = float(a.std(ddof=1)), int(a.size)
        s2, n2 = float(b.std(ddof=1)), int(b.size)
    v1, v2 = s1**2, s2**2
    df1, df2 = n1 - 1, n2 - 1
    f = v1 / v2
    p = 2.0 * min(stats.f.cdf(f, df1, df2), stats.f.sf(f, df1, df2))
    return {
        "Prueba": "F — igualdad de varianzas",
        "Estadístico F": float(f),
        "p-valor": float(min(p, 1.0)),
        "Grados de libertad 1": df1,
        "Grados de libertad 2": df2,
        "Varianza 1": v1,
        "Varianza 2": v2,
    }


ALTERNATIVAS = {
    "Bilateral (≠)": "two-sided",
    "Cola izquierda (<)": "less",
    "Cola derecha (>)": "greater",
}
