"""Estadística descriptiva: medidas resumen y tablas de frecuencia."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _clean(series) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").dropna()


def describe_numeric(series) -> dict | None:
    """Devuelve un diccionario con las principales medidas descriptivas."""
    s = _clean(series)
    n = int(s.size)
    if n == 0:
        return None

    media = s.mean()
    std = s.std(ddof=1) if n > 1 else np.nan
    var = s.var(ddof=1) if n > 1 else np.nan
    q1 = s.quantile(0.25) if n > 1 else np.nan
    q3 = s.quantile(0.75) if n > 1 else np.nan
    moda = s.mode()
    cv = (std / media * 100.0) if (n > 1 and media != 0) else np.nan

    return {
        "n": n,
        "Media": media,
        "Mediana": s.median(),
        "Moda": moda.iloc[0] if not moda.empty else np.nan,
        "Desviación estándar": std,
        "Varianza": var,
        "Mínimo": s.min(),
        "Máximo": s.max(),
        "Rango": s.max() - s.min(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "Coef. de variación (%)": cv,
        "Asimetría": stats.skew(s, bias=False) if n > 2 else np.nan,
        "Curtosis": stats.kurtosis(s, bias=False) if n > 3 else np.nan,
        "Suma": s.sum(),
    }


def summary_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Tabla resumen para una o varias columnas numéricas."""
    cols = columns or list(df.select_dtypes(include=[np.number]).columns)
    rows = {}
    for col in cols:
        info = describe_numeric(df[col])
        if info:
            rows[col] = info
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


def frequency_table(series: pd.Series, sort_by: str = "Frecuencia") -> pd.DataFrame:
    """Tabla de frecuencias absolutas, relativas y acumuladas."""
    s = pd.Series(series).dropna()
    vc = s.value_counts()
    total = int(vc.sum())
    table = pd.DataFrame(
        {
            "Frecuencia": vc.astype(int),
            "Porcentaje (%)": (vc / total * 100.0).round(2),
        }
    )
    table["Frec. acumulada"] = table["Frecuencia"].cumsum()
    table["% acumulado"] = table["Porcentaje (%)"].cumsum().round(2)
    if sort_by == "Categoría":
        table = table.sort_index()
    table.index.name = s.name
    return table


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in numeric_columns(df)]
