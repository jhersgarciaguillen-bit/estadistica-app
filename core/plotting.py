"""Gráficos de apoyo (devuelven figuras de matplotlib)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PRIMARY = "#2563eb"
ACCENT = "#dc2626"
GRID = "#e5e7eb"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def histogram(series, bins: int = 20, with_kde: bool = True):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(
        s,
        bins=bins,
        density=with_kde,
        color=PRIMARY,
        alpha=0.75,
        edgecolor="white",
    )
    if with_kde and s.nunique() > 1:
        kde = stats.gaussian_kde(s)
        xs = np.linspace(s.min(), s.max(), 300)
        ax.plot(xs, kde(xs), color=ACCENT, linewidth=2, label="Densidad (KDE)")
        ax.legend(frameon=False)
    ax.set_ylabel("Densidad" if with_kde else "Frecuencia")
    ax.set_title("Histograma")
    _style(ax)
    fig.tight_layout()
    return fig


def boxplot(series):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.boxplot(s, vert=False, patch_artist=True,
               boxprops={"facecolor": PRIMARY, "alpha": 0.6},
               medianprops={"color": ACCENT, "linewidth": 2})
    ax.set_yticks([])
    ax.set_xlabel("Valor")
    ax.set_title("Diagrama de caja (boxplot)")
    _style(ax)
    fig.tight_layout()
    return fig


def boxplot_groups(df: pd.DataFrame, value_col: str, group_col: str):
    data = df[[value_col, group_col]].dropna()
    labels = list(data[group_col].unique())
    groups = [data.loc[data[group_col] == g, value_col] for g in labels]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.boxplot(
        groups, labels=[str(l) for l in labels], patch_artist=True,
        boxprops={"facecolor": PRIMARY, "alpha": 0.6},
        medianprops={"color": ACCENT, "linewidth": 2},
    )
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} por {group_col}")
    _style(ax)
    fig.tight_layout()
    return fig


def qqplot(series):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    stats.probplot(s, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(PRIMARY)
    ax.get_lines()[0].set_markeredgecolor(PRIMARY)
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color(ACCENT)
    ax.set_title("Q-Q normal")
    _style(ax)
    fig.tight_layout()
    return fig


def bar_frequency(series, top: int | None = None):
    s = pd.Series(series).dropna()
    vc = s.value_counts()
    if top:
        vc = vc.head(top)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(vc.index.astype(str), vc.values, color=PRIMARY, alpha=0.85)
    ax.set_ylabel("Frecuencia")
    ax.set_title("Frecuencias por categoría")
    _style(ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def scatter(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(df[x], df[y], color=PRIMARY, alpha=0.6, edgecolor="white")
    if df[x].notna().sum() > 2:
        coef = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
        xs = np.linspace(df[x].min(), df[x].max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color=ACCENT, linewidth=2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    _style(ax)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 2, 1.1 * len(corr) + 1))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    return fig
