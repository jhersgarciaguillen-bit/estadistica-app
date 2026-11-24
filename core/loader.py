"""Carga de datos: archivos del usuario y dataset de ejemplo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Carga un CSV o Excel subido a Streamlit en un DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Usa CSV o Excel (.xlsx/.xls).")

    if df.empty:
        raise ValueError("El archivo no contiene filas de datos.")
    return df


def demo_data() -> pd.DataFrame:
    """Genera un conjunto de datos de ejemplo reproducible."""
    rng = np.random.default_rng(2024)
    n = 150

    grupo = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33])
    medias = {"A": 60.0, "B": 68.0, "C": 75.0}
    puntuacion = np.round(
        np.array([rng.normal(medias[g], 10.0) for g in grupo]), 1
    )
    edad = rng.integers(18, 66, size=n)
    genero = rng.choice(["Masculino", "Femenino"], size=n)
    satisfaccion = rng.integers(1, 11, size=n)
    salario = np.round(
        rng.normal(2500, 700, size=n) + (puntuacion - 60) * 25, 2
    )
    aprobado = np.where(puntuacion >= 65, "Sí", "No")

    return pd.DataFrame(
        {
            "Grupo": grupo,
            "Genero": genero,
            "Edad": edad,
            "Salario": salario,
            "Puntuacion": puntuacion,
            "Satisfaccion": satisfaccion,
            "Aprobado": aprobado,
        }
    )
