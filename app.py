"""Aplicación web de Estadística Descriptiva e Inferencial (Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import descriptive, inferential, plotting
from core.loader import demo_data, load_dataframe

st.set_page_config(
    page_title="Estadística Descriptiva e Inferencial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    .sidebar-brand {
        background: linear-gradient(135deg, #4f46e5, #7c3aed 60%, #db2777);
        color: #ffffff;
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 8px 20px rgba(79, 70, 229, .25);
    }
    .sidebar-brand .logo { font-size: 1.6rem; }
    .sidebar-brand h1 { color: #fff; font-size: 1.12rem; margin: 6px 0 2px 0; }
    .sidebar-brand p { color: #ede9fe; font-size: .8rem; margin: 0; }

    .data-chip {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0 10px 0;
        font-size: .85rem;
        color: #475569;
    }
    .data-chip b { color: #0f172a; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 16px;
        color: #475569;
        font-weight: 600;
        font-size: .9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #4f46e5;
        box-shadow: 0 1px 3px rgba(15, 23, 42, .08);
    }

    /* Métricas */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
    }
    div[data-testid="stMetricLabel"] p { color: #64748b; font-weight: 600; font-size: .8rem; }
    div[data-testid="stMetricValue"] { color: #0f172a; font-size: 1.35rem; }

    /* Botones */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        font-weight: 600;
        transition: all .15s ease;
    }
    .stButton > button:hover {
        border-color: #4f46e5;
        color: #4f46e5;
        box-shadow: 0 2px 8px rgba(79, 70, 229, .15);
    }

    /* Expanders y contenedores con borde */
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    /* Encabezado de página */
    .page-head { display: flex; gap: 14px; align-items: center; margin: 2px 0 20px 0; }
    .page-head .ico {
        font-size: 1.9rem;
        background: #eef2ff;
        border-radius: 14px;
        padding: 9px 12px;
        line-height: 1;
    }
    .page-head h1 { margin: 0; font-size: 1.65rem; color: #0f172a; letter-spacing: -.02em; }
    .page-head p { margin: 3px 0 0 0; color: #64748b; font-size: .92rem; }

    /* Hero de inicio */
    .hero {
        background: linear-gradient(135deg, #4f46e5, #7c3aed 55%, #db2777);
        color: #fff;
        padding: 34px 38px;
        border-radius: 22px;
        box-shadow: 0 14px 34px rgba(79, 70, 229, .28);
    }
    .hero h1 { color: #fff; margin: 0 0 8px 0; font-size: 2rem; letter-spacing: -.02em; }
    .hero p { color: #ede9fe; margin: 0; font-size: 1.02rem; max-width: 700px; }
    .pill {
        display: inline-block;
        background: rgba(255, 255, 255, .16);
        border: 1px solid rgba(255, 255, 255, .28);
        padding: 5px 13px;
        border-radius: 999px;
        font-size: .78rem;
        margin-top: 16px;
    }

    /* Tarjetas */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
    }
    .card .emoji { font-size: 1.7rem; display: block; margin-bottom: 10px; }
    .card h3 { margin: 0 0 6px 0; font-size: 1.02rem; color: #0f172a; }
    .card p { margin: 0; color: #64748b; font-size: .9rem; line-height: 1.45; }

    .step {
        display: flex; gap: 12px; align-items: flex-start;
        padding: 10px 0; border-bottom: 1px dashed #e2e8f0;
    }
    .step:last-child { border-bottom: none; }
    .step .num {
        flex: 0 0 auto; width: 26px; height: 26px; border-radius: 50%;
        background: #eef2ff; color: #4f46e5; font-weight: 700; font-size: .85rem;
        display: flex; align-items: center; justify-content: center;
    }
    .step .txt { color: #475569; font-size: .92rem; }
    .step .txt b { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Estado y carga de datos
# --------------------------------------------------------------------------- #
if "df" not in st.session_state:
    st.session_state.df = demo_data()
    st.session_state.data_name = "Dataset de ejemplo"


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "—"
        return f"{float(value):,.6g}"
    return str(value)


def page_head(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="page-head">
            <div class="ico">{icon}</div>
            <div>
                <h1>{title}</h1>
                {f'<p>{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resultado(res: dict, alpha: float | None = None) -> None:
    with st.container(border=True):
        st.markdown("#### Resultado")
        if "Límite inferior" in res and "Límite superior" in res:
            est = res.get(
                "Estimación",
                res.get(
                    "Diferencia (x̄1 - x̄2)", res.get("Diferencia (p̂1 - p̂2)")
                ),
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Límite inferior", _fmt(res["Límite inferior"]))
            c2.metric("Estimación", _fmt(est) if est is not None else "—")
            c3.metric("Límite superior", _fmt(res["Límite superior"]))

        if "p-valor" in res:
            c1, c2 = st.columns(2)
            c1.metric("p-valor", f"{res['p-valor']:.6g}")
            rechaza = alpha is not None and res["p-valor"] < alpha
            if alpha is not None:
                c2.metric(
                    "Decisión", "Rechazar H0" if rechaza else "No rechazar H0"
                )
                mensaje = inferential.interpretar_p(res["p-valor"], alpha)
                if rechaza:
                    st.success(mensaje)
                else:
                    st.warning(mensaje)

        detalles = [
            {"Concepto": k, "Valor": _fmt(v)}
            for k, v in res.items()
            if not isinstance(v, pd.DataFrame)
        ]
        with st.expander("Ver detalles del cálculo"):
            st.dataframe(
                pd.DataFrame(detalles), hide_index=True, width="stretch"
            )
        for k, v in res.items():
            if isinstance(v, pd.DataFrame):
                st.caption(k)
                st.dataframe(v, width="stretch")


def num_cols(df: pd.DataFrame) -> list[str]:
    return descriptive.numeric_columns(df)


def cat_cols(df: pd.DataFrame) -> list[str]:
    return descriptive.categorical_columns(df)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="logo">📊</div>
        <h1>Estadística</h1>
        <p>Descriptiva e inferencial</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.df
st.sidebar.markdown(
    f"""
    <div class="data-chip">
        <b>{st.session_state.data_name}</b><br>
        {df.shape[0]} filas · {df.shape[1]} columnas
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar.expander("📁 Cargar datos", expanded=False):
    uploaded = st.file_uploader(
        "Subir CSV o Excel", type=["csv", "xlsx", "xls"]
    )
    if uploaded is not None:
        try:
            st.session_state.df = load_dataframe(uploaded)
            st.session_state.data_name = uploaded.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {exc}")
    if st.button("Cargar datos de ejemplo", width="stretch"):
        st.session_state.df = demo_data()
        st.session_state.data_name = "Dataset de ejemplo"

df = st.session_state.df

seccion = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "🗂️ Datos",
        "📊 Descriptiva",
        "🔬 Inferencial",
    ],
    label_visibility="collapsed",
)

with st.sidebar.expander("❓ Guía rápida"):
    st.markdown(
        "- **Datos**: revisa y carga tu archivo.\n"
        "- **Descriptiva**: resume y grafica variables.\n"
        "- **Inferencial**: intervalos y pruebas de hipótesis.\n\n"
        "Usa el dataset de ejemplo para explorar sin cargar nada."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Hecho con Streamlit · Python")


# --------------------------------------------------------------------------- #
# Página: Inicio
# --------------------------------------------------------------------------- #
def page_inicio(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Estadística clara, sin complicaciones</h1>
            <p>Explora tus datos, calcula medidas descriptivas y realiza
            intervalos de confianza y pruebas de hipótesis desde una sola
            aplicación.</p>
            <span class="pill">CSV · Excel · Datos de ejemplo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    cards = [
        (
            "🗂️",
            "1. Prepara tus datos",
            "Sube un CSV o Excel desde la barra lateral, o usa el dataset "
            "de ejemplo. Revisa tipos de variable y valores nulos.",
        ),
        (
            "📊",
            "2. Describe",
            "Obtén media, mediana, dispersión, cuartiles, frecuencias y "
            "gráficos como histogramas, cajas y correlaciones.",
        ),
        (
            "🔬",
            "3. Infiere",
            "Calcula intervalos de confianza y contrasta hipótesis con "
            "pruebas t, z, ANOVA, chi-cuadrado y normalidad.",
        ),
    ]
    for col, (emoji, title, text) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <span class="emoji">{emoji}</span>
                    <h3>{title}</h3>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("### Dataset actual")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filas", df.shape[0])
    m2.metric("Columnas", df.shape[1])
    m3.metric("Numéricas", len(num_cols(df)))
    m4.metric("Categóricas", len(cat_cols(df)))

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Cómo empezar")
        st.markdown(
            """
            <div class="step"><div class="num">1</div>
              <div class="txt"><b>Carga tus datos</b> o prueba con el ejemplo.</div></div>
            <div class="step"><div class="num">2</div>
              <div class="txt"><b>Explora la sección Descriptiva</b> para entender
              cada variable y sus relaciones.</div></div>
            <div class="step"><div class="num">3</div>
              <div class="txt"><b>Ve a Inferencial</b> para estimar parámetros
              y contrastar hipótesis.</div></div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("##### Vista previa")
        st.dataframe(df.head(8), width="stretch", height=260)


# --------------------------------------------------------------------------- #
# Página: Datos
# --------------------------------------------------------------------------- #
def page_datos(df: pd.DataFrame) -> None:
    page_head(
        "🗂️",
        "Conjunto de datos",
        "Revisa la estructura, los tipos de variable y los valores nulos.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Valores nulos", int(df.isna().sum().sum()))
    c4.metric("Columnas numéricas", len(num_cols(df)))

    tab_prev, tab_tipos, tab_resumen = st.tabs(
        ["Vista previa", "Tipos de variable", "Resumen numérico"]
    )

    with tab_prev:
        st.caption("Primeras 50 filas del conjunto de datos.")
        st.dataframe(df.head(50), width="stretch")

    with tab_tipos:
        st.caption("Clasificación automática de cada columna.")
        tipos = pd.DataFrame(
            {
                "Columna": df.columns,
                "Tipo": [str(t) for t in df.dtypes],
                "No nulos": [int(df[c].notna().sum()) for c in df.columns],
                "Nulos": [int(df[c].isna().sum()) for c in df.columns],
                "Únicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
            }
        )
        st.dataframe(tipos, hide_index=True, width="stretch")

    with tab_resumen:
        tabla = descriptive.summary_table(df)
        if tabla.empty:
            st.info("No hay columnas numéricas.")
        else:
            st.caption("Medidas descriptivas de todas las variables numéricas.")
            st.dataframe(tabla, width="stretch")


# --------------------------------------------------------------------------- #
# Página: Descriptiva
# --------------------------------------------------------------------------- #
def page_descriptiva(df: pd.DataFrame) -> None:
    page_head(
        "📊",
        "Estadística descriptiva",
        "Resume, visualiza y explora el comportamiento de tus variables.",
    )
    ncols = num_cols(df)
    ccols = cat_cols(df)

    tab_resumen, tab_una, tab_rel = st.tabs(
        ["Resumen general", "Una variable", "Relaciones"]
    )

    with tab_resumen:
        with st.container(border=True):
            st.markdown("#### Medidas por variable numérica")
            st.caption(
                "Tabla comparativa con las principales medidas de cada columna."
            )
            tabla = descriptive.summary_table(df)
            if tabla.empty:
                st.info("No hay columnas numéricas.")
            else:
                st.dataframe(tabla, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### Valores nulos")
                nulos = df.isna().sum()
                nulos = nulos[nulos > 0]
                if nulos.empty:
                    st.success("No hay valores nulos.")
                else:
                    st.dataframe(
                        nulos.rename("Nulos").to_frame(), width="stretch"
                    )
        with c2:
            with st.container(border=True):
                st.markdown("#### Variables categóricas")
                if ccols:
                    st.write(", ".join(f"`{c}`" for c in ccols))
                    st.caption("Úsalas para agrupar, comparar y cruzar.")
                else:
                    st.caption("Sin variables categóricas.")

    with tab_una:
        col = st.selectbox(
            "Selecciona una variable",
            list(df.columns),
            help="Elige la columna que quieres describir.",
        )
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            info = descriptive.describe_numeric(serie)
            if info is None:
                st.warning("La variable no tiene datos numéricos válidos.")
                return
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media", _fmt(info["Media"]))
            c2.metric("Mediana", _fmt(info["Mediana"]))
            c3.metric("Desv. estándar", _fmt(info["Desviación estándar"]))
            c4.metric("n", _fmt(info["n"]))

            with st.expander("Ver todas las medidas"):
                medidas = pd.DataFrame(
                    {
                        "Medida": list(info.keys()),
                        "Valor": [_fmt(v) for v in info.values()],
                    }
                )
                st.dataframe(medidas, hide_index=True, width="stretch")

            c_left, c_right = st.columns(2)
            bins = c_left.slider("Número de intervalos del histograma", 5, 60, 20)
            with c_left:
                with st.container(border=True):
                    st.pyplot(plotting.histogram(serie, bins=bins))
            with c_right:
                with st.container(border=True):
                    st.pyplot(plotting.boxplot(serie))
            with st.container(border=True):
                st.pyplot(plotting.qqplot(serie))
                st.caption(
                    "El gráfico Q-Q compara los cuantiles de los datos con "
                    "los de una normal: puntos cerca de la línea = normalidad."
                )
        else:
            with st.container(border=True):
                st.markdown("#### Tabla de frecuencias")
                orden = st.radio(
                    "Ordenar por",
                    ["Frecuencia", "Categoría"],
                    horizontal=True,
                )
                tabla = descriptive.frequency_table(serie, sort_by=orden)
                st.dataframe(tabla, width="stretch")
            with st.container(border=True):
                st.pyplot(plotting.bar_frequency(serie))

    with tab_rel:
        if len(ncols) >= 2:
            with st.container(border=True):
                st.markdown("#### Matriz de correlación")
                st.caption(
                    "Valores cercanos a ±1 indican relación lineal fuerte."
                )
                fig = plotting.correlation_heatmap(df)
                if fig is not None:
                    st.pyplot(fig)

            with st.container(border=True):
                st.markdown("#### Dispersión entre dos variables")
                c1, c2 = st.columns(2)
                x = c1.selectbox("Variable X", ncols, index=0)
                y = c2.selectbox(
                    "Variable Y", ncols, index=min(1, len(ncols) - 1)
                )
                if x != y:
                    st.pyplot(plotting.scatter(df[[x, y]].dropna(), x, y))
                    r = df[[x, y]].dropna().corr().iloc[0, 1]
                    st.caption(f"Correlación de Pearson r = {r:.4f}")
                else:
                    st.info("Selecciona dos variables distintas.")
        else:
            st.info("Se necesitan al menos dos variables numéricas.")

        if len(ccols) >= 2:
            with st.container(border=True):
                st.markdown("#### Tabla de contingencia")
                st.caption("Cruza dos variables categóricas.")
                c1, c2 = st.columns(2)
                a = c1.selectbox("Fila", ccols, index=0)
                b = c2.selectbox(
                    "Columna", ccols, index=min(1, len(ccols) - 1)
                )
                if a != b:
                    st.dataframe(
                        pd.crosstab(df[a], df[b]), width="stretch"
                    )


# --------------------------------------------------------------------------- #
# Página: Inferencial
# --------------------------------------------------------------------------- #
CI_DESC = {
    "Media": "Estima el promedio poblacional a partir de una muestra.",
    "Proporción": "Estima la proporción poblacional de éxitos.",
    "Varianza": "Estima la variabilidad poblacional (varianza y desviación).",
    "Diferencia de medias": "Estima cuánto difieren las medias de dos grupos.",
    "Diferencia de proporciones": "Estima la diferencia entre dos proporciones.",
}

TEST_DESC = {
    "t — una muestra (media)": "Compara la media de una muestra con un valor μ0.",
    "Z — una muestra (σ conocida)": "Igual que la t, pero con σ poblacional conocida.",
    "t — dos muestras independientes": "Compara las medias de dos grupos independientes.",
    "t — muestras pareadas": "Compara dos mediciones sobre las mismas unidades.",
    "Z — una proporción": "Contrasta una proporción muestral frente a p0.",
    "Z — dos proporciones": "Compara las proporciones de dos grupos.",
    "F — igualdad de varianzas": "Compara la variabilidad de dos grupos.",
}


def _selector_datos(prefix: str, ncols: list[str]):
    modo = st.radio(
        "Origen de los datos",
        ["Columna del dataset", "Estadísticos muestrales"],
        horizontal=True,
        key=f"{prefix}_modo",
    )
    if modo == "Columna del dataset":
        if not ncols:
            st.warning("No hay columnas numéricas.")
            return None
        col = st.selectbox("Columna", ncols, key=f"{prefix}_col")
        return {"column": col}
    return {"manual": True}


def _inputs_manual(prefix: str, need_std: bool = True):
    c1, c2, c3, _ = st.columns(4)
    x_bar = c1.number_input("Media (x̄)", value=0.0, key=f"{prefix}_xbar")
    n = int(
        c2.number_input("n", min_value=2, value=30, step=1, key=f"{prefix}_n")
    )
    s = (
        c3.number_input(
            "Desv. estándar (s)", min_value=0.0, value=1.0, key=f"{prefix}_s"
        )
        if need_std
        else None
    )
    return x_bar, s, n


def page_inferencial(df: pd.DataFrame) -> None:
    page_head(
        "🔬",
        "Estadística inferencial",
        "Estima parámetros y contrasta hipótesis a partir de los datos.",
    )
    ncols = num_cols(df)
    ccols = cat_cols(df)

    tab_ic, tab_pruebas, tab_anova, tab_chi, tab_norm = st.tabs(
        [
            "Intervalos de confianza",
            "Pruebas de hipótesis",
            "ANOVA",
            "Chi-cuadrado",
            "Normalidad",
        ]
    )

    # --------------------------- Intervalos -------------------------------- #
    with tab_ic:
        st.markdown("#### Intervalo de confianza")
        st.caption(
            "Elige el parámetro que quieres estimar y el nivel de confianza."
        )
        tipo = st.selectbox(
            "Parámetro",
            list(CI_DESC),
            key="ic_tipo",
            help="El parámetro poblacional que deseas estimar.",
        )
        st.info(CI_DESC[tipo])
        conf = st.slider(
            "Nivel de confianza",
            0.80,
            0.999,
            0.95,
            0.001,
            format="%.3f",
            key="ic_conf",
        )

        if tipo == "Media":
            origen = _selector_datos("ic_media", ncols)
            if origen is None:
                return
            conocido = st.checkbox(
                "Se conoce la desviación poblacional σ", key="ic_sigma"
            )
            if "column" in origen:
                serie = df[origen["column"]].dropna()
                sigma = None
                if conocido:
                    sigma = st.number_input(
                        "σ poblacional",
                        min_value=0.0,
                        value=1.0,
                        key="ic_sigma_val",
                    )
                res = inferential.ci_mean(series=serie, sigma=sigma, conf=conf)
            else:
                x_bar, s, n = _inputs_manual(
                    "ic_media", need_std=not conocido
                )
                sigma = None
                if conocido:
                    sigma = st.number_input(
                        "σ poblacional",
                        min_value=0.0,
                        value=1.0,
                        key="ic_sigma_val",
                    )
                res = inferential.ci_mean(
                    x_bar=x_bar, s=s, n=n, sigma=sigma, conf=conf
                )
            render_resultado(res)

        elif tipo == "Proporción":
            c1, c2 = st.columns(2)
            x = c1.number_input(
                "Éxitos (x)", min_value=0.0, value=10.0, step=1.0
            )
            n = c2.number_input("n", min_value=1.0, value=50.0, step=1.0)
            metodo = st.selectbox(
                "Método",
                ["wald", "wilson"],
                help="Wilson es más fiable con muestras pequeñas.",
            )
            if x > n:
                st.error("Los éxitos no pueden superar a n.")
            else:
                render_resultado(
                    inferential.ci_proportion(x, int(n), conf, method=metodo)
                )

        elif tipo == "Varianza":
            c1, c2 = st.columns(2)
            n = int(c1.number_input("n", min_value=2, value=30, step=1))
            s = c2.number_input(
                "Desv. estándar muestral (s)", min_value=0.0, value=1.0
            )
            render_resultado(inferential.ci_variance(s, n, conf))

        elif tipo == "Diferencia de medias":
            modo = st.radio(
                "Origen", ["Dos columnas", "Estadísticos"], horizontal=True
            )
            if modo == "Dos columnas":
                if len(ncols) < 2:
                    st.warning("Se requieren al menos dos columnas numéricas.")
                else:
                    c1, c2 = st.columns(2)
                    a = c1.selectbox("Grupo 1", ncols, index=0)
                    b = c2.selectbox("Grupo 2", ncols, index=1)
                    iguales = st.checkbox("Asumir varianzas iguales")
                    if a != b:
                        render_resultado(
                            inferential.ci_diff_means(
                                sample1=df[a],
                                sample2=df[b],
                                conf=conf,
                                equal_var=iguales,
                            )
                        )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    x1 = st.number_input("Media 1", value=0.0)
                    s1 = st.number_input("Desv. 1", min_value=0.0, value=1.0)
                    n1 = int(st.number_input("n1", min_value=2, value=30))
                with c2:
                    x2 = st.number_input("Media 2", value=0.0)
                    s2 = st.number_input("Desv. 2", min_value=0.0, value=1.0)
                    n2 = int(st.number_input("n2", min_value=2, value=30))
                iguales = st.checkbox("Asumir varianzas iguales")
                render_resultado(
                    inferential.ci_diff_means(
                        x1=x1,
                        s1=s1,
                        n1=n1,
                        x2=x2,
                        s2=s2,
                        n2=n2,
                        conf=conf,
                        equal_var=iguales,
                    )
                )

        else:  # Diferencia de proporciones
            c1, c2 = st.columns(2)
            with c1:
                x1 = st.number_input(
                    "Éxitos grupo 1", min_value=0.0, value=20.0
                )
                n1 = st.number_input("n grupo 1", min_value=1.0, value=100.0)
            with c2:
                x2 = st.number_input(
                    "Éxitos grupo 2", min_value=0.0, value=30.0
                )
                n2 = st.number_input("n grupo 2", min_value=1.0, value=100.0)
            if x1 <= n1 and x2 <= n2:
                render_resultado(
                    inferential.ci_diff_proportions(
                        x1, int(n1), x2, int(n2), conf
                    )
                )
            else:
                st.error("Los éxitos no pueden superar a n.")

    # --------------------------- Pruebas ----------------------------------- #
    with tab_pruebas:
        st.markdown("#### Pruebas de hipótesis")
        st.caption(
            "Selecciona la prueba, define la hipótesis alternativa y el "
            "nivel de significancia α."
        )
        prueba = st.selectbox(
            "Prueba",
            list(TEST_DESC),
            key="prueba_tipo",
            help="Elige según el tipo de variable y el número de grupos.",
        )
        st.info(TEST_DESC[prueba])

        c1, c2 = st.columns(2)
        alpha = c1.slider(
            "Nivel de significancia α", 0.001, 0.10, 0.05, 0.001
        )
        alt_label = c2.selectbox(
            "Hipótesis alternativa", list(inferential.ALTERNATIVAS)
        )
        alternative = inferential.ALTERNATIVAS[alt_label]

        with st.expander("🧭 ¿Qué prueba debo elegir?"):
            st.markdown(
                "- **Media de un grupo** → t de una muestra.\n"
                "- **Dos grupos independientes** → t de dos muestras.\n"
                "- **Antes/después en las mismas unidades** → t pareada.\n"
                "- **Proporciones** → Z de una o dos proporciones.\n"
                "- **Más de dos medias** → ANOVA.\n"
                "- **Relación entre categóricas** → Chi-cuadrado."
            )

        if prueba == "t — una muestra (media)":
            if not ncols:
                st.warning("No hay columnas numéricas.")
            else:
                col = st.selectbox("Columna", ncols)
                mu0 = st.number_input("Media hipotética (μ0)", value=0.0)
                res = inferential.t_test_1samp(
                    df[col], mu0, alternative=alternative
                )
                st.caption(f"H0: μ = {mu0}  ·  H1: alternativa {alt_label}")
                render_resultado(res, alpha)

        elif prueba == "Z — una muestra (σ conocida)":
            x_bar, _, n = _inputs_manual("z1", need_std=False)
            c1, c2 = st.columns(2)
            sigma = c1.number_input(
                "σ conocida", min_value=0.0, value=1.0
            )
            mu0 = c2.number_input("Media hipotética (μ0)", value=0.0)
            if sigma > 0:
                res = inferential.z_test_1samp(
                    x_bar, sigma, n, mu0, alternative=alternative
                )
                st.caption(f"H0: μ = {mu0}  ·  H1: alternativa {alt_label}")
                render_resultado(res, alpha)
            else:
                st.error("σ debe ser mayor que 0.")

        elif prueba == "t — dos muestras independientes":
            if len(ncols) < 2:
                st.warning("Se requieren al menos dos columnas numéricas.")
            else:
                c1, c2 = st.columns(2)
                a = c1.selectbox("Grupo 1", ncols, index=0)
                b = c2.selectbox("Grupo 2", ncols, index=1)
                iguales = st.checkbox("Asumir varianzas iguales")
                if a != b:
                    res = inferential.t_test_ind(
                        sample1=df[a],
                        sample2=df[b],
                        equal_var=iguales,
                        alternative=alternative,
                    )
                    st.caption(f"H0: μ1 = μ2  ·  H1: alternativa {alt_label}")
                    render_resultado(res, alpha)

        elif prueba == "t — muestras pareadas":
            if len(ncols) < 2:
                st.warning("Se requieren dos columnas numéricas pareadas.")
            else:
                c1, c2 = st.columns(2)
                a = c1.selectbox("Medición 1", ncols, index=0)
                b = c2.selectbox("Medición 2", ncols, index=1)
                if a != b:
                    res = inferential.t_test_rel(
                        df[a], df[b], alternative=alternative
                    )
                    st.caption(f"H0: μd = 0  ·  H1: alternativa {alt_label}")
                    render_resultado(res, alpha)

        elif prueba == "Z — una proporción":
            c1, c2, c3 = st.columns(3)
            x = c1.number_input("Éxitos (x)", min_value=0.0, value=10.0)
            n = c2.number_input("n", min_value=1.0, value=50.0)
            p0 = c3.number_input(
                "p0", min_value=0.0, max_value=1.0, value=0.5
            )
            if 0 < x <= n:
                res = inferential.z_test_prop_1(
                    x, int(n), p0, alternative=alternative
                )
                st.caption(f"H0: p = {p0}  ·  H1: alternativa {alt_label}")
                render_resultado(res, alpha)
            else:
                st.error("Revisa los valores: 0 ≤ x ≤ n.")

        elif prueba == "Z — dos proporciones":
            c1, c2 = st.columns(2)
            with c1:
                x1 = st.number_input(
                    "Éxitos grupo 1", min_value=0.0, value=20.0
                )
                n1 = st.number_input("n grupo 1", min_value=1.0, value=100.0)
            with c2:
                x2 = st.number_input(
                    "Éxitos grupo 2", min_value=0.0, value=30.0
                )
                n2 = st.number_input("n grupo 2", min_value=1.0, value=100.0)
            if x1 <= n1 and x2 <= n2:
                res = inferential.z_test_prop_2(
                    x1, int(n1), x2, int(n2), alternative=alternative
                )
                st.caption(f"H0: p1 = p2  ·  H1: alternativa {alt_label}")
                render_resultado(res, alpha)

        else:  # F varianzas
            if len(ncols) < 2:
                st.warning("Se requieren dos columnas numéricas.")
            else:
                c1, c2 = st.columns(2)
                a = c1.selectbox("Muestra 1", ncols, index=0)
                b = c2.selectbox("Muestra 2", ncols, index=1)
                if a != b:
                    res = inferential.f_test_var(
                        sample1=df[a], sample2=df[b]
                    )
                    st.caption("H0: σ1² = σ2²  ·  H1: σ1² ≠ σ2²")
                    render_resultado(res, alpha)

    # --------------------------- ANOVA ------------------------------------- #
    with tab_anova:
        st.markdown("#### ANOVA de un factor")
        st.caption(
            "Compara las medias de tres o más grupos definidos por una "
            "variable categórica."
        )
        if not ncols or not ccols:
            st.warning("Se requiere una variable numérica y una categórica.")
        else:
            with st.container(border=True):
                c1, c2 = st.columns(2)
                valor = c1.selectbox(
                    "Variable numérica (respuesta)", ncols
                )
                grupo = c2.selectbox("Variable de grupo", ccols)
                data = df[[valor, grupo]].dropna()
                grupos = [
                    g[valor].to_numpy()
                    for _, g in data.groupby(grupo)
                    if len(g) > 0
                ]
                if len(grupos) < 2:
                    st.warning("Se necesitan al menos dos grupos con datos.")
                else:
                    resumen = (
                        data.groupby(grupo)[valor]
                        .agg(
                            n="count",
                            Media="mean",
                            Desv="std",
                            Mín="min",
                            Máx="max",
                        )
                        .rename_axis("Grupo")
                    )
                    st.dataframe(resumen, width="stretch")
                    st.pyplot(plotting.boxplot_groups(data, valor, grupo))
                    alpha_anova = st.slider(
                        "α", 0.001, 0.10, 0.05, 0.001, key="anova_alpha"
                    )
                    res = inferential.anova_oneway(grupos)
                    st.caption(
                        f"H0: todas las medias de {valor} son iguales"
                    )
                    render_resultado(res, alpha_anova)

    # --------------------------- Chi-cuadrado ------------------------------ #
    with tab_chi:
        st.markdown("#### Pruebas Chi-cuadrado")
        st.caption(
            "Comprueba si una distribución se ajusta a lo esperado o si dos "
            "variables categóricas son independientes."
        )
        modo = st.radio(
            "Tipo", ["Bondad de ajuste", "Independencia"], horizontal=True
        )

        if modo == "Bondad de ajuste":
            if not ccols:
                st.warning("No hay variables categóricas.")
            else:
                with st.container(border=True):
                    col = st.selectbox("Variable categórica", ccols)
                    conteos = df[col].value_counts()
                    st.write("Frecuencias observadas:")
                    st.dataframe(
                        conteos.rename("Observado").to_frame(),
                        width="stretch",
                    )
                    uniforme = st.checkbox("Esperado uniforme", value=True)
                    if uniforme:
                        res = inferential.chi2_gof(conteos.to_numpy())
                    else:
                        st.caption(
                            "Introduce las frecuencias esperadas en el mismo "
                            "orden."
                        )
                        esperado = []
                        for cat in conteos.index:
                            esperado.append(
                                st.number_input(
                                    f"Esperado — {cat}",
                                    min_value=0.0,
                                    value=1.0,
                                    key=f"exp_{cat}",
                                )
                            )
                        res = inferential.chi2_gof(
                            conteos.to_numpy(), esperado
                        )
                    alpha_chi = st.slider(
                        "α", 0.001, 0.10, 0.05, 0.001, key="chi_alpha"
                    )
                    render_resultado(res, alpha_chi)

        else:
            if len(ccols) < 2:
                st.warning("Se requieren al menos dos variables categóricas.")
            else:
                with st.container(border=True):
                    c1, c2 = st.columns(2)
                    a = c1.selectbox(
                        "Variable fila", ccols, index=0, key="chi_fila"
                    )
                    b = c2.selectbox(
                        "Variable columna",
                        ccols,
                        index=min(1, len(ccols) - 1),
                        key="chi_col",
                    )
                    if a != b:
                        tabla = pd.crosstab(df[a], df[b])
                        st.write("Tabla de contingencia:")
                        st.dataframe(tabla, width="stretch")
                        alpha_chi = st.slider(
                            "α", 0.001, 0.10, 0.05, 0.001, key="chi_alpha2"
                        )
                        res = inferential.chi2_independencia(tabla)
                        st.caption(f"H0: {a} y {b} son independientes")
                        render_resultado(res, alpha_chi)

    # --------------------------- Normalidad -------------------------------- #
    with tab_norm:
        st.markdown("#### Prueba de normalidad (Shapiro-Wilk)")
        st.caption(
            "Contrasta si los datos provienen de una distribución normal, "
            "requisito de muchas pruebas paramétricas."
        )
        if not ncols:
            st.warning("No hay columnas numéricas.")
        else:
            col = st.selectbox("Columna", ncols, key="norm_col")
            serie = df[col].dropna()
            if serie.size < 3:
                st.warning("Se necesitan al menos 3 observaciones.")
            else:
                alpha_n = st.slider(
                    "α", 0.001, 0.10, 0.05, 0.001, key="norm_alpha"
                )
                res = inferential.shapiro(serie)
                st.caption(
                    "H0: los datos provienen de una distribución normal"
                )
                render_resultado(res, alpha_n)
                with st.container(border=True):
                    st.pyplot(plotting.qqplot(serie))
                if res["p-valor"] >= alpha_n:
                    st.info(
                        "No se rechaza la normalidad: los datos son "
                        "compatibles con una distribución normal."
                    )
                else:
                    st.info(
                        "Se rechaza la normalidad: los datos no parecen "
                        "seguir una distribución normal."
                    )


# --------------------------------------------------------------------------- #
# Enrutamiento
# --------------------------------------------------------------------------- #
if seccion == "🏠 Inicio":
    page_inicio(df)
elif seccion == "🗂️ Datos":
    page_datos(df)
elif seccion == "📊 Descriptiva":
    page_descriptiva(df)
else:
    page_inferencial(df)
