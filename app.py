import streamlit as st
import numpy as np
import sympy as sp
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metodos import (
    biseccion, falsa_posicion, razon_dorada, interpolacion,
    newton_raphson, newton_optimizacion, busqueda_aleatoria
)
from utils.graficar import graficar_1d, graficar_busqueda_aleatoria


# ──────────────────────────────────────────────
# Símbolos de SymPy (reutilizables)
# ──────────────────────────────────────────────
x, y = sp.symbols('x y')

# Funciones matemáticas permitidas al parsear
FUNCIONES_PERMITIDAS = {
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
    'sqrt': sp.sqrt, 'abs': sp.Abs,
    'pi': sp.pi, 'e': sp.E,
}


# ──────────────────────────────────────────────
# Helpers con SymPy
# ──────────────────────────────────────────────
def parsear_funcion(expr_str, variables=(x,)):
    """Convierte un string en una expresión simbólica y una función numérica."""
    try:
        expr = sp.sympify(expr_str, locals=FUNCIONES_PERMITIDAS)
    except (sp.SympifyError, TypeError, ValueError) as e:
        raise ValueError(f"Expresión inválida: {e}")
    func_num = sp.lambdify(variables, expr, modules='numpy')
    return expr, func_num


def derivar(expr, var, orden=1):
    """Calcula la derivada simbólica y la convierte a función numérica."""
    derivada = sp.diff(expr, var, orden)
    func_num = sp.lambdify((var,), derivada, modules='numpy')
    return derivada, func_num


def derivar_parcial(expr, var, orden=1):
    """Derivada parcial para funciones multivariable."""
    derivada = sp.diff(expr, var, orden)
    func_num = sp.lambdify((x, y), derivada, modules='numpy')
    return derivada, func_num


def mostrar_expr(label, expr):
    """Renderiza una expresión de SymPy en LaTeX."""
    st.latex(f"{label} = {sp.latex(expr)}")


# ──────────────────────────────────────────────
# Configuración de la página
# ──────────────────────────────────────────────
st.set_page_config(page_title="Métodos de Optimización", page_icon="📐", layout="wide")

st.title("📐 Métodos Numéricos de Optimización")
st.markdown("Selecciona un método en la barra lateral. Las funciones se escriben con notación matemática natural (ej: `sin(x)`, `x**2`, `exp(-x)`).")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.header("Método")
metodo = st.sidebar.selectbox(
    "Elige un método:",
    [
        "Bisección",
        "Falsa Posición",
        "Razón Dorada",
        "Interpolación Cuadrática",
        "Newton-Raphson (raíz)",
        "Newton (optimización)",
        "Búsqueda Aleatoria",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Funciones permitidas:**")
st.sidebar.markdown("`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `pi`, `e`")


# ──────────────────────────────────────────────
# BISECCIÓN
# ──────────────────────────────────────────────
if metodo == "Bisección":
    st.header("Método de Bisección")
    st.markdown("Encuentra una **raíz** de $f(x) = 0$ en un intervalo $[x_l, x_u]$ donde $f$ cambia de signo.")

    col1, col2 = st.columns([1, 1])
    with col1:
        expr_str = st.text_input("f(x) =", value="3*x**2 - 120*x + 100", key="bis_f")
        xl = st.number_input("xl (límite inferior)", value=38.0, key="bis_xl")
        xu = st.number_input("xu (límite superior)", value=40.0, key="bis_xu")
    with col2:
        error = st.number_input("Tolerancia de error", value=0.0001, format="%.6f", key="bis_err")
        max_iter = st.number_input("Máximo de iteraciones", value=200, key="bis_max")

    try:
        expr, _ = parsear_funcion(expr_str, variables=(x,))
        mostrar_expr("f(x)", expr)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if st.button("Calcular", type="primary", key="bis_calc"):
        try:
            _, f = parsear_funcion(expr_str, variables=(x,))
            raiz, df = biseccion(f, xl, xu, error, max_iter)
            st.success(f"✅ Raíz encontrada: **x = {raiz:.6f}** en {len(df)} iteraciones")
            mostrar_expr("f(x_{raíz})", expr.subs(x, raiz))
            st.dataframe(df, use_container_width=True)
            fig = graficar_1d("Bisección", f, (min(xl, raiz)-2, max(xu, raiz)+2), df, raiz)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ──────────────────────────────────────────────
# FALSA POSICIÓN
# ──────────────────────────────────────────────
elif metodo == "Falsa Posición":
    st.header("Método de Falsa Posición")
    st.markdown("Interpolación lineal para encontrar una **raíz**. Incluye la modificación de Illinois.")

    col1, col2 = st.columns([1, 1])
    with col1:
        expr_str = st.text_input("f(x) =", value="3*x**2 - 120*x + 100", key="fp_f")
        xl = st.number_input("xl", value=38.0, key="fp_xl")
        xu = st.number_input("xu", value=40.0, key="fp_xu")
    with col2:
        error = st.number_input("Tolerancia", value=0.0001, format="%.6f", key="fp_err")
        max_iter = st.number_input("Máximo iteraciones", value=200, key="fp_max")
        illinois = st.checkbox("Activar modificación de Illinois", value=True, key="fp_ill")

    try:
        expr, _ = parsear_funcion(expr_str, variables=(x,))
        mostrar_expr("f(x)", expr)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if st.button("Calcular", type="primary", key="fp_calc"):
        try:
            _, f = parsear_funcion(expr_str, variables=(x,))
            raiz, df = falsa_posicion(f, xl, xu, error, max_iter, illinois)
            st.success(f"✅ Raíz encontrada: **x = {raiz:.6f}** en {len(df)} iteraciones")
            st.dataframe(df, use_container_width=True)
            fig = graficar_1d("Falsa Posición", f, (min(xl, raiz)-2, max(xu, raiz)+2), df, raiz)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ──────────────────────────────────────────────
# RAZÓN DORADA
# ──────────────────────────────────────────────
elif metodo == "Razón Dorada":
    st.header("Método de Razón Dorada")
    st.markdown("Encuentra el **óptimo** (máximo o mínimo) de una función unimodal en $[x_l, x_u]$.")

    col1, col2 = st.columns([1, 1])
    with col1:
        expr_str = st.text_input("f(x) =", value="-(x-3)**2 + 10", key="rd_f")
        xl = st.number_input("xl", value=0.0, key="rd_xl")
        xu = st.number_input("xu", value=5.0, key="rd_xu")
    with col2:
        error = st.number_input("Tolerancia", value=0.0001, format="%.6f", key="rd_err")
        max_iter = st.number_input("Máximo iteraciones", value=200, key="rd_max")
        modo = st.selectbox("Modo", ["max", "min"], key="rd_modo")

    try:
        expr, _ = parsear_funcion(expr_str, variables=(x,))
        mostrar_expr("f(x)", expr)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if st.button("Calcular", type="primary", key="rd_calc"):
        try:
            _, f = parsear_funcion(expr_str, variables=(x,))
            optimo, df = razon_dorada(f, xl, xu, error, modo, max_iter)
            etiqueta = "Máximo" if modo == "max" else "Mínimo"
            st.success(f"✅ {etiqueta} encontrado: **x = {optimo:.6f}**, f(x) = {f(optimo):.6f}")
            st.dataframe(df, use_container_width=True)
            fig = graficar_1d("Razón Dorada", f, (xl, xu), df, optimo, optimizacion=True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ──────────────────────────────────────────────
# INTERPOLACIÓN CUADRÁTICA
# ──────────────────────────────────────────────
elif metodo == "Interpolación Cuadrática":
    st.header("Interpolación Cuadrática")
    st.markdown("Ajusta una parábola a 3 puntos $(x_0, x_1, x_2)$ para encontrar el **óptimo**.")

    col1, col2 = st.columns([1, 1])
    with col1:
        expr_str = st.text_input("f(x) =", value="2*sin(x) - (x**2)/10", key="ic_f")
        x0 = st.number_input("x₀", value=0.0, key="ic_x0")
        x1 = st.number_input("x₁", value=1.0, key="ic_x1")
        x2 = st.number_input("x₂", value=4.0, key="ic_x2")
    with col2:
        error = st.number_input("Tolerancia", value=0.0001, format="%.6f", key="ic_err")
        max_iter = st.number_input("Máximo iteraciones", value=200, key="ic_max")
        modo = st.selectbox("Modo", ["max", "min"], key="ic_modo")

    try:
        expr, _ = parsear_funcion(expr_str, variables=(x,))
        mostrar_expr("f(x)", expr)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if st.button("Calcular", type="primary", key="ic_calc"):
        try:
            _, f = parsear_funcion(expr_str, variables=(x,))
            optimo, df = interpolacion(f, x0, x1, x2, error, modo, max_iter)
            etiqueta = "Máximo" if modo == "max" else "Mínimo"
            st.success(f"✅ {etiqueta} encontrado: **x = {optimo:.6f}**, f(x) = {f(optimo):.6f}")
            st.dataframe(df, use_container_width=True)
            fig = graficar_1d("Interpolación", f, (min(x0, optimo)-1, max(x2, optimo)+1), df, optimo, optimizacion=True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ──────────────────────────────────────────────
# NEWTON-RAPHSON (raíz)
# ──────────────────────────────────────────────
elif metodo == "Newton-Raphson (raíz)":
    st.header("Newton-Raphson (raíz)")
    st.markdown("Usa la derivada $f'(x)$ para encontrar una **raíz** de $f(x) = 0$. Converge muy rápido.")

    col1, col2 = st.columns([1, 1])
    with col1:
        expr_str = st.text_input("f(x) =", value="3*x**2 - 120*x + 100", key="nr_f")
        x0 = st.number_input("x₀ (punto inicial)", value=38.0, key="nr_x0")
    with col2:
        error = st.number_input("Tolerancia", value=0.0001, format="%.6f", key="nr_err")
        max_iter = st.number_input("Máximo iteraciones", value=100, key="nr_max")

    try:
        expr, _ = parsear_funcion(expr_str, variables=(x,))
        expr_df, _ = derivar(expr, x, orden=1)
        mostrar_expr("f(x)", expr)
        mostrar_expr("f'(x)", expr_df)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if st.button("Calcular", type="primary", key="nr_calc"):
        try:
            _, f = parsear_funcion(expr_str, variables=(x,))
            _, df_func = derivar(expr, x, orden=1)
            raiz, df = newton_raphson(f, df_func, x0, error, max_iter)
            st.success(f"✅ Raíz encontrada: **x = {raiz:.6f}** en {len(df)} iteraciones")
            st.dataframe(df, use_container_width=True)
            fig = graficar_1d("Newton-Raphson", f, (min(x0, raiz)-2, max(x0, raiz)+2), df, raiz)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ──────────────────────────────────────────────
# NEWTON (optimización)
# ──────────────────────────────────────────────
elif metodo == "Newton (optimización)":
    st.header("Newton para Optimización")
    st.markdown("Usa $f'(x)$ y $f''(x)$ para encontrar donde $f'(x) = 0$ (el **óptimo**).")

    col1, col2 = st.columns([1, 1])
    with col1:
        expr_str = st.text_input("f(x) =", value="2*sin(x) - (x**2)/10", key="no_f")
        x0 = st.number_input("x₀ (punto inicial)", value=2.5, key="no_x0")
    with col2:
        error = st.number_input("Tolerancia", value=0.0001, format="%.6f", key="no_err")
        max_iter = st.number_input("Máximo iteraciones", value=100, key="no_max")

    try:
        expr, _ = parsear_funcion(expr_str, variables=(x,))
        expr_df, _ = derivar(expr, x, orden=1)
        expr_ddf, _ = derivar(expr, x, orden=2)
        mostrar_expr("f(x)", expr)
        mostrar_expr("f'(x)", expr_df)
        mostrar_expr("f''(x)", expr_ddf)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if st.button("Calcular", type="primary", key="no_calc"):
        try:
            _, f = parsear_funcion(expr_str, variables=(x,))
            _, df_func = derivar(expr, x, orden=1)
            _, ddf_func = derivar(expr, x, orden=2)
            optimo, df = newton_optimizacion(f, df_func, ddf_func, x0, error, max_iter)
            etiqueta = "Máximo" if f(optimo) > f(x0) else "Mínimo"
            st.success(f"✅ {etiqueta} encontrado: **x = {optimo:.6f}**, f(x) = {f(optimo):.6f}")
            st.dataframe(df, use_container_width=True)
            fig = graficar_1d("Newton Optimización", f, (min(x0, optimo)-1, max(x0, optimo)+1), df, optimo, optimizacion=True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ──────────────────────────────────────────────
# BÚSQUEDA ALEATORIA
# ──────────────────────────────────────────────
elif metodo == "Búsqueda Aleatoria":
    st.header("Búsqueda Aleatoria")
    st.markdown("Muestreo aleatorio uniforme para optimización de **n variables**.")

    col1, col2 = st.columns([1, 1])
    with col1:
        expr_str = st.text_input("f(x, y) =", value="y - x - 2*x**2 - 2*x*y - y**2", key="ba_f")
        n_iter = st.number_input("Número de iteraciones", value=300, min_value=10, step=50, key="ba_n")
        semilla = st.number_input("Semilla (reproducibilidad)", value=42, key="ba_sem")
    with col2:
        modo = st.selectbox("Modo", ["max", "min"], key="ba_modo")
        st.markdown("**Rangos de búsqueda:**")
        x_min = st.number_input("x mín", value=-2.0, key="ba_xmin")
        x_max = st.number_input("x máx", value=2.0, key="ba_xmax")
        y_min = st.number_input("y mín", value=1.0, key="ba_ymin")
        y_max = st.number_input("y máx", value=3.0, key="ba_ymax")

    try:
        expr, _ = parsear_funcion(expr_str, variables=(x, y))
        mostrar_expr("f(x, y)", expr)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if st.button("Calcular", type="primary", key="ba_calc"):
        try:
            _, f = parsear_funcion(expr_str, variables=(x, y))
            rangos = [(x_min, x_max), (y_min, y_max)]
            mejor_punto, mejor_valor, df = busqueda_aleatoria(
                f, rangos, n_iter, semilla, modo
            )
            etiqueta = "Máximo" if modo == "max" else "Mínimo"
            st.success(
                f"✅ {etiqueta} encontrado:\n\n"
                f"- x = {mejor_punto[0]:.4f}\n"
                f"- y = {mejor_punto[1]:.4f}\n"
                f"- f(x,y) = {mejor_valor:.4f}"
            )
            st.dataframe(df.head(20), use_container_width=True)
            if len(df) > 20:
                st.caption(f"Mostrando las primeras 20 de {len(df)} iteraciones.")
            fig = graficar_busqueda_aleatoria(f, rangos, df, mejor_punto, mejor_valor, modo)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")