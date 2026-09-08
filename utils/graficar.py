import numpy as np
import matplotlib.pyplot as plt

def graficar_1d(metodo, f, rango_x, df_iter, result_x, optimizacion=False):
    """Gráfica genérica para métodos de una variable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df_iter['Iteracion'], df_iter['Error'], marker='o', linestyle='-', color='blue')
    ax1.set_title(f'{metodo} - Error por Iteración')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Error')
    ax1.grid(True, alpha=0.4)

    x_vals = np.linspace(rango_x[0], rango_x[1], 400)
    y_vals = f(x_vals)
    ax2.plot(x_vals, y_vals, label='f(x)', color='purple')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    etiqueta = "Óptimo" if optimizacion else "Raíz"
    ax2.axvline(result_x, color="red", linestyle=':', label=f'{etiqueta} x={result_x:.4f}')
    ax2.scatter(result_x, f(result_x), color='red', marker='o', s=50, zorder=5)
    ax2.set_title(f'{metodo} - Función y {etiqueta}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig


def graficar_busqueda_aleatoria(f, rangos, df_iter, mejor_punto, mejor_valor, modo="max", resolucion=150):
    """Gráficas de diagnóstico para búsqueda aleatoria."""
    n_variables = len(rangos)

    if n_variables == 2:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(7, 5))
        ax1 = None

    ax0.scatter(df_iter["Iteracion"], df_iter["f(x)"], color="lightgray", s=8, alpha=0.5, label="Muestras")
    ax0.plot(df_iter["Iteracion"], df_iter["mejor_f"], color="blue", linewidth=2, label="Mejor acumulado")
    ax0.axhline(mejor_valor, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Mejor: {mejor_valor:.4f}")
    ax0.set_title("Búsqueda Aleatoria - Convergencia")
    ax0.set_xlabel("Iteración")
    ax0.set_ylabel("f(x)")
    ax0.legend()
    ax0.grid(True)

    if n_variables == 2 and ax1 is not None:
        (x_min, x_max), (y_min, y_max) = rangos
        xg = np.linspace(x_min, x_max, resolucion)
        yg = np.linspace(y_min, y_max, resolucion)
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = f(Xg, Yg)

        cont = ax1.contourf(Xg, Yg, Zg, levels=25, cmap="viridis", alpha=0.85)
        fig.colorbar(cont, ax=ax1, label="f(x, y)")
        ax1.scatter(df_iter["x0"], df_iter["x1"], c=df_iter["f(x)"], cmap="plasma",
                    edgecolor="black", linewidth=0.3, s=25, label="Puntos muestreados")
        etiqueta = "Máximo" if modo == "max" else "Mínimo"
        ax1.scatter(mejor_punto[0], mejor_punto[1], color="red", marker="*", s=250,
                    edgecolor="black", linewidth=1, zorder=5, label=f"{etiqueta} hallado")
        ax1.set_title("Exploración del espacio")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig