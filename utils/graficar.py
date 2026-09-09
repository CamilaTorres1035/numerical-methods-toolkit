import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importación requerida para la vista 3D

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


def graficar_busqueda_aleatoria(f, rangos, df_iter, mejor_punto, mejor_valor, 
                                 modo="max", resolucion=150,
                                 optimo_teorico=None, valor_teorico=None):
    """
    Gráficas de diagnóstico para búsqueda aleatoria:
      1) Convergencia: mejor valor encontrado hasta cada iteración.
      2) Dispersión 2D: puntos muestreados sobre mapa de contornos de f.
      3) Superficie 3D: visualización tridimensional de la exploración.
    """
    n_variables = len(rangos)
    
    # --- Estructura: 3 paneles si es 2D, 1 panel si es 1D/multivariable genérico ---
    if n_variables == 2:
        fig = plt.figure(figsize=(18, 5))
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(7, 5))
        ax1, ax2 = None, None

    # ============================================================
    # PANEL 1: Convergencia
    # ============================================================
    ax0.scatter(df_iter["Iteracion"], df_iter["f(x)"], 
                color="lightgray", s=8, alpha=0.5, 
                label="Muestras evaluadas", zorder=1)
    ax0.plot(df_iter["Iteracion"], df_iter["mejor_f"], 
             color="blue", linewidth=2, 
             label="Mejor valor acumulado", zorder=3)
    ax0.axhline(mejor_valor, color="red", linestyle="--", linewidth=1, alpha=0.7,
                label=f"Mejor hallado: {mejor_valor:.4f}", zorder=2)
    
    if valor_teorico is not None:
        ax0.axhline(valor_teorico, color="green", linestyle=":", linewidth=1.5, alpha=0.8,
                    label=f"Óptimo teórico: {valor_teorico:.4f}", zorder=2)
    
    etiqueta = "Máximo" if modo == "max" else "Mínimo"
    ax0.set_title("Búsqueda Aleatoria - Convergencia")
    ax0.set_xlabel("Iteración")
    ax0.set_ylabel("f(x)")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    if n_variables == 2:
        (x_min, x_max), (y_min, y_max) = rangos
        xg = np.linspace(x_min, x_max, resolucion)
        yg = np.linspace(y_min, y_max, resolucion)
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = f(Xg, Yg)

        # ============================================================
        # PANEL 2: Mapa de contorno 2D
        # ============================================================
        cont = ax1.contourf(Xg, Yg, Zg, levels=30, cmap="viridis", alpha=0.85)
        fig.colorbar(cont, ax=ax1, label="f(x, y)")

        n_iter = len(df_iter)
        sizes = np.linspace(5, 25, n_iter)
        ax1.scatter(df_iter["x0"], df_iter["x1"], 
                    c=range(n_iter), cmap="YlOrRd", 
                    s=sizes, alpha=0.6, edgecolor="none",
                    label="Puntos muestreados", zorder=2)
        ax1.scatter(mejor_punto[0], mejor_punto[1], 
                    color="red", marker="*", s=350,
                    edgecolor="white", linewidth=1.5, zorder=5, 
                    label=f"{etiqueta} hallado")
        
        if optimo_teorico is not None:
            ax1.scatter(optimo_teorico[0], optimo_teorico[1], 
                        color="lime", marker="D", s=150,
                        edgecolor="black", linewidth=1.5, zorder=5,
                        label="Óptimo teórico")
        
        ax1.set_title("Exploración del espacio (vista 2D)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)

        # ============================================================
        # PANEL 3: Superficie 3D
        # ============================================================
        surf = ax2.plot_surface(Xg, Yg, Zg, cmap="viridis", 
                                 alpha=0.6, edgecolor="none",
                                 antialiased=True)
        
        zs = df_iter["f(x)"].values
        ax2.scatter(df_iter["x0"], df_iter["x1"], zs, 
                    c=zs, cmap="plasma", s=12, alpha=0.7,
                    edgecolor="none", depthshade=True,
                    label="Muestras")
        
        ax2.scatter(mejor_punto[0], mejor_punto[1], mejor_valor, 
                    color="red", marker="*", s=200,
                    edgecolor="white", linewidth=1.5,
                    depthshade=False, zorder=10,
                    label=f"{etiqueta} hallado")
        
        if optimo_teorico is not None and valor_teorico is not None:
            ax2.scatter(optimo_teorico[0], optimo_teorico[1], valor_teorico, 
                        color="lime", marker="D", s=120,
                        edgecolor="black", linewidth=1.5,
                        depthshade=False, zorder=10,
                        label="Óptimo teórico")
        
        # Línea de proyección vertical hasta la base del punto óptimo
        ax2.plot([mejor_punto[0], mejor_punto[0]], 
                 [mejor_punto[1], mejor_punto[1]], 
                 [Zg.min(), mejor_valor], 
                 color="red", linestyle=":", linewidth=1, alpha=0.6)
        
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("f(x, y)")
        ax2.set_title("Exploración 3D de la superficie")
        ax2.view_init(elev=25, azim=-60)
        ax2.legend(loc="upper left", fontsize=7)
        
        fig.colorbar(surf, ax=ax2, shrink=0.6, label="f(x, y)")

    plt.tight_layout()
    return fig