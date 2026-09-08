import pandas as pd

def newton_optimizacion(f, df, ddf, x0, error, max_iter=100, tol_segunda_derivada=1e-14):
    """Método de Newton para optimización 1D."""
    iteraciones = 0
    iter_data = []
    xi = x0

    while True:
        ddfx = ddf(xi)
        if abs(ddfx) < tol_segunda_derivada:
            raise ZeroDivisionError(f"f''(x) ≈ 0 en x={xi}: el método diverge")
        x_next = xi - df(xi) / ddfx
        current_error = abs(x_next - xi)
        iter_data.append({
            "Iteracion": iteraciones + 1,
            "x0": xi, "f(x0)": f(xi),
            "x1": x_next, "f(x1)": f(x_next),
            "Error": current_error
        })
        xi = x_next
        iteraciones += 1
        if current_error < error or iteraciones >= max_iter:
            break

    return xi, pd.DataFrame(iter_data)