import pandas as pd

def newton_raphson(f, df, x0, error, max_iter=100, tol_derivada=1e-14):
    """Método de Newton-Raphson para encontrar una raíz de f."""
    iteraciones = 0
    iter_data = []
    xi = x0

    while True:
        dfx = df(xi)
        if abs(dfx) < tol_derivada:
            raise ZeroDivisionError(f"df(x) ≈ 0 en x={xi}: el método diverge")
        x_next = xi - f(xi) / dfx
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