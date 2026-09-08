import pandas as pd

def interpolacion(f, x0, x1, x2, error, modo="max", max_iter=200, tol_den=1e-14):
    """Interpolación cuadrática para optimización 1D."""
    if modo not in ("max", "min"):
        raise ValueError("El parámetro 'modo' debe ser 'max' o 'min'")

    iter_data = []
    x3_old = None
    iteraciones = 0
    x3 = None

    while iteraciones < max_iter:
        f0, f1, f2 = f(x0), f(x1), f(x2)
        den = 2*f0*(x1 - x2) + 2*f1*(x2 - x0) + 2*f2*(x0 - x1)
        if abs(den) < tol_den:
            raise ZeroDivisionError("Los tres puntos son (casi) colineales")
        num = f0*(x1**2 - x2**2) + f1*(x2**2 - x0**2) + f2*(x0**2 - x1**2)
        x3 = num / den
        f3 = f(x3)
        current_error = abs(x3 - x3_old) if x3_old is not None else abs(x2 - x0)
        iter_data.append({
            "Iteracion": iteraciones + 1,
            "x0": x0, "f(x0)": f0,
            "x1": x1, "f(x1)": f1,
            "x2": x2, "f(x2)": f2,
            "x3": x3, "f(x3)": f3,
            "Error": current_error
        })
        if current_error < error:
            break
        mejora = f3 > f1 if modo == "max" else f3 < f1
        if mejora:
            if x3 > x1: x0 = x1
            else: x2 = x1
        else:
            if x3 > x1: x2 = x3
            else: x0 = x3
        x1 = x3
        x3_old = x3
        iteraciones += 1

    return x3, pd.DataFrame(iter_data)