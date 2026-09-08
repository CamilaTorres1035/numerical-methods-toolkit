import pandas as pd

def biseccion(f, xl, xu, error, max_iter=200):
    """Método de bisección para encontrar una raíz de f en [xl, xu]."""
    if xl >= xu:
        raise ValueError("Se requiere xl < xu")
    fxl, fxu = f(xl), f(xu)
    if fxl * fxu > 0:
        raise ValueError("f(xl) y f(xu) deben tener signos opuestos")

    iter_data = []
    xr = (xl + xu) / 2
    iteraciones = 0

    while abs(xu - xl) > error and iteraciones < max_iter:
        xr = (xl + xu) / 2
        fxr = f(xr)
        current_error = abs(xu - xl)
        iter_data.append({
            "Iteracion": iteraciones + 1,
            "x0": xl, "f(x0)": fxl,
            "x1": xu, "f(x1)": fxu,
            "x2": xr, "f(x2)": fxr,
            "Error": current_error
        })
        if fxr == 0:
            break
        elif fxl * fxr < 0:
            xu, fxu = xr, fxr
        else:
            xl, fxl = xr, fxr
        iteraciones += 1

    return xr, pd.DataFrame(iter_data)