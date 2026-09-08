import pandas as pd

def falsa_posicion(f, xl, xu, error, max_iter=200, illinois=True):
    """Método de falsa posición con modificación de Illinois opcional."""
    if xl >= xu:
        raise ValueError("Se requiere xl < xu")
    fxl, fxu = f(xl), f(xu)
    if fxl * fxu > 0:
        raise ValueError("f(xl) y f(xu) deben tener signos opuestos")

    iter_data = []
    xr_old = None
    lado_repetido = 0
    iteraciones = 0

    while iteraciones < max_iter:
        denom = fxl - fxu
        if denom == 0:
            raise ZeroDivisionError("f(xl) == f(xu): la fórmula no está definida")
        xr = xu - (fxu * (xl - xu)) / denom
        fxr = f(xr)
        current_error = abs(xr - xr_old) if xr_old is not None else abs(xu - xl)
        iter_data.append({
            "Iteracion": iteraciones + 1,
            "x0": xl, "f(x0)": fxl,
            "x1": xu, "f(x1)": fxu,
            "x2": xr, "f(x2)": fxr,
            "Error": current_error
        })
        if abs(fxr) < 1e-12 or current_error < error:
            break
        if fxl * fxr < 0:
            xu, fxu = xr, fxr
            lado_repetido = lado_repetido + 1 if lado_repetido >= 0 else 1
            if illinois and lado_repetido >= 2:
                fxl /= 2
        else:
            xl, fxl = xr, fxr
            lado_repetido = lado_repetido - 1 if lado_repetido <= 0 else -1
            if illinois and lado_repetido <= -2:
                fxu /= 2
        xr_old = xr
        iteraciones += 1

    return xr, pd.DataFrame(iter_data)