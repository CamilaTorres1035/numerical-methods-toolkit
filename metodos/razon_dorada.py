import math
import pandas as pd

def razon_dorada(f, xl, xu, error, modo="max", max_iter=200):
    """Método de la sección dorada para optimización 1D."""
    if xl >= xu:
        raise ValueError("Se requiere xl < xu")
    if modo not in ("max", "min"):
        raise ValueError("El parámetro 'modo' debe ser 'max' o 'min'")

    phi = (1 + math.sqrt(5)) / 2
    R = phi - 1
    iteraciones = 0
    iter_data = []

    d = R * (xu - xl)
    x1 = xl + d
    x2 = xu - d
    f1 = f(x1)
    f2 = f(x2)

    while (xu - xl) > error and iteraciones < max_iter:
        current_error = xu - xl
        iter_data.append({
            "Iteracion": iteraciones + 1,
            "x0": xl, "f(x0)": f(xl),
            "x1": xu, "f(x1)": f(xu),
            "x2": x1, "f(x2)": f1,
            "x3": x2, "f(x3)": f2,
            "Error": current_error
        })
        cambiar_limite_derecho = f2 > f1 if modo == "max" else f2 < f1
        if cambiar_limite_derecho:
            xu = x1; x1 = x2; f1 = f2
            d = R * (xu - xl); x2 = xu - d; f2 = f(x2)
        else:
            xl = x2; x2 = x1; f2 = f1
            d = R * (xu - xl); x1 = xl + d; f1 = f(x1)
        iteraciones += 1

    x_opt = (xl + xu) / 2
    return x_opt, pd.DataFrame(iter_data)