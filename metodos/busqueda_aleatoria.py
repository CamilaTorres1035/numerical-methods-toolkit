import numpy as np
import pandas as pd

def busqueda_aleatoria(f, rangos, n_iteraciones, semilla=42, modo="max"):
    """Búsqueda aleatoria para optimización de n variables."""
    if modo not in ("max", "min"):
        raise ValueError("El parámetro 'modo' debe ser 'max' o 'min'")
    for lo, hi in rangos:
        if lo >= hi:
            raise ValueError(f"Rango inválido {(lo, hi)}")
    if n_iteraciones < 1:
        raise ValueError("n_iteraciones debe ser >= 1")

    rng = np.random.default_rng(semilla)
    n_variables = len(rangos)
    lo = np.array([r[0] for r in rangos])
    hi = np.array([r[1] for r in rangos])

    puntos = lo + (hi - lo) * rng.random((n_iteraciones, n_variables))
    valores = np.array([f(*p) for p in puntos])

    mejor_historico = np.maximum.accumulate(valores) if modo == "max" else np.minimum.accumulate(valores)
    func_extremo = np.argmax if modo == "max" else np.argmin
    idx_mejor_historico = np.array([func_extremo(valores[:i+1]) for i in range(n_iteraciones)])

    filas = []
    for i in range(n_iteraciones):
        error_actual = abs(mejor_historico[i] - mejor_historico[i-1]) if i > 0 else 0.0
        fila = {"Iteracion": i + 1}
        for v in range(n_variables):
            fila[f"x{v}"] = puntos[i, v]
        fila["f(x)"] = valores[i]
        for v in range(n_variables):
            fila[f"mejor_x{v}"] = puntos[idx_mejor_historico[i], v]
        fila["mejor_f"] = mejor_historico[i]
        fila["Error"] = error_actual
        filas.append(fila)

    df_iter = pd.DataFrame(filas)
    idx_final = idx_mejor_historico[-1]
    return puntos[idx_final], mejor_historico[-1], df_iter