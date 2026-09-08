import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

def parsear_funcion(expr_str, variables=(x,)):
    """
    Convierte un string en una función numérica evaluables con NumPy.
    
    Args:
        expr_str: string con la expresión, ej: "sin(x) + x**2"
        variables: tupla de símbolos de SymPy que usa la función
    
    Returns:
        (expr_simbolica, funcion_numerica)
    """
    # Diccionario de funciones permitidas (seguridad + notación natural)
    funciones_permitidas = {
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
        'sqrt': sp.sqrt, 'abs': sp.Abs,
        'pi': sp.pi, 'e': sp.E,
    }
    
    try:
        expr = sp.sympify(expr_str, locals=funciones_permitidas)
    except sp.SympifyError as e:
        raise ValueError(f"Expresión inválida: {e}")
    
    func_num = sp.lambdify(variables, expr, modules='numpy')
    return expr, func_num

def calcular_derivada(expr, var, orden=1):
    """
    Calcula la derivada simbólica de una expresión.
    
    Args:
        expr: expresión simbólica de SymPy
        var: variable respecto a la cual derivar
        orden: orden de la derivada (1, 2, ...)
    
    Returns:
        (expr_derivada_simbolica, funcion_derivada_numerica)
    """
    derivada = sp.diff(expr, var, orden)
    func_num = sp.lambdify((var,), derivada, modules='numpy')
    return derivada, func_num

def calcular_derivada_parcial(expr, var, orden=1):
    """Derivada parcial para funciones multivariable."""
    derivada = sp.diff(expr, var, orden)
    func_num = sp.lambdify((x, y), derivada, modules='numpy')
    return derivada, func_num