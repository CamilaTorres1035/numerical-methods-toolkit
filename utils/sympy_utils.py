import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

FUNCIONES_PERMITIDAS = {
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
    'sqrt': sp.sqrt, 'abs': sp.Abs,
    'pi': sp.pi, 'e': sp.E,
}

def parsear_funcion(expr_str, variables=(x,)):
    try:
        expr = sp.sympify(expr_str, locals=FUNCIONES_PERMITIDAS)
    except (sp.SympifyError, TypeError, ValueError) as e:
        raise ValueError(f"Expresión inválida: {e}")
    
    func_num = sp.lambdify(variables, expr, modules='numpy')
    return expr, func_num

def parsear_funcion_dinamica(expr_str):
    """
    Detecta automáticamente N variables en la expresión y genera 
    una función numérica que acepta N argumentos f(*p).
    """
    try:
        expr = sp.sympify(expr_str, locals=FUNCIONES_PERMITIDAS)
    except (sp.SympifyError, TypeError, ValueError) as e:
        raise ValueError(f"Expresión inválida: {e}")
    
    # Extraer variables ordenadas alfabéticamente
    vars_simbolos = sorted(list(expr.free_symbols), key=lambda s: s.name)
    if not vars_simbolos:
        raise ValueError("La expresión debe contener al menos una variable.")
    
    func_num = sp.lambdify(vars_simbolos, expr, modules='numpy')
    return expr, func_num, vars_simbolos

def calcular_derivada(expr, var, orden=1):
    derivada = sp.diff(expr, var, orden)
    func_num = sp.lambdify((var,), derivada, modules='numpy')
    return derivada, func_num

def calcular_derivada_parcial(expr, var, orden=1):
    derivada = sp.diff(expr, var, orden)
    func_num = sp.lambdify((x, y), derivada, modules='numpy')
    return derivada, func_num