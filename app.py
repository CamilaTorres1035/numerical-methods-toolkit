import streamlit as st
import numpy as np
import math
import sys
import os

# Agregar el directorio raíz al path para importar los módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metodos import biseccion, falsa_posicion, razon_dorada, interpolacion, newton_raphson, newton_optimizacion, busqueda_aleatoria
from utils.graficar import graficar_1d, graficar_busqueda_aleatoria


