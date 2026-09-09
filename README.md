# Métodos Numéricos de Optimización

Implementación de métodos numéricos para búsqueda de raíces y optimización, desarrollada originalmente en un **Jupyter Notebook** y complementada con una **interfaz web en Streamlit**.

--

## Estructura del Proyecto

```text
    /
    ├── notebooks/
    │ └── optimizacion.ipynb # Implementación y pruebas de los métodos
    ├── metodos/ # Funciones de los métodos numéricos
    ├── utils/ # Utilidades (graficación, sympy)
    ├── app.py # Interfaz Streamlit
    └── requirements.txt
```

## Métodos Implementados

- Bisección
- Falsa Posición
- Razón Dorada
- Interpolación
- Newton
- Newton-Raphson
- Búsqueda Aleatoria (N variables)

## Ejecución Local

```bash
    git clone https://github.com/CamilaTorres1035/numerical-methods-toolkit.git
    cd numerical-methods-toolkit
    python -m venv .venv
    source .venv/bin/activate # Linux/Mac
    venv\Scripts\Activate.ps1 # Windows
    pip install -r requirements.txt
    streamlit run app.py
```

## Deploy en Streamlit Community Cloud

## Tecnologías

Python · Streamlit · NumPy · Pandas · Matplotlib · SymPy

---

## Autor

<div align="center">

**Maria Camila Torres Chica**

[![GitHub](https://img.shields.io/badge/GitHub-CamilaTorres1035-181717?style=plastic&logo=github&logoColor=white)](https://github.com/CamilaTorres1035)

</div>
