# Métodos Numéricos de Optimización

Implementación de métodos numéricos para **búsqueda de raíces** y **optimización**, desarrollada originalmente en un **Jupyter Notebook** y complementada con una **interfaz web en Streamlit**.

 <div align="center">
     
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![SymPy](https://img.shields.io/badge/SymPy-3B5526?style=flat)
 
</div>

---

## Estructura del Proyecto

```text
/
├── notebooks/
│   └── optimizacion.ipynb   # Implementación y pruebas de los métodos
├── metodos/                 # Funciones de los métodos numéricos
├── utils/                   # Utilidades (graficación, sympy)
├── app.py                   # Interfaz Streamlit
└── requirements.txt
```

---

| # | Método |
|---|--------|
| 1 | Bisección |
| 2 | Falsa Posición |
| 3 | Razón Dorada |
| 4 | Interpolación |
| 5 | Newton |
| 6 | Newton-Raphson |
| 7 | Búsqueda Aleatoria (N variables) |

---

## Ejecución Local

```bash
git clone https://github.com/CamilaTorres1035/numerical-methods-toolkit.git
cd numerical-methods-toolkit
 
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
venv\Scripts\Activate.ps1      # Windows
 
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy en Streamlit Community Cloud

[Métodos de Optimización](https://camss-numerical-methods-toolkit.streamlit.app/)

---

## Autor

<div align="center">

**Maria Camila Torres Chica**

[![GitHub](https://img.shields.io/badge/GitHub-CamilaTorres1035-181717?style=plastic&logo=github&logoColor=white)](https://github.com/CamilaTorres1035)

</div>
