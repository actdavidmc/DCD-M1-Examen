# Examen Módulo 1 — Ciencia de Datos (G33)

Este repositorio contiene la solución del **Examen G33**. El flujo implementa:

- Carga de datos desde un `.zip` (sin necesidad de extraer manualmente).
- Validación de consistencia **CSV vs XLSX**.
- Perfilado exploratorio de los CSV.
- Construcción del modelo relacional y definición de la **unidad muestral**.
- Ingeniería de variables a nivel de unidad muestral.
- Limpieza reproducible (nulos, variables unarias, correlación perfecta, outliers).
- Reducción/selección de variables: **PCA 2D**, **clustering de variables**, **SelectKBest**, **WoE/IV**.
- Exportación de entregables en `data/`.

## Estructura

```
.
├── Examen-G33.pdf
├── reporte.ipynb
├── examen/
│   ├── cargar_datos.py
│   ├── validar_datos.py
│   ├── perfil_datos.py
│   ├── analiticas.py
│   └── modelado.py
├── data/
│   ├── usuarios.csv
│   ├── restaurantes.csv
│   └── pca_2d.png
└── requirements.txt
```

## Requisitos e instalación

Crear y activar entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Ejecución

Abrir el notebook y correr todo:

```bash
jupyter lab
# o
jupyter notebook
```

Notebook principal:

- `reporte.ipynb`

## Entregables

Se generan / incluyen:

- `data/usuarios.csv`
- `data/restaurantes.csv`
- `data/pca_2d.png`
- `reporte.ipynb`
