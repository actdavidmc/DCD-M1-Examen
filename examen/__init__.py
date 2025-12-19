"""
Paquete `examen`

Módulos:
- cargar_datos: carga de tablas desde ZIP en /data
- validar_datos: comparación CSV vs XLSX
- perfil_datos: perfilado y reporte relacional
- analiticas: construcción de dimensiones y base de modelado
- modelado: limpieza, PCA, clustering variables, SelectKBest, WoE/IV
"""

from . import cargar_datos, validar_datos, perfil_datos, analiticas, modelado

__all__ = [
    "cargar_datos",
    "validar_datos",
    "perfil_datos",
    "analiticas",
    "modelado",
]

__version__ = "0.1.0"