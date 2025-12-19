from __future__ import annotations
from typing import Optional
from pathlib import Path, PurePosixPath

import zipfile
import pandas as pd
import io


def encontrar_raiz_del_proyecto() -> Path:
    """
    Encuentra la raíz del repositorio subiendo por los directorios padre desde este archivo.

    La función busca marcadores típicos de un proyecto Python con control de versiones:
    - Una carpeta `.git/`, o
    - Un archivo `requirements.txt`.

    Returns
    -------
    Path
        Ruta absoluta a la raíz del proyecto.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra ningún marcador en la jerarquía de directorios padre.
    """
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists() or (p / "requirements.txt").exists():
            return p
    raise FileNotFoundError("No se encontró la raíz del proyecto")


def cargar_dfs_desde_zip_en_data() -> dict[str, pd.DataFrame]:
    """
    Carga en memoria todos los archivos CSV y XLSX contenidos en el único archivo .zip
    ubicado dentro de la carpeta `data/` del proyecto.

    La función:
    1) Detecta la raíz del proyecto con `encontrar_raiz_del_proyecto()`.
    2) Busca archivos `.zip` en `<raíz>/data/` y exige que exista exactamente uno.
    3) Abre el zip e itera sus miembros.
    4) Para cada miembro con extensión `.csv` o `.xlsx`, lo lee directamente desde el zip
       (sin extraer a disco) y lo almacena en un diccionario.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario donde:
        - La llave es un nombre normalizado tipo: `<archivo>_<ext>` (ej. `users_csv`, `restaurants_xlsx`).
        - El valor es el DataFrame cargado con pandas.

    Raises
    ------
    ValueError
        Si no existe exactamente 1 archivo `.zip` en la carpeta `data/`, o si el zip no
        contiene archivos `.csv` o `.xlsx`.
    FileNotFoundError
        Si no se encuentra la raíz del proyecto (propagado desde `encontrar_raiz_del_proyecto()`).
    zipfile.BadZipFile
        Si el archivo encontrado no es un zip válido o está corrupto.
    """
    root = encontrar_raiz_del_proyecto()
    data_dir = root / "data"

    zip_files = list(data_dir.glob("*.zip"))
    if len(zip_files) != 1:
        raise ValueError(
            f"Esperaba 1 zip en {data_dir}, encontré {len(zip_files)}: {[z.name for z in zip_files]}"
        )

    zip_path = zip_files[0]
    dfs: dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue

            low = name.lower()
            if not (low.endswith(".csv") or low.endswith(".xlsx")):
                continue

            # ---- construir key: quitar prefijo "datos/" y usar "<stem>_<ext>" ----
            p = PurePosixPath(name)  # paths internos del zip usan /
            stem = p.stem  # users
            ext = p.suffix.lstrip(".").lower()  # csv / xlsx
            key = f"{stem}_{ext}"  # users_csv, users_xlsx

            # evitar colisiones si existieran repetidos
            if key in dfs:
                i = 2
                new_key = f"{key}_{i}"
                while new_key in dfs:
                    i += 1
                    new_key = f"{key}_{i}"
                key = new_key
            # --------------------------------------------------------------------

            print(f"Leyendo {name} -> key={key}")

            if low.endswith(".csv"):
                with z.open(name) as f:
                    dfs[key] = pd.read_csv(f)

            elif low.endswith(".xlsx"):
                data = z.read(name)
                dfs[key] = pd.read_excel(io.BytesIO(data), engine="openpyxl")

    if not dfs:
        raise ValueError("No se encontraron .csv o .xlsx dentro del zip")

    return dfs


def obtener_csvs(
    dfs: dict[str, pd.DataFrame],
    ordenar: bool = True,
    orden: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Filtra el diccionario `dfs` y regresa solo los DataFrames provenientes de CSV.
    Opcionalmente los ordena en un orden canónico (útil para lectura/diagrama).

    Parameters
    ----------
    dfs
        Diccionario general devuelto por `cargar_dfs_desde_zip_en_data()`.
    ordenar
        Si True, reordena los DataFrames usando `orden` (si se provee) o un orden
        canónico por defecto.
    orden
        Lista de llaves en el orden deseado. Las llaves no incluidas se agregan al final
        preservando su orden original. Si es None, se usa el orden canónico por defecto.

    Returns
    -------
    dict[str, pd.DataFrame]
        Sub-diccionario con llaves que terminan en '_csv', y (si `ordenar=True`) ordenado.
    """
    dfs_csv = {k: v for k, v in dfs.items() if k.endswith("_csv")}

    if not ordenar:
        return dfs_csv

    if orden is None:
        orden = [
            "usercuisine_csv", "users_csv", "userpayment_csv",
            "ratings_csv",
            "parking_csv", "restaurants_csv", "cuisine_csv",
            "payment_methods_csv", "hours_csv",
        ]

    out = {k: dfs_csv[k] for k in orden if k in dfs_csv}
    for k in dfs_csv:
        if k not in out:
            out[k] = dfs_csv[k]
    return out
