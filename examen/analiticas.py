from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def pivot_hours_por_dia(hours_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte hours(placeID, hours, days) a columnas por día (Mon..Sun) con horarios.

    - 'days' puede contener múltiples días separados por ';' (p.ej. 'Mon;Tue;Wed;').
    - Si un placeID tiene múltiples rangos para el mismo día, se concatenan con '|'.

    Parameters
    ----------
    hours_df
        DataFrame con columnas ['placeID','hours','days'].

    Returns
    -------
    pd.DataFrame
        DataFrame a nivel placeID con columnas:
        ['placeID','hours_Mon','hours_Tue',...,'hours_Sun']
    """
    df = hours_df.copy()

    # Normaliza strings
    df["hours"] = df["hours"].astype(str).str.strip()
    df["days"] = df["days"].astype(str).str.strip()

    # Explode de días: "Mon;Tue;Wed;" -> ["Mon","Tue","Wed"]
    df["day"] = df["days"].str.split(";")
    df = df.explode("day")
    df["day"] = df["day"].astype(str).str.strip()
    df = df[df["day"].isin(DAY_ORDER)]

    # Agrega por placeID y day
    agg = (
        df.groupby(["placeID", "day"])["hours"]
          .apply(lambda x: "|".join(sorted(set(x))))
          .reset_index()
    )

    # Pivot
    wide = agg.pivot(index="placeID", columns="day", values="hours").reset_index()

    # Renombra columnas
    wide.columns.name = None
    wide = wide.rename(columns={d: f"hours_{d}" for d in DAY_ORDER if d in wide.columns})

    # Asegura que existan todas las columnas
    for d in DAY_ORDER:
        col = f"hours_{d}"
        if col not in wide.columns:
            wide[col] = np.nan

    # Orden final
    return wide[["placeID"] + [f"hours_{d}" for d in DAY_ORDER]]


def _agg_list(df: pd.DataFrame, group_col: str, val_col: str, sep: str = ", ") -> pd.Series:
    """
    Agrega valores categóricos a nivel de llave en forma de lista única unida por separador.

    Parameters
    ----------
    df
        DataFrame fuente.
    group_col
        Columna llave para agrupar (p. ej., 'userID' o 'placeID').
    val_col
        Columna de valores a agregar (p. ej., 'Rcuisine', 'Upayment').
    sep
        Separador para concatenar valores únicos.

    Returns
    -------
    pd.Series
        Serie indexada por group_col con string agregado.
    """
    return (
        df.dropna(subset=[val_col])
          .groupby(group_col)[val_col]
          .apply(lambda x: sep.join(sorted(set(map(str, x)))))
    )


def construir_dim_usuario(
    dfs_csv: dict[str, pd.DataFrame],
    ratings_key: str = "ratings_csv",
    users_key: str = "users_csv",
    usercuisine_key: str = "usercuisine_csv",
    userpayment_key: str = "userpayment_csv",
    ref_year: int = 2025,
    ) -> pd.DataFrame:
    """
    Construye tabla analítica de usuarios (1 fila por userID).

    Integra:
    - atributos base del usuario (users)
    - agregados por userID desde usercuisine y userpayment
    - edad derivada (ref_year - birth_year)

    Nota: se filtra al universo de userID presentes en ratings para mantener coherencia del problema.

    Parameters
    ----------
    dfs_csv
        Diccionario con DataFrames CSV.
    ratings_key, users_key, usercuisine_key, userpayment_key
        Llaves dentro de dfs_csv.
    ref_year
        Año de referencia para calcular edad.

    Returns
    -------
    pd.DataFrame
        Tabla analítica de usuarios (usuarios.csv) a nivel userID.
    """
    ratings = dfs_csv[ratings_key].copy()
    users = dfs_csv[users_key].copy()
    usercuisine = dfs_csv[usercuisine_key].copy()
    userpayment = dfs_csv[userpayment_key].copy()

    valid_user = set(ratings["userID"].unique())

    dim_user = users[users["userID"].isin(valid_user)].copy()

    uc = usercuisine[usercuisine["userID"].isin(valid_user)].copy()
    up = userpayment[userpayment["userID"].isin(valid_user)].copy()

    dim_user = dim_user.merge(_agg_list(uc, "userID", "Rcuisine").rename("user_cuisines"),
                              on="userID", how="left")
    dim_user = dim_user.merge(uc.groupby("userID")["Rcuisine"].nunique().rename("n_user_cuisines"),
                              on="userID", how="left")

    dim_user = dim_user.merge(_agg_list(up, "userID", "Upayment").rename("user_payments"),
                              on="userID", how="left")
    dim_user = dim_user.merge(up.groupby("userID")["Upayment"].nunique().rename("n_user_payments"),
                              on="userID", how="left")

    dim_user["age_ref"] = ref_year - pd.to_numeric(dim_user["birth_year"], errors="coerce")

    return dim_user


def construir_dim_restaurante(
    dfs_csv: dict[str, pd.DataFrame],
    ratings_key: str = "ratings_csv",
    restaurants_key: str = "restaurants_csv",
    cuisine_key: str = "cuisine_csv",
    parking_key: str = "parking_csv",
    hours_key: str = "hours_csv",
    payment_methods_key: str = "payment_methods_csv",
    ) -> pd.DataFrame:
    """
    Construye tabla analítica de restaurantes (1 fila por placeID).

    Integra:
    - atributos base del restaurante (restaurants)
    - agregados por placeID desde cuisine, parking, hours, payment_methods

    Nota: se filtra al universo de placeID presentes en ratings para mantener coherencia del problema.

    Parameters
    ----------
    dfs_csv
        Diccionario con DataFrames CSV.
    ratings_key, restaurants_key, cuisine_key, parking_key, hours_key, payment_methods_key
        Llaves dentro de dfs_csv.

    Returns
    -------
    pd.DataFrame
        Tabla analítica de restaurantes (restaurantes.csv) a nivel placeID.
    """
    ratings = dfs_csv[ratings_key].copy()
    restaurants = dfs_csv[restaurants_key].copy()
    cuisine = dfs_csv[cuisine_key].copy()
    parking = dfs_csv[parking_key].copy()
    hours = dfs_csv[hours_key].copy()
    payment_methods = dfs_csv[payment_methods_key].copy()

    valid_place = set(ratings["placeID"].unique())

    dim_place = restaurants[restaurants["placeID"].isin(valid_place)].copy()

    cu = cuisine[cuisine["placeID"].isin(valid_place)].copy()
    pk = parking[parking["placeID"].isin(valid_place)].copy()
    hr = hours[hours["placeID"].isin(valid_place)].copy()
    pm = payment_methods[payment_methods["placeID"].isin(valid_place)].copy()

    dim_place = dim_place.merge(_agg_list(cu, "placeID", "Rcuisine").rename("rest_cuisines"),
                                on="placeID", how="left")
    dim_place = dim_place.merge(cu.groupby("placeID")["Rcuisine"].nunique().rename("n_rest_cuisines"),
                                on="placeID", how="left")

    dim_place = dim_place.merge(_agg_list(pm, "placeID", "Rpayment").rename("rest_payments"),
                                on="placeID", how="left")
    dim_place = dim_place.merge(pm.groupby("placeID")["Rpayment"].nunique().rename("n_rest_payments"),
                                on="placeID", how="left")

    dim_place = dim_place.merge(_agg_list(pk, "placeID", "parking_lot").rename("rest_parking_types"),
                                on="placeID", how="left")
    dim_place = dim_place.merge(pk.groupby("placeID")["parking_lot"].nunique().rename("n_rest_parking_types"),
                                on="placeID", how="left")

    dim_place = dim_place.merge(hr.groupby("placeID").size().rename("n_hours_rows"),
                                on="placeID", how="left")
    dim_place = dim_place.merge(hr.groupby("placeID")["days"].nunique().rename("n_days_unique"),
                                on="placeID", how="left")

    return dim_place


def construir_base_modelado(
    dfs_csv: dict[str, pd.DataFrame],
    dim_user: pd.DataFrame,
    dim_place: pd.DataFrame,
    ratings_key: str = "ratings_csv",
    ) -> pd.DataFrame:
    """
    Construye la base de modelado a unidad muestral de evaluación (ratings).

    Realiza LEFT JOIN desde ratings hacia dim_user y dim_place, preservando:
    - número de filas de ratings (no debe explotar)
    - grano: 1 fila = 1 evaluación userID-placeID

    Parameters
    ----------
    dfs_csv
        Diccionario con DataFrames CSV.
    dim_user
        Tabla analítica de usuarios (1 fila por userID).
    dim_place
        Tabla analítica de restaurantes (1 fila por placeID).
    ratings_key
        Llave de ratings en dfs_csv.

    Returns
    -------
    pd.DataFrame
        Base final para análisis/modelado.
    """
    ratings = dfs_csv[ratings_key].copy()

    base = ratings.merge(dim_user, on="userID", how="left")
    base = base.merge(dim_place, on="placeID", how="left", suffixes=("", "_rest"))

    return base


def exportar_tablas_analiticas(
    dim_user: pd.DataFrame,
    dim_place: pd.DataFrame,
    out_dir: str | Path = ".",
    usuarios_name: str = "usuarios.csv",
    restaurantes_name: str = "restaurantes.csv",
    ) -> tuple[Path, Path]:
    """
    Exporta tablas analíticas requeridas por el examen: usuarios.csv y restaurantes.csv.

    Parameters
    ----------
    dim_user
        Tabla analítica de usuarios.
    dim_place
        Tabla analítica de restaurantes.
    out_dir
        Carpeta de salida.
    usuarios_name, restaurantes_name
        Nombres de archivo requeridos.

    Returns
    -------
    tuple[Path, Path]
        Rutas de los CSV exportados (usuarios, restaurantes).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_users = out_dir / usuarios_name
    p_rest = out_dir / restaurantes_name

    dim_user.to_csv(p_users, index=False)
    dim_place.to_csv(p_rest, index=False)

    return p_users, p_rest
