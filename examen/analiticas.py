from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Iterable

import pandas as pd
import numpy as np
import re

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def pivot_hours_por_dia(hours_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte hours(placeID, hours, days) a columnas por día (Mon..Sun) con horarios.

    - 'days' puede contener múltiples días separados por ';' (p.ej. 'Mon;Tue;Wed;').
    - 'hours' puede venir con ';' al final (p.ej. '00:00-00:00;'), se normaliza.
    - Si un placeID tiene múltiples rangos para el mismo día, se concatenan con '|'.
    """
    df = hours_df.copy()

    df["hours"] = (
        df["hours"].astype(str)
          .str.strip()
          .str.replace(r";+$", "", regex=True)
    )
    df["days"] = df["days"].astype(str).str.strip()

    df["day"] = df["days"].str.split(";")
    df = df.explode("day")
    df["day"] = df["day"].astype(str).str.strip()
    df = df[df["day"].isin(DAY_ORDER)]

    agg = (
        df.groupby(["placeID", "day"])["hours"]
          .apply(lambda x: "|".join(sorted(set(x))))
          .reset_index()
    )

    wide = agg.pivot(index="placeID", columns="day", values="hours").reset_index()

    wide.columns.name = None
    wide = wide.rename(columns={d: f"hours_{d}" for d in DAY_ORDER if d in wide.columns})

    import numpy as np
    for d in DAY_ORDER:
        col = f"hours_{d}"
        if col not in wide.columns:
            wide[col] = np.nan

    return wide[["placeID"] + [f"hours_{d}" for d in DAY_ORDER]]


NA_TOKENS_DEFAULT = {"?", "", "NA", "N/A", "NULL", "null", "None", "none", "nan"}


def normalizar_texto_columnas(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    na_tokens: Iterable[str] = NA_TOKENS_DEFAULT,
    strip: bool = True,
    ) -> pd.DataFrame:
    """
    Normaliza columnas tipo texto/categoría:
    - strip espacios
    - reemplaza tokens tipo '?' por NaN

    Parameters
    ----------
    df
        DataFrame de entrada.
    cols
        Columnas a normalizar. Si None, aplica a object/category.
    na_tokens
        Tokens a considerar como faltantes.
    strip
        Si True, aplica str.strip().

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con columnas normalizadas.
    """
    out = df.copy()
    if cols is None:
        cols = [c for c in out.columns
                if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_categorical_dtype(out[c])]

    na_tokens = set(na_tokens)

    for c in cols:
        if c not in out.columns:
            continue
        s = out[c]

        # Mantén NaN reales
        s2 = s.astype("string")

        if strip:
            s2 = s2.str.strip()

        # Reemplaza tokens exactos por NaN
        s2 = s2.replace(list(na_tokens), pd.NA)

        out[c] = s2
    return out


def estandarizar_coordenadas(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    decimals: int = 6,
    validar_rango: bool = True,
    ) -> pd.DataFrame:
    """
    Redondea lat/lon a 'decimals' y opcionalmente invalida valores fuera de rango.

    Returns copia del DF.
    """
    out = df.copy()
    if lat_col in out.columns:
        out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce").round(decimals)
        if validar_rango:
            out.loc[(out[lat_col] < -90) | (out[lat_col] > 90), lat_col] = np.nan

    if lon_col in out.columns:
        out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce").round(decimals)
        if validar_rango:
            out.loc[(out[lon_col] < -180) | (out[lon_col] > 180), lon_col] = np.nan

    return out


def limpiar_users(users: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza específica de users:
    - normaliza texto y reemplaza '?' por NaN en variables categóricas clave
    - estandariza coordenadas (6 decimales) y valida rangos

    Returns copia.
    """
    out = users.copy()

    cols_cat = [
        "smoker", "drink_level", "dress_preference", "ambience", "transport",
        "marital_status", "hijos", "interest", "personality", "religion",
        "activity", "color", "budget",
    ]
    out = normalizar_texto_columnas(out, cols=cols_cat)

    out = estandarizar_coordenadas(out, lat_col="latitude", lon_col="longitude", decimals=6, validar_rango=True)
    return out


def limpiar_restaurants(rest: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza específica de restaurants:
    - Reemplaza '?' por NaN en address/zip/url/fax/city/state/country
    - Elimina columnas no informativas: fax (100% '?')
    - Estandariza city/state/country (minúsculas/strip) y corrige variantes frecuentes
    - Estandariza coordenadas

    Returns copia.
    """
    out = rest.copy()

    cols_txt = ["address", "zip", "url", "fax", "city", "state", "country", "name"]
    out = normalizar_texto_columnas(out, cols=cols_txt)

    if "url" in out.columns:
        out["url"] = out["url"].astype("string").str.strip()
        out.loc[out["url"].isin(["no", "No", "NO"]), "url"] = pd.NA

    for c in ["city", "state", "country"]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()
            out[c] = out[c].str.replace(r"\s+", " ", regex=True)

    if "country" in out.columns:
        out.loc[out["country"].str.lower() == "mexico", "country"] = "Mexico"

    if "state" in out.columns:
        st = out["state"].astype("string")
        st_low = st.str.lower()

        out.loc[st_low.isin(["slp", "s.l.p.", "s.l.p", "s l p", "s.l.p. "]), "state"] = "SLP"
        out.loc[st_low == "morelos", "state"] = "Morelos"
        out.loc[st_low == "tamaulipas", "state"] = "Tamaulipas"
        out.loc[st_low == "san luis potosi", "state"] = "San Luis Potosi"

    if "city" in out.columns:
        ct = out["city"].astype("string")
        ct_low = ct.str.lower()
        out.loc[ct_low == "san luis potosi", "city"] = "San Luis Potosi"
        out.loc[ct_low == "cuernavaca", "city"] = "Cuernavaca"
        out.loc[ct_low.str.replace(" ", "", regex=False) == "victoria", "city"] = "victoria"

    out = estandarizar_coordenadas(out, lat_col="latitude", lon_col="longitude", decimals=6, validar_rango=True)
    return out


def _colapsar_espacios_series(s: pd.Series) -> pd.Series:
    """Recorta y colapsa múltiples espacios en columnas string."""
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def _normalizar_listas_sep(
    s: pd.Series,
    sep_in: str = ",",
    sep_out: str = " | ",
    sort_unique: bool = True,
) -> pd.Series:
    """
    Normaliza columnas con listas en texto (p.ej. 'VISA, cash') a formato canónico.
    """
    def norm_one(x):
        if pd.isna(x):
            return pd.NA
        parts = [p.strip() for p in str(x).split(sep_in)]
        parts = [p for p in parts if p != ""]
        if not parts:
            return pd.NA
        if sort_unique:
            parts = sorted(set(parts))
        return sep_out.join(parts)

    return s.apply(norm_one).astype("string")


def limpieza_final_base(base: pd.DataFrame) -> pd.DataFrame:
    """
    Estandarización final de `base` (después de unir):
    - Normaliza columnas tipo lista (agregados) a separador ' | ' y ordena únicos
    - Convierte booleans en texto si quedaron como 'True'/'False'
    - Colapsa espacios en campos de texto comunes

    No elimina filas.
    """
    out = base.copy()

    for c in ["user_cuisines", "user_payments", "rest_cuisines", "rest_payments", "rest_parking_types"]:
        if c in out.columns:
            out[c] = _normalizar_listas_sep(out[c], sep_in=",", sep_out=" | ", sort_unique=True)

    if "smoker" in out.columns and (pd.api.types.is_object_dtype(out["smoker"]) or pd.api.types.is_string_dtype(out["smoker"])):
        s = out["smoker"].astype("string").str.strip()
        out["smoker"] = s.map({"true": True, "false": False, "True": True, "False": False}).astype("boolean")

    if "franchise" in out.columns and (pd.api.types.is_object_dtype(out["franchise"]) or pd.api.types.is_string_dtype(out["franchise"])):
        f = out["franchise"].astype("string").str.strip().str.lower()
        out["franchise"] = f.map({"t": True, "f": False}).astype("boolean")

    for c in ["name", "address", "url", "city", "state", "country"]:
        if c in out.columns:
            out[c] = _colapsar_espacios_series(out[c])

    return out


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
    tmp = df.dropna(subset=[val_col]).copy()
    tmp[val_col] = tmp[val_col].astype("string").str.strip()
    tmp = tmp[tmp[val_col].notna()]

    return (
        tmp.groupby(group_col)[val_col]
           .apply(lambda x: sep.join(sorted(set(x.astype(str)))))
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
    users = limpiar_users(users)
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
    restaurants = limpiar_restaurants(restaurants)
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

    hours_pivot = pivot_hours_por_dia(hr)  # hr ya filtrado a valid_place
    dim_place = dim_place.merge(hours_pivot, on="placeID", how="left")

    return dim_place


def construir_base_modelado(
    dfs_csv: dict[str, pd.DataFrame],
    dim_user: pd.DataFrame,
    dim_place: pd.DataFrame,
    ratings_key: str = "ratings_csv",
    clean: bool = True,
    ) -> pd.DataFrame:
    """
    ... (tu docstring igual, añade explicación de clean)
    """
    ratings = dfs_csv[ratings_key].copy()

    base = ratings.merge(dim_user, on="userID", how="left")
    base = base.merge(dim_place, on="placeID", how="left", suffixes=("", "_rest"))

    if clean:
        base = limpieza_final_base(base)

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
