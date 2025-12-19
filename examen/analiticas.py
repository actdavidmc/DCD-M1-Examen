from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

NA_TOKENS_DEFAULT = {"?", "", "NA", "N/A", "NULL", "null", "None", "none", "nan"}


def _to_key(x: object) -> str:
    """
    Convierte un valor a una llave comparable:
    - lower
    - elimina espacios y puntuación (solo deja [a-z0-9])

    Ej: "S.L.P." -> "slp", "San Luis Potosi" -> "sanluispotosi"
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    # deja solo alfanumérico
    return "".join(ch for ch in s if ch.isalnum())


def _colapsar_espacios(s: pd.Series) -> pd.Series:
    """strip + colapsa múltiples espacios internos."""
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def normalizar_texto_columnas(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    na_tokens: Iterable[str] = NA_TOKENS_DEFAULT,
    strip: bool = True,
    ) -> pd.DataFrame:
    """
    Normaliza columnas tipo texto/categoría:
    - opcional strip de espacios
    - reemplaza tokens tipo '?' por NaN (match exacto)

    Parameters
    ----------
    df
        DataFrame de entrada.
    cols
        Columnas a normalizar. Si None, aplica a object/category.
    na_tokens
        Tokens a considerar como faltantes (match exacto).
    strip
        Si True, aplica str.strip().

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con columnas normalizadas.
    """
    out = df.copy()
    if cols is None:
        cols = [
            c for c in out.columns
            if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_categorical_dtype(out[c])
        ]

    na_tokens = set(na_tokens)

    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype("string")
        if strip:
            s = s.str.strip()
        s = s.replace(list(na_tokens), pd.NA)
        out[c] = s

    return out


def estandarizar_coordenadas(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    decimals: int = 6,
    validar_rango: bool = True,
    ) -> pd.DataFrame:
    """
    Redondea lat/lon a 'decimals' y opcionalmente invalida valores fuera de rango (set NaN).

    No elimina filas.

    Returns
    -------
    pd.DataFrame
        Copia del DF con coordenadas estandarizadas.
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


def estandarizar_ubicacion(
    df: pd.DataFrame,
    city_col: str = "city",
    state_col: str = "state",
    country_col: str = "country",
    canon_city_slp: str = "San Luis Potosi",
    canon_state_slp: str = "San Luis Potosi",
    canon_country_mex: str = "Mexico",
    ) -> pd.DataFrame:
    """
    Estandariza city/state/country para reducir variantes:
    - city: "slp", "s.l.p." o "san luis potos" -> San Luis Potosi
    - state: "san luis potosi", "s.l.p.", "slp" -> SLP
    - country: "mexico" -> Mexico

    No elimina columnas ni filas.

    Returns
    -------
    pd.DataFrame
        Copia del DF con ubicación estandarizada.
    """
    out = df.copy()

    if city_col in out.columns:
        city = _colapsar_espacios(out[city_col])
        city_key = city.apply(_to_key)

        slp_city_keys = {"slp", "sanluispotosi", "sanluispotos", "sanluispoto", "sanluispotosi "}
        city = city.mask(city_key.isin(slp_city_keys), canon_city_slp)

        city = city.mask(city_key == "cuernavaca", "Cuernavaca")

        victoria_keys = {"victoria", "ciudadvictoria", "cdvictoria", "cdvictoria"}
        city = city.mask(city_key.isin(victoria_keys), "Ciudad Victoria")

        out[city_col] = city

    if state_col in out.columns:
        state = _colapsar_espacios(out[state_col])
        state_key = state.apply(_to_key)

        slp_state_keys = {
            "slp", "slp.", "slp..", "slp ", "slp-", "slp_", "slp0",
            "sanluispotosi", "sanluispotos", "sanluispoto",
            "slp", "slp", "slp",
        }

        state = state.mask(state_key.isin(slp_state_keys) | (state_key == "slp"), canon_state_slp)

        state = state.mask(state_key == "morelos", "Morelos")
        state = state.mask(state_key == "tamaulipas", "Tamaulipas")
        state = state.mask(state_key == "mexico", "Mexico")

        out[state_col] = state

    if country_col in out.columns:
        country = _colapsar_espacios(out[country_col])
        country_key = country.apply(_to_key)
        country = country.mask(country_key == "mexico", canon_country_mex)
        out[country_col] = country

    return out


def pivot_hours_por_dia(hours_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte hours(placeID, hours, days) a columnas por día (Mon..Sun) con horarios.

    - 'days' puede contener múltiples días separados por ';' (p.ej. 'Mon;Tue;Wed;').
    - 'hours' puede venir con ';' al final (p.ej. '00:00-00:00;'), se normaliza.
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

    for d in DAY_ORDER:
        col = f"hours_{d}"
        if col not in wide.columns:
            wide[col] = np.nan

    return wide[["placeID"] + [f"hours_{d}" for d in DAY_ORDER]]


def normalizar_bool_smoker(s: pd.Series) -> pd.Series:
    """
    Convierte smoker a dtype boolean:
    - 'true'/'false' (cualquier case) -> True/False
    - '?'/'NA'/''/None -> <NA>
    """
    x = s.astype("string").str.strip().str.lower()
    x = x.replace(list(NA_TOKENS_DEFAULT), pd.NA)
    out = x.map({"true": True, "false": False}).astype("boolean")
    return out


def normalizar_bool_franchise(s: pd.Series) -> pd.Series:
    """
    Convierte franchise a dtype boolean:
    - 't'/'f' (cualquier case) -> True/False
    - '?'/'NA'/''/None -> <NA>
    """
    x = s.astype("string").str.strip().str.lower()
    x = x.replace(list(NA_TOKENS_DEFAULT), pd.NA)
    out = x.map({"t": True, "f": False}).astype("boolean")
    return out


def limpiar_users(users: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza específica de users:
    - normaliza texto y reemplaza '?' por NaN en variables categóricas clave
    - estandariza coordenadas (6 decimales) y valida rangos

    No elimina columnas ni filas.
    """
    out = users.copy()

    cols_cat = [
        "smoker", "drink_level", "dress_preference", "ambience", "transport",
        "marital_status", "hijos", "interest", "personality", "religion",
        "activity", "color", "budget",
    ]
    out = normalizar_texto_columnas(out, cols=cols_cat)
    out = estandarizar_coordenadas(out, lat_col="latitude", lon_col="longitude", decimals=6, validar_rango=True)

    if "smoker" in out.columns:
        out["smoker"] = normalizar_bool_smoker(out["smoker"])

    return out


def limpiar_restaurants(rest: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza específica de restaurants:
    - Reemplaza '?' por NaN en address/zip/url/fax/city/state/country/name
    - Normaliza espacios
    - Estandariza city/state/country para reducir variantes
    - Estandariza coordenadas

    No elimina columnas ni filas.
    """
    out = rest.copy()

    cols_txt = ["address", "zip", "url", "fax", "city", "state", "country", "name"]
    out = normalizar_texto_columnas(out, cols=cols_txt)

    if "url" in out.columns:
        out["url"] = out["url"].astype("string").str.strip()
        out.loc[out["url"].isin(["no", "No", "NO"]), "url"] = pd.NA

    for c in ["address", "zip", "url", "fax", "city", "state", "country", "name"]:
        if c in out.columns:
            out[c] = _colapsar_espacios(out[c])

    out = estandarizar_ubicacion(out, city_col="city", state_col="state", country_col="country")

    out = estandarizar_coordenadas(out, lat_col="latitude", lon_col="longitude", decimals=6, validar_rango=True)

    if "franchise" in out.columns:
        out["franchise"] = normalizar_bool_franchise(out["franchise"])

    return out


def _agg_list(df: pd.DataFrame, group_col: str, val_col: str, sep: str = ", ") -> pd.Series:
    """
    Agrega valores categóricos a nivel de llave en forma de lista única unida por separador.
    - Normaliza texto (strip) y reemplaza tokens NA antes de agregar.
    - No modifica el DF original.

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
    tmp = normalizar_texto_columnas(tmp, cols=[val_col])
    tmp = tmp.dropna(subset=[val_col])

    return (
        tmp.groupby(group_col)[val_col]
           .apply(lambda x: sep.join(sorted(set(x.astype(str)))))
    )


def _normalizar_listas_post(s: pd.Series, sep_in: str = ",", sep_out: str = " | ") -> pd.Series:
    """
    Normaliza columnas de listas en texto tras agregación: separa por sep_in, strip, dedup, sort, une por sep_out.
    """
    def norm_one(x):
        if pd.isna(x):
            return pd.NA
        parts = [p.strip() for p in str(x).split(sep_in)]
        parts = [p for p in parts if p != ""]
        if not parts:
            return pd.NA
        parts = sorted(set(parts))
        return sep_out.join(parts)

    return s.apply(norm_one).astype("string")


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

    Nota: se filtra al universo de userID presentes en ratings.

    Returns
    -------
    pd.DataFrame
        Tabla analítica de usuarios (usuarios.csv) a nivel userID.
    """
    ratings = dfs_csv[ratings_key].copy()
    users = limpiar_users(dfs_csv[users_key].copy())
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

    for c in ["user_cuisines", "user_payments"]:
        if c in dim_user.columns:
            dim_user[c] = _normalizar_listas_post(dim_user[c], sep_in=",", sep_out=" | ")

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

    Nota: se filtra al universo de placeID presentes en ratings.

    Returns
    -------
    pd.DataFrame
        Tabla analítica de restaurantes (restaurantes.csv) a nivel placeID.
    """
    ratings = dfs_csv[ratings_key].copy()
    restaurants = limpiar_restaurants(dfs_csv[restaurants_key].copy())
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

    hours_pivot = pivot_hours_por_dia(hr)
    dim_place = dim_place.merge(hours_pivot, on="placeID", how="left")

    for c in ["rest_cuisines", "rest_payments", "rest_parking_types"]:
        if c in dim_place.columns:
            dim_place[c] = _normalizar_listas_post(dim_place[c], sep_in=",", sep_out=" | ")

    dim_place = estandarizar_ubicacion(dim_place, city_col="city", state_col="state", country_col="country")

    return dim_place


def limpieza_final_base(base: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza/estandarización final posterior a uniones:
    - normaliza listas a ' | '
    - colapsa espacios
    - estandariza city/state/country nuevamente por seguridad
    - convierte franchise t/f y smoker True/False si quedaron como string (sin borrar columnas)

    No elimina filas.
    """
    out = base.copy()

    for c in ["user_cuisines", "user_payments", "rest_cuisines", "rest_payments", "rest_parking_types"]:
        if c in out.columns:
            out[c] = _normalizar_listas_post(out[c], sep_in=",", sep_out=" | ")

    for c in ["name", "address", "url", "city", "state", "country"]:
        if c in out.columns:
            out[c] = _colapsar_espacios(out[c])

    out = estandarizar_ubicacion(out, city_col="city", state_col="state", country_col="country")

    if "smoker" in out.columns and (pd.api.types.is_object_dtype(out["smoker"]) or pd.api.types.is_string_dtype(out["smoker"])):
        s = out["smoker"].astype("string").str.strip().str.lower()
        out["smoker"] = s.map({"true": True, "false": False}).astype("boolean")

    if "franchise" in out.columns and (pd.api.types.is_object_dtype(out["franchise"]) or pd.api.types.is_string_dtype(out["franchise"])):
        f = out["franchise"].astype("string").str.strip().str.lower()
        out["franchise"] = f.map({"t": True, "f": False}).astype("boolean")

    return out


def construir_base_modelado(
    dfs_csv: dict[str, pd.DataFrame],
    dim_user: pd.DataFrame,
    dim_place: pd.DataFrame,
    ratings_key: str = "ratings_csv",
    clean: bool = True,
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
    clean
        Si True, aplica estandarización final posterior a joins (sin eliminar columnas/filas).

    Returns
    -------
    pd.DataFrame
        Base final para análisis/modelado.
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
        Nombres de archivo.

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


def agregar_features_ratings(
    df: pd.DataFrame,
    user_col: str = "userID",
    place_col: str = "placeID",
    rating_col: str = "rating",
    food_col: str = "food_rating",
    service_col: str = "service_rating",
) -> pd.DataFrame:
    """
    Agrega variables derivadas al grano de la unidad muestral ratings (1 fila = userID-placeID).

    Features creadas:
    1) subrating_mean: mean(food_rating, service_rating)
    2) subrating_gap: food_rating - service_rating
    3) subrating_gap_abs: |food_rating - service_rating|
    4) user_mean_rating_loo: promedio de rating por userID excluyendo la fila actual (leave-one-out)
    5) place_mean_rating_loo: promedio de rating por placeID excluyendo la fila actual (leave-one-out)

    Parameters
    ----------
    df
        DataFrame a nivel evaluación (ratings ya unido o ratings puro).
    user_col, place_col
        Columnas llave.
    rating_col, food_col, service_col
        Columnas de calificación.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con las nuevas columnas agregadas.
    """
    out = df.copy()

    # 1-3: derivadas de subratings (no requieren groupby)
    food = pd.to_numeric(out[food_col], errors="coerce")
    serv = pd.to_numeric(out[service_col], errors="coerce")

    out["subrating_mean"] = (food + serv) / 2.0
    out["subrating_gap"] = food - serv
    out["subrating_gap_abs"] = (food - serv).abs()

    # 4-5: leave-one-out para rating
    r = pd.to_numeric(out[rating_col], errors="coerce")

    # user LOO
    user_sum = r.groupby(out[user_col]).transform("sum")
    user_cnt = r.groupby(out[user_col]).transform("count")
    out["user_mean_rating_loo"] = np.where(
        user_cnt > 1,
        (user_sum - r) / (user_cnt - 1),
        np.nan,
    )

    # place LOO
    place_sum = r.groupby(out[place_col]).transform("sum")
    place_cnt = r.groupby(out[place_col]).transform("count")
    out["place_mean_rating_loo"] = np.where(
        place_cnt > 1,
        (place_sum - r) / (place_cnt - 1),
        np.nan,
    )

    return out
