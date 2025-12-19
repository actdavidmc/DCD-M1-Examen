from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Any

import pandas as pd


def tabla_perfil_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla de perfil por variable (sin modificar los datos), útil para EDA previo a limpieza.

    Columnas incluidas:
    - Variable
    - Tipo_inferido (Numerica / Categorica / Texto / Booleano / Fecha-Hora / Otro)
    - Dtype
    - Nulos, %Nulos
    - Unicos, %Cardinalidad (Unicos / N * 100)
    - Min, Max, Media (si aplica a numéricas)
    - Moda, Top_3 (si aplica a categóricas/texto)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a perfilar.

    Returns
    -------
    pd.DataFrame
        Tabla con un registro por columna.
    """
    n = len(df)
    rows = []

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)

        nulos = int(s.isna().sum())
        pct_nulos = (nulos / n * 100) if n else 0.0

        unicos = int(s.nunique(dropna=True))
        pct_card = (unicos / n * 100) if n else 0.0

        # inferencia simple
        if pd.api.types.is_bool_dtype(s):
            tipo = "Booleano"
        elif pd.api.types.is_datetime64_any_dtype(s):
            tipo = "Fecha-Hora"
        elif pd.api.types.is_numeric_dtype(s):
            tipo = "Numerica"
        elif pd.api.types.is_categorical_dtype(s):
            tipo = "Categorica"
        elif pd.api.types.is_object_dtype(s):
            # si son muchos únicos, lo tratamos como "Texto"
            tipo = "Texto" if (n and (unicos / n) > 0.2) else "Categorica"
        else:
            tipo = "Otro"

        # defaults
        min_v = max_v = media_v = None
        moda_v = None
        top3_v = None

        # numéricas: min/max/media
        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().any():
                min_v = float(s_num.min())
                max_v = float(s_num.max())
                media_v = float(s_num.mean())

        # categóricas/texto: moda + top 3
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s):
            vc = s.value_counts(dropna=True)
            if len(vc) > 0:
                moda_v = vc.index[0]
                top3 = vc.head(3)
                top3_v = ", ".join([f"{idx} ({int(cnt)})" for idx, cnt in top3.items()])

        rows.append({
            "Variable": col,
            "Tipo_inferido": tipo,
            "Dtype": dtype,
            "Nulos": nulos,
            "%Nulos": round(pct_nulos, 2),
            "Unicos": unicos,
            "%Cardinalidad": round(pct_card, 2),
            "Min": min_v,
            "Max": max_v,
            "Media": media_v,
            "Moda": moda_v,
            "Top_3": top3_v,
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(["%Nulos", "%Cardinalidad"], ascending=False).reset_index(drop=True)
    return out


def perfilar_todos_los_dfs(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Aplica `tabla_perfil_variables` a todos los DataFrames en `dfs`.

    Parameters
    ----------
    dfs
        Diccionario {nombre: DataFrame}.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {nombre: tabla_perfil} para cada dataset.
    """
    return {name: tabla_perfil_variables(df) for name, df in dfs.items()}


def perfilar_todos_largo(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Genera una tabla única (formato largo) con el perfil de todos los DataFrames,
    agregando una columna 'Dataset'.

    Parameters
    ----------
    dfs
        Diccionario {nombre: DataFrame}.

    Returns
    -------
    pd.DataFrame
        Tabla concatenada con columna extra 'Dataset'.
    """
    frames = []
    for name, df in dfs.items():
        t = tabla_perfil_variables(df).copy()
        t.insert(0, "Dataset", name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class KeyCardinality:
    """Resumen de cardinalidad para una llave dentro de un DataFrame."""
    rows: int
    unique_keys: int
    min_per_key: int
    max_per_key: int
    mean_per_key: float
    keys_in_universe: Optional[int] = None
    pct_keys_in_universe: Optional[float] = None


def detectar_llaves_candidatas(df: pd.DataFrame) -> list[str]:
    """
    Detecta llaves candidatas comunes para el examen (IDs).

    Args:
        df: DataFrame a inspeccionar.

    Returns:
        Lista de nombres de columnas que parecen llaves (p. ej., userID, placeID, id).
    """
    candidates = []
    for c in df.columns:
        cl = c.lower()
        if cl in {"userid", "placeid", "id"} or cl.endswith("id"):
            candidates.append(c)
    return candidates


def cardinalidad_por_llave(
    df: pd.DataFrame,
    key: str,
    universe: Optional[set[Any]] = None,
) -> KeyCardinality:
    """
    Calcula cardinalidad de registros por llave y (opcionalmente) cobertura contra un universo.

    Útil para:
    - identificar relaciones 1-a-1 vs 1-a-muchos (mean_per_key > 1 sugiere 1-a-muchos)
    - detectar si una tabla trae IDs fuera del universo (ratings)

    Args:
        df: DataFrame a analizar.
        key: Columna llave (p. ej., 'userID' o 'placeID').
        universe: Conjunto de llaves válidas (por ejemplo, set(ratings['userID'])).

    Returns:
        KeyCardinality con métricas de distribución por llave y cobertura si se provee universe.
    """
    if key not in df.columns:
        raise KeyError(f"La columna '{key}' no existe en el DataFrame.")

    s = df.groupby(key).size()
    rows = int(len(df))
    unique_keys = int(s.shape[0])
    min_per_key = int(s.min())
    max_per_key = int(s.max())
    mean_per_key = float(s.mean())

    keys_in_universe = None
    pct_keys_in_universe = None

    if universe is not None:
        keys = df[key].dropna().unique()
        in_universe = sum(k in universe for k in keys)
        keys_in_universe = int(in_universe)
        pct_keys_in_universe = float(in_universe / max(len(keys), 1))

    return KeyCardinality(
        rows=rows,
        unique_keys=unique_keys,
        min_per_key=min_per_key,
        max_per_key=max_per_key,
        mean_per_key=mean_per_key,
        keys_in_universe=keys_in_universe,
        pct_keys_in_universe=pct_keys_in_universe,
    )


def reporte_relacional(
    dfs: Dict[str, pd.DataFrame],
    ratings_key: str = "ratings_csv",
    keys: Iterable[str] = ("userID", "placeID"),
) -> pd.DataFrame:
    """
    Genera un reporte tabular de llaves, cardinalidad y cobertura (vs universo de ratings).

    Args:
        dfs: Diccionario {nombre: DataFrame}, típicamente el resultado de obtener_csvs().
        ratings_key: Clave del DataFrame de hechos (ratings) dentro de dfs.
        keys: Llaves a evaluar (por defecto userID y placeID).

    Returns:
        DataFrame con columnas:
        ['df','key','rows','unique_keys','min_per_key','max_per_key','mean_per_key',
         'keys_in_universe','pct_keys_in_universe']
    """
    if ratings_key not in dfs:
        raise KeyError(f"No se encontró '{ratings_key}' en dfs.")

    ratings = dfs[ratings_key]
    universes = {
        "userID": set(ratings["userID"].unique()) if "userID" in ratings.columns else set(),
        "placeID": set(ratings["placeID"].unique()) if "placeID" in ratings.columns else set(),
    }

    rows_out = []
    for name, df in dfs.items():
        for k in keys:
            if k in df.columns:
                st = cardinalidad_por_llave(df, k, universe=universes.get(k))
                rows_out.append({
                    "df": name,
                    "key": k,
                    "rows": st.rows,
                    "unique_keys": st.unique_keys,
                    "min_per_key": st.min_per_key,
                    "max_per_key": st.max_per_key,
                    "mean_per_key": st.mean_per_key,
                    "keys_in_universe": st.keys_in_universe,
                    "pct_keys_in_universe": st.pct_keys_in_universe,
                })

    return pd.DataFrame(rows_out).sort_values(["key", "df"]).reset_index(drop=True)

