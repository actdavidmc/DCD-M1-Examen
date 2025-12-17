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
