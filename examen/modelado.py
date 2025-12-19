from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import FunctionTransformer

from sklearn.cluster import AgglomerativeClustering


def construir_objetivo(
    base: pd.DataFrame,
    tipo: str = "binaria",
    regla_binaria: str = "rating_eq_2",
    col_rating: str = "rating",
    out_col: str = "target",
    ) -> pd.DataFrame:
    """
    Crea la variable objetivo y la agrega como columna a la base.

    Parameters
    ----------
    base
        DataFrame base a unidad muestral (1 fila = 1 evaluación userID-placeID).
    tipo
        "binaria" o "multiclase".
    regla_binaria
        Regla para binaria:
        - "rating_eq_2": 1 si rating==2, si no 0
        - "rating_ge_1": 1 si rating>=1, si no 0
    col_rating
        Columna que contiene el rating global (esperado 0/1/2).
    out_col
        Nombre de la columna objetivo.

    Returns
    -------
    pd.DataFrame
        Copia de base con la columna objetivo agregada.
    """
    df = base.copy()
    r = pd.to_numeric(df[col_rating], errors="coerce")

    if tipo.lower() in {"multiclase", "multi", "categorica"}:
        df[out_col] = r.astype("Int64")
        return df

    # binaria
    if regla_binaria == "rating_ge_1":
        df[out_col] = (r >= 1).astype("Int64")
    else:
        # default: rating_eq_2
        df[out_col] = (r == 2).astype("Int64")

    return df


@dataclass
class LimpiezaConfig:
    """
    Configuración para limpieza.

    missing_threshold:
        Umbral para "variables poco pobladas" (p.ej. 0.65 = 65% nulos).
    drop_missing:
        Si True, elimina columnas con %nulos > missing_threshold.
    drop_unarias:
        Si True, elimina variables unarias (<=1 valor distinto incluyendo NA).
    drop_corr_abs_eq_1:
        Si True, elimina columnas numéricas que estén en pares con |corr|==1.
    outlier_cols:
        Columnas numéricas a evaluar para outliers (si None, usa una lista típica).
    outlier_strategy:
        "none" (solo evidencia), "winsorize" (clip p1/p99), o "iqr_clip" (clip por IQR).
    """
    missing_threshold: float = 0.65
    drop_missing: bool = True
    drop_unarias: bool = True
    drop_corr_abs_eq_1: bool = True
    outlier_cols: Optional[List[str]] = None
    outlier_strategy: str = "winsorize"


def reporte_limpieza(
    df: pd.DataFrame,
    missing_threshold: float = 0.65,
    ) -> Dict[str, pd.DataFrame]:
    """
    Genera evidencia para la sección de limpieza.

    Retorna un diccionario con tablas:
    - missing: %nulos y bandera (>threshold)
    - unarias: columnas con <=1 valor distinto (incluye NA)
    - corr_abs_eq_1: pares numéricos con |corr|==1 (si existen)

    Parameters
    ----------
    df
        Dataset (idealmente X sin target) a revisar.
    missing_threshold
        Umbral para "poco pobladas".

    Returns
    -------
    dict[str, pd.DataFrame]
        Tablas de evidencia.
    """
    out: Dict[str, pd.DataFrame] = {}

    miss = df.isna().mean().sort_values(ascending=False)
    out["missing"] = (
        pd.DataFrame({
            "col": miss.index,
            "pct_null": (miss.values * 100).round(2),
            "flag_poco_poblada": (miss.values > missing_threshold),
        })
        .sort_values("pct_null", ascending=False)
        .reset_index(drop=True)
    )

    nun = df.nunique(dropna=False).sort_values()
    out["unarias"] = (
        pd.DataFrame({"col": nun.index, "nunique_incl_na": nun.values})
        .query("nunique_incl_na <= 1")
        .reset_index(drop=True)
    )

    num = df.select_dtypes(include=["number"]).copy()
    pairs = []
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if pd.notna(v) and abs(float(v)) == 1.0:
                    pairs.append({"col_a": cols[i], "col_b": cols[j], "corr": float(v)})
    out["corr_abs_eq_1"] = pd.DataFrame(pairs).sort_values(["col_a", "col_b"]).reset_index(drop=True)

    return out


def _drop_corr_abs_eq_1(df: pd.DataFrame) -> List[str]:
    """
    Devuelve columnas a eliminar cuando existe al menos un par con |corr|==1.
    Mantiene col_a y propone eliminar col_b (convención reproducible).
    """
    rep = reporte_limpieza(df, missing_threshold=2.0)  # threshold irrelevante aquí
    if rep["corr_abs_eq_1"].empty:
        return []
    return rep["corr_abs_eq_1"]["col_b"].dropna().unique().tolist()


def _winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return x
    lo = x.quantile(p)
    hi = x.quantile(1 - p)
    return x.clip(lo, hi)


def _iqr_clip_series(s: pd.Series, k: float = 1.5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return x
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return x.clip(lo, hi)


def aplicar_limpieza(
    df: pd.DataFrame,
    config: LimpiezaConfig = LimpiezaConfig(),
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Aplica limpieza con trazabilidad.

    Procesos (según config):
    - variables poco pobladas: elimina si %nulos > threshold
    - variables unarias: elimina
    - correlación perfecta: elimina col_b en pares |corr|==1
    - outliers: opcional (winsorización o clip por IQR) en columnas seleccionadas

    Returns
    -------
    (df_limpio, reporte)
        df_limpio: copia del dataset tras limpieza
        reporte: dict con listas de columnas eliminadas y estrategia de outliers
    """
    X = df.copy()
    rep: Dict[str, object] = {"shape_before": X.shape}

    # 1) poco pobladas
    miss = X.isna().mean()
    drop_missing = miss[miss > config.missing_threshold].index.tolist()
    rep["drop_poco_pobladas"] = drop_missing
    if config.drop_missing and drop_missing:
        X = X.drop(columns=drop_missing, errors="ignore")

    nun = X.nunique(dropna=False)
    drop_unarias = nun[nun <= 1].index.tolist()
    rep["drop_unarias"] = drop_unarias
    if config.drop_unarias and drop_unarias:
        X = X.drop(columns=drop_unarias, errors="ignore")

    drop_corr = []
    if config.drop_corr_abs_eq_1:
        drop_corr = _drop_corr_abs_eq_1(X.select_dtypes(include=["number"]))
    rep["drop_corr_abs_eq_1"] = drop_corr
    if drop_corr:
        X = X.drop(columns=drop_corr, errors="ignore")

    if config.outlier_cols is None:
        config.outlier_cols = [c for c in ["age_ref", "weight", "height", "user_rest_distance_km"] if c in X.columns]

    rep["outlier_cols"] = config.outlier_cols
    rep["outlier_strategy"] = config.outlier_strategy

    if config.outlier_strategy == "winsorize":
        for c in config.outlier_cols:
            if c in X.columns:
                X[c] = _winsorize_series(X[c], p=0.01)

    elif config.outlier_strategy == "iqr_clip":
        for c in config.outlier_cols:
            if c in X.columns:
                X[c] = _iqr_clip_series(X[c], k=1.5)


    rep["shape_after"] = X.shape
    return X, rep


def _num_sanitize(X):
    """
    Sanitiza entrada para pipeline numérico:
    - Convierte cualquier cosa no numérica a NaN con to_numeric(coerce)
    - Convierte pd.NA (NAType) a np.nan
    - Regresa np.ndarray float
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X2 = X.copy()
        if isinstance(X2, pd.Series):
            X2 = X2.to_frame()
        X2 = X2.apply(pd.to_numeric, errors="coerce")
        return X2.to_numpy(dtype=float)

    # ndarray/list -> fuerza object y luego a numeric
    arr = np.asarray(X, dtype=object)
    # convierte elemento a elemento a float, lo no convertible -> np.nan
    out = np.empty(arr.shape, dtype=float)
    it = np.nditer(arr, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
    for v in it:
        val = v.item()
        try:
            if val is pd.NA:
                out[it.multi_index] = np.nan
            else:
                out[it.multi_index] = float(val)
        except Exception:
            out[it.multi_index] = np.nan
    return out


def _cat_sanitize(X):
    """
    Sanitiza entrada para pipeline categórico:
    - Convierte pd.NA (NAType) a np.nan
    - Fuerza a strings (object) sin perder np.nan
    - Regresa np.ndarray object
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X2 = X.copy()
        if isinstance(X2, pd.Series):
            X2 = X2.to_frame()

        # pasar todo a object, y toda NA a np.nan por máscara (no comparación)
        X2 = X2.astype(object)
        X2 = X2.where(pd.notna(X2), np.nan)

        # strings limpias
        for c in X2.columns:
            X2[c] = X2[c].astype(str)
            X2.loc[X2[c].isin(["nan", "None", "NA", "N/A", "NULL", "null", ""]), c] = np.nan

        return X2.to_numpy(dtype=object)

    arr = np.asarray(X, dtype=object)
    mask = np.frompyfunc(lambda v: (v is pd.NA) or (v is None), 1, 1)(arr).astype(bool)
    arr = arr.copy()
    arr[mask] = np.nan
    return arr.astype(object)


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocesador robusto (anti pd.NA/NAType) para sklearn.
    - Numéricas: _num_sanitize -> Imputer(median) -> StandardScaler
    - Categóricas: _cat_sanitize -> Imputer(most_frequent) -> OneHot
    """
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("san", FunctionTransformer(_num_sanitize, validate=False, feature_names_out="one-to-one")),
                ("imp", SimpleImputer(strategy="median", missing_values=np.nan)),
                ("sc", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("san", FunctionTransformer(_cat_sanitize, validate=False, feature_names_out="one-to-one")),
                ("imp", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return pre


def pca_2d_y_png(
    X: pd.DataFrame,
    out_png: str | Path = "pca_2d.png",
    random_state: int = 42,
    ) -> Dict[str, object]:
    """
    Ejecuta PCA a 2 dimensiones (sobre X preprocesado) y guarda un PNG con la nube.
    """
    import matplotlib.pyplot as plt

    # ✅ clave: pd.NA -> np.nan para sklearn
    X = X.copy().replace({pd.NA: np.nan})

    pre = construir_preprocesador(X)
    X_proc = pre.fit_transform(X)

    # ColumnTransformer puede devolver matriz dispersa; PCA requiere densa.
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X_proc)
    explained = float(np.sum(pca.explained_variance_ratio_))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], s=10)
    plt.title("PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return {"Z": Z, "explained_var": explained, "png_path": out_png}


def clustering_variables_representantes(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    distance_threshold: float = 0.30,
    random_state: int = 42,
    top_n: int = 25,
    ) -> pd.DataFrame:
    """
    Clustering jerárquico de variables en el espacio preprocesado (one-hot + num escaladas).

    Distancia entre variables: 1 - |corr|.
    Selección: 1 variable representante por cluster:
      - si y se provee: mayor mutual information con y
      - si no: mayor varianza

    Returns
    -------
    pd.DataFrame
        columnas: cluster, feature, score
    """
    pre = construir_preprocesador(X)
    X_proc = pre.fit_transform(X)
    feat_names = list(pre.get_feature_names_out())

    # --- FIX: eliminar columnas constantes (varianza 0) para evitar warnings en corrcoef ---
    var = np.var(X_proc, axis=0)
    mask = var > 0

    # si casi todo es constante, no hay clustering útil
    if np.sum(mask) < 2:
        return pd.DataFrame(columns=["cluster", "feature", "score"])

    X_proc = X_proc[:, mask]
    feat_names = [n for i, n in enumerate(feat_names) if mask[i]]

    # correlación entre columnas (variables)
    corr = np.corrcoef(X_proc, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1 - np.abs(corr)

    # compatibilidad sklearn (metric vs affinity)
    try:
        clust = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        clust = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    labels = clust.fit_predict(dist)

    if y is not None:
        y_arr = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).values
        score = mutual_info_classif(
            X_proc, y_arr, discrete_features="auto", random_state=random_state
        )
    else:
        score = np.var(X_proc, axis=0)

    rep_idx = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        best = idx[np.argmax(score[idx])]
        rep_idx.append(best)

    out = pd.DataFrame({
        "cluster": labels[rep_idx],
        "feature": [feat_names[i] for i in rep_idx],
        "score": [float(score[i]) for i in rep_idx],
    }).sort_values("score", ascending=False).reset_index(drop=True)

    return out.head(top_n)


def selectkbest_top(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 15,
    ) -> pd.DataFrame:
    """
    Aplica SelectKBest (ANOVA F) sobre el dataset preprocesado (one-hot + num).

    Returns
    -------
    pd.DataFrame
        columnas: feature, f_score (desc)
    """
    pre = construir_preprocesador(X)
    X_proc = pre.fit_transform(X)
    feat_names = list(pre.get_feature_names_out())

    y_arr = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).values

    # --- FIX: quitar columnas constantes para evitar warnings en f_classif ---
    var = np.var(X_proc, axis=0)
    mask = var > 0

    if np.sum(mask) == 0:
        return pd.DataFrame(columns=["feature", "f_score"])

    X_proc = X_proc[:, mask]
    feat_names = [n for i, n in enumerate(feat_names) if mask[i]]

    sel = SelectKBest(score_func=f_classif, k=min(k, X_proc.shape[1]))
    sel.fit(X_proc, y_arr)

    scores = sel.scores_
    # --- FIX: si hay NaN en scores, que no rompa el ordenamiento ---
    scores = np.nan_to_num(scores, nan=-np.inf)

    idx = np.argsort(scores)[::-1][:min(k, len(scores))]

    return pd.DataFrame({
        "feature": [feat_names[i] for i in idx],
        "f_score": [float(scores[i]) for i in idx],
    })


def ranking_iv(
    X: pd.DataFrame,
    y_bin: pd.Series,
    n_bins: int = 5,
    eps: float = 1e-6,
    ) -> pd.DataFrame:
    """
    Calcula IV por variable (aprox.) para objetivo binario.

    - Numéricas: binning por cuantiles (qcut)
    - Categóricas: categorías tal cual
    - NA se trata como "(missing)"

    Returns
    -------
    pd.DataFrame
        columnas: variable, iv (desc)
    """
    y = pd.to_numeric(y_bin, errors="coerce").fillna(0).astype(int)
    X2 = X.copy()

    total_good = (y == 0).sum()
    total_bad = (y == 1).sum()

    rows = []
    for col in X2.columns:
        s = X2[col]
        df = pd.DataFrame({"x": s, "y": y})

        if pd.api.types.is_numeric_dtype(df["x"]):
            xnum = pd.to_numeric(df["x"], errors="coerce")
            if xnum.nunique(dropna=True) > n_bins:
                bins = pd.qcut(xnum, q=n_bins, duplicates="drop")
            else:
                bins = xnum
            df["bin"] = bins.astype("string")
        else:
            df["bin"] = df["x"].astype("string")

        df["bin"] = df["bin"].fillna("(missing)")
        df.loc[df["bin"].astype("string").str.lower().isin(["nan", "none"]), "bin"] = "(missing)"

        g = df.groupby("bin")["y"].agg(["count", "sum"])
        g = g.rename(columns={"sum": "bad"})
        g["good"] = g["count"] - g["bad"]

        g["dist_good"] = (g["good"] + eps) / (total_good + eps)
        g["dist_bad"] = (g["bad"] + eps) / (total_bad + eps)

        woe = np.log(g["dist_good"] / g["dist_bad"])
        iv = float(((g["dist_good"] - g["dist_bad"]) * woe).sum())

        rows.append({"variable": col, "iv": iv})

    out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return out
