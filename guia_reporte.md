Guia de pasos y celdas sugeridas para el notebook del examen. Copia/pega en `reporte.ipynb` segun avance.

## Setup y carga
```python
import pandas as pd
import numpy as np
import examen.cargar_datos as cd
import examen.validar_datos as vd
import examen.perfil_datos as ped
import examen.analiticas as an
from importlib import reload

# autoreload modulos
cd, vd, ped, an = map(reload, (cd, vd, ped, an))

dfs = cd.cargar_dfs_desde_zip_en_data()
dfs_csv = cd.obtener_csvs(dfs)
list(dfs.keys()), list(dfs_csv.keys())
```

## Validacion CSV vs XLSX
```python
reps = vd.reporte_todos_pares_completo(dfs)
for base, rep in reps.items():
    print(rep["summary"])
    display(rep["diffs"].head())
```

## Perfilado y relaciones
```python
perf = ped.perfilar_todos_los_dfs(dfs_csv)
rel = ped.reporte_relacional(dfs_csv, ratings_key="ratings_csv")
list(perf.keys())
```

## Tablas analiticas y base de modelado
```python
dim_user = an.construir_dim_usuario(dfs_csv)
dim_place = an.construir_dim_restaurante(dfs_csv)

an.exportar_tablas_analiticas(dim_user, dim_place, out_dir="./data")
base = an.construir_base_modelado(dfs_csv, dim_user, dim_place)
base.to_csv("base.csv", index=False)
base.shape, base.head()
```

## Ingenieria de variables (>=5) y objetivo
```python
df = base.copy()

# Ejemplos de nuevas variables; ajusta segun analisis
df["age_bucket"] = pd.cut(df["age_ref"], bins=[0,25,35,50,80], labels=["joven","adulto","maduro","senior"])
df["is_smoker"] = (df["smoker"] == "true").astype(int)
df["payments_count"] = df["n_user_payments"].fillna(0)
df["cuisines_count"] = df["n_user_cuisines"].fillna(0)
df["rest_cuisines_count"] = df["n_rest_cuisines"].fillna(0)
df["rest_parking_count"] = df["n_rest_parking_types"].fillna(0)

# Variable objetivo: ejemplo binaria (alta calificacion)
df["target_high_rating"] = (df["rating"] >= 4).astype(int)
df.shape, df.head()
```

## Limpieza: nulos, outliers, unarias, correlacion 1:1
```python
clean = df.copy()

# 1) Quitar columnas unarias
unarias = [c for c in clean.columns if clean[c].nunique(dropna=True) <= 1]
clean = clean.drop(columns=unarias)

# 2) Imputacion simple: num -> mediana, cat -> moda
num_cols = clean.select_dtypes(include=[np.number]).columns
cat_cols = clean.select_dtypes(exclude=[np.number]).columns
clean[num_cols] = clean[num_cols].apply(lambda s: s.fillna(s.median()))
clean[cat_cols] = clean[cat_cols].apply(lambda s: s.fillna(s.mode().iloc[0]))

# 3) Outliers (IQR) en numericas: clip a [Q1-1.5*IQR, Q3+1.5*IQR]
def clip_iqr(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return s.clip(lo, hi)
clean[num_cols] = clean[num_cols].apply(clip_iqr)

# 4) Correlacion perfecta (|1|) entre numericas: drop duplicadas
corr = clean[num_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] == 1)]
clean = clean.drop(columns=to_drop_corr)

clean.shape, unarias, to_drop_corr
```

## Split y escalado
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

y = clean["target_high_rating"]
X = clean.drop(columns=["target_high_rating"])

num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

pre = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = Pipeline([
    ("prep", pre),
    ("model", LogisticRegression(max_iter=1000))
])

clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
```

## PCA (2D) y visualizacion
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

num_data = clean[num_cols]
scaled = StandardScaler().fit_transform(num_data)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(scaled)

plt.figure(figsize=(6,5))
plt.scatter(coords[:,0], coords[:,1], c=y, cmap="viridis", s=10, alpha=0.7)
plt.title("PCA 2D")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.colorbar(label="target_high_rating")
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=150)
plt.show()
```

## Seleccion de variables
```python
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import Pipeline

selector = SelectKBest(score_func=f_classif, k=15)

pipe_sel = Pipeline([
    ("prep", pre),
    ("sel", selector),
    ("model", LogisticRegression(max_iter=1000))
])

pipe_sel.fit(X_train, y_train)
print(classification_report(y_test, pipe_sel.predict(X_test)))
```

## Guardar artefactos finales
```python
clean.to_csv("base_limpia.csv", index=False)
print("Archivos generados:", ["data/usuarios.csv", "data/restaurantes.csv", "base.csv", "base_limpia.csv", "pca_2d.png"])
```
