import pandas as pd


def reporte_diferencias_csv_xlsx_completo(dfs: dict[str, pd.DataFrame], base: str) -> dict[str, object]:
    """
    Reporte COMPLETO de diferencias entre {base}_csv y {base}_xlsx.

    - NaN-safe: NaN == NaN.
    - Compara por posición (sin ordenar).
    - Si difiere #filas, compara hasta el mínimo.
    """
    a = dfs.get(f"{base}_csv")
    b = dfs.get(f"{base}_xlsx")
    if a is None or b is None:
        raise KeyError(f"No encontré el par '{base}_csv' y '{base}_xlsx'.")

    summary = {
        "base": base,
        "csv_shape": a.shape,
        "xlsx_shape": b.shape,
        "same_columns": None,
        "equals_pandas": None,
        "mismatch_cells": None,
        "note": "",
    }

    cols_a, cols_b = list(a.columns), list(b.columns)
    solo_a = [c for c in cols_a if c not in cols_b]
    solo_b = [c for c in cols_b if c not in cols_a]

    if solo_a or solo_b:
        summary["same_columns"] = False
        summary["note"] = f"Columnas solo en CSV: {solo_a} | Columnas solo en XLSX: {solo_b}"
        return {"summary": summary, "diffs": None}

    summary["same_columns"] = True

    a2 = a.reset_index(drop=True)
    b2 = b.reset_index(drop=True)

    n_min = min(len(a2), len(b2))
    if len(a2) != len(b2):
        summary["note"] = f"Filas distintas: csv={len(a2)} vs xlsx={len(b2)}. Comparo primeras {n_min} filas."
        a2 = a2.iloc[:n_min].copy()
        b2 = b2.iloc[:n_min].copy()

    summary["equals_pandas"] = bool(a2.equals(b2))
    if summary["equals_pandas"]:
        summary["mismatch_cells"] = 0
        return {"summary": summary, "diffs": pd.DataFrame(columns=["row", "col", "csv", "xlsx"])}

    diffs_mask = (a2.ne(b2)) & ~(a2.isna() & b2.isna())
    mismatch = int(diffs_mask.to_numpy().sum())
    summary["mismatch_cells"] = mismatch

    if mismatch == 0:
        return {"summary": summary, "diffs": pd.DataFrame(columns=["row", "col", "csv", "xlsx"])}

    mask_stacked = diffs_mask.stack()
    mismatch_index = mask_stacked[mask_stacked].index

    a_stacked = a2.stack()
    b_stacked = b2.stack()

    diffs_df = pd.DataFrame(
        {"csv": a_stacked.loc[mismatch_index].values,
         "xlsx": b_stacked.loc[mismatch_index].values},
        index=mismatch_index,
    ).reset_index()

    diffs_df.columns = ["row", "col", "csv", "xlsx"]
    return {"summary": summary, "diffs": diffs_df}


def reporte_todos_pares_completo(dfs: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    """
    Genera reporte COMPLETO para todos los pares *_csv vs *_xlsx encontrados en dfs.
    Ej: users_csv/users_xlsx, restaurants_csv/restaurants_xlsx, etc.
    """
    bases = sorted({k[:-4] for k in dfs if k.endswith("_csv") and (k[:-4] + "_xlsx") in dfs})
    return {base: reporte_diferencias_csv_xlsx_completo(dfs, base) for base in bases}