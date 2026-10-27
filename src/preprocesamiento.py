"""Preprocesamiento compartido por la Parte 1 (PD) y la Parte 2 (LGD).

Genera los datos procesados desde el archivo original con un solo comando:

    python -m src.preprocesamiento

La selección de variables se justifica en notebooks/procesamiento.ipynb
(ANOVA, matriz de correlación de Pearson y V de Cramér).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_PROCESADOS = RAIZ / "datasets" / "creditos_procesados.csv.gz"

# Codificación de loan_status según el enunciado; los estados que no aparecen son NA y se excluyen
MAPEO_INCUMPLIMIENTO = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Solo los castigados tienen recuperaciones observadas, y por tanto LGD (docs/parte2_lgd.md)
ESTADOS_CASTIGADOS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]

# Filas con nulos en estas columnas se eliminan, como en procesamiento.ipynb
COLUMNAS_SIN_NULOS = [
    "total_rev_hi_lim", "tot_cur_bal", "tot_coll_amt",
    "emp_length", "revol_util", "collections_12_mths_ex_med",
]

COLUMNAS_NUMERICAS = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "revol_bal", "revol_util", "total_acc", "collections_12_mths_ex_med",
    "acc_now_delinq", "tot_cur_bal",
]

_LETRAS = ["A", "B", "C", "D", "E", "F", "G"]
MAPEOS_ORDINALES = {
    "sub_grade": {f"{l}{n}": i for i, (l, n) in enumerate((l, n) for l in _LETRAS for n in range(1, 6))},
    "verification_status": {"Not Verified": 0, "Source Verified": 1, "Verified": 2},
}

COLUMNAS_ONE_HOT = ["term", "home_ownership", "purpose", "addr_state"]

COLUMNAS_META = ["id", "issue_d", "loan_status", "incumple", "ead", "lgd"]

# Partición temporal común a ambas tareas (≈ 61/20/19 % en clasificación)
INICIO_VALIDACION = "2014-05-01"
INICIO_PRUEBA = "2014-11-01"


def cargar_datos_originales():
    import kagglehub

    ruta = kagglehub.dataset_download("ranadeep/credit-risk-dataset")
    return pd.read_csv(os.path.join(ruta, "loan", "loan.csv"), low_memory=False)


def construir_objetivos(df_og):
    df = df_og[df_og["loan_status"].isin(MAPEO_INCUMPLIMIENTO)].copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["incumple"] = df["loan_status"].map(MAPEO_INCUMPLIMIENTO)

    castigado = df["loan_status"].isin(ESTADOS_CASTIGADOS)
    ead = df["funded_amnt"] - df["total_rec_prncp"]
    recuperacion_neta = df["recoveries"] - df["collection_recovery_fee"]
    df["ead"] = ead.where(castigado)
    df["lgd"] = (1 - recuperacion_neta / ead).clip(0, 1).where(castigado)
    return df


def codificar(df, columnas=None):
    """Variables de entrada sin escalar. Con `columnas`, reordena y completa los dummies faltantes con 0."""
    X = df[COLUMNAS_NUMERICAS + list(MAPEOS_ORDINALES) + COLUMNAS_ONE_HOT].copy()
    for columna, mapeo in MAPEOS_ORDINALES.items():
        X[columna] = X[columna].map(mapeo)
    X = pd.get_dummies(X, columns=COLUMNAS_ONE_HOT, drop_first=True, dtype=int)
    if columnas is not None:
        X = X.reindex(columns=columnas, fill_value=0)
    return X


def generar_datos(df_og=None):
    if df_og is None:
        df_og = cargar_datos_originales()
    df = construir_objetivos(df_og).dropna(subset=COLUMNAS_SIN_NULOS)
    X = codificar(df)
    if X.isna().any().any():
        raise ValueError("Quedaron nulos en las variables de entrada")
    return pd.concat([df[COLUMNAS_META], X], axis=1).reset_index(drop=True)


def cargar_datos_procesados(ruta=RUTA_PROCESADOS):
    return pd.read_csv(ruta, parse_dates=["issue_d"])


def columnas_entrada(df):
    return [c for c in df.columns if c not in COLUMNAS_META]


def particion_temporal(fechas, inicio_validacion=INICIO_VALIDACION, inicio_prueba=INICIO_PRUEBA):
    return np.select(
        [fechas < pd.Timestamp(inicio_validacion), fechas < pd.Timestamp(inicio_prueba)],
        ["train", "val"],
        default="test",
    )


if __name__ == "__main__":
    df = generar_datos()
    df.to_csv(RUTA_PROCESADOS, index=False)
    print(f"{len(df):,} créditos y {len(columnas_entrada(df))} variables de entrada")
    print(f"Con LGD observada: {df['lgd'].notna().sum():,}")
    print(f"Guardado en {RUTA_PROCESADOS}")
