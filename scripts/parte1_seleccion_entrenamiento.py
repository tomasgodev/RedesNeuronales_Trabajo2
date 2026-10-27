"""Repite la selección de variables de notebooks/procesamiento.ipynb solo con el periodo de entrenamiento.

Uso:
    python scripts/parte1_seleccion_entrenamiento.py

En el notebook, ANOVA, correlación de Pearson y V de Cramér se calcularon con todos los créditos,
incluidos los de validación y prueba. Este script aplica las mismas pruebas, con la misma limpieza
(celdas 3 a 9) y la misma función cramers_v (celda 14), a todos los créditos y solo a los de
entrenamiento (2012-08 a 2014-04), para ver si la selección cambia. No modifica nada.
Necesita el archivo original (python -m src.preprocesamiento lo descarga la primera vez).

Reglas implícitas en el notebook (celdas 11, 13 y 16):
- ANOVA: se descarta la variable numérica si p > 0.05.
- Pearson: entre dos numéricas con |r| > 0.8 se descarta la redundante (se conservan loan_amnt y revol_bal).
- V de Cramér con el incumplimiento: se descartaron las de V <= 0.03 y se conservaron las de V >= 0.06;
  grade se descartó por tener V = 1 con sub_grade.

Salida: resultados/parte1/seleccion_entrenamiento.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.preprocesamiento import (  # noqa: E402
    COLUMNAS_NUMERICAS, COLUMNAS_ONE_HOT, MAPEO_INCUMPLIMIENTO, MAPEOS_ORDINALES, cargar_datos_originales,
    particion_temporal,
)

DESCARTADAS_NOTEBOOK = {"pub_rec", "tot_coll_amt", "funded_amnt", "funded_amnt_inv", "installment",
                        "total_rev_hi_lim", "emp_length", "pymnt_plan", "initial_list_status", "application_type", "grade"}


def cramers_v(x, y):
    # COPIA de notebooks/procesamiento.ipynb, celda 14
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    if n == 0 or r == 1 or k == 1:
        return 0.0

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def limpiar(df_og):
    """Celdas 3 a 8 del notebook."""
    df = df_og[df_og["loan_status"].isin(MAPEO_INCUMPLIMIENTO.keys())].copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["loan_status"] = df["loan_status"].map(MAPEO_INCUMPLIMIENTO)
    excluir = ["id", "member_id", "policy_code", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d", "url",
               "emp_title", "desc", "title", "zip_code", "earliest_cr_line", "out_prncp", "out_prncp_inv",
               "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
               "recoveries", "collection_recovery_fee", "last_pymnt_amnt"]
    con_nulos = ["annual_inc_joint", "verification_status_joint", "dti_joint", "il_util", "mths_since_rcnt_il",
                 "inq_fi", "open_rv_24m", "open_acc_6m", "all_util", "inq_last_12m", "total_cu_tl", "open_il_12m",
                 "max_bal_bc", "open_il_6m", "open_il_24m", "open_rv_12m", "total_bal_il", "desc",
                 "mths_since_last_record", "mths_since_last_major_derog", "mths_since_last_delinq", "next_pymnt_d"]
    df = df.drop(columns=excluir).drop(columns=con_nulos, errors="ignore")
    return df.dropna(subset=["total_rev_hi_lim", "tot_cur_bal", "tot_coll_amt", "emp_length", "revol_util",
                             "collections_12_mths_ex_med"]).copy()


def pruebas(df):
    """Celdas 9 a 14: ANOVA, correlaciones relevantes y V de Cramér."""
    numericas = df.select_dtypes(include=["number"]).drop(columns=["loan_status"])
    categoricas = df.select_dtypes(include=["object", "category", "str"])
    f, p = f_classif(numericas, df["loan_status"])
    filas = [{"variable": c, "prueba": "ANOVA (p)", "valor": pv} for c, pv in zip(numericas.columns, p)]
    corr = numericas.corr()
    for a, b in [("loan_amnt", "funded_amnt"), ("loan_amnt", "funded_amnt_inv"), ("loan_amnt", "installment"),
                 ("revol_bal", "total_rev_hi_lim")]:
        filas.append({"variable": b, "prueba": f"Pearson con {a}", "valor": corr.loc[a, b]})
    for c in categoricas.columns:
        filas.append({"variable": c, "prueba": "V de Cramér con incumplimiento", "valor": cramers_v(categoricas[c], df["loan_status"])})
    filas.append({"variable": "grade", "prueba": "V de Cramér con sub_grade", "valor": cramers_v(df["grade"], df["sub_grade"])})
    return pd.DataFrame(filas)


def decidir(fila):
    v = fila["valor"]
    if fila["prueba"].startswith("ANOVA"):
        return v > 0.05
    if fila["prueba"].startswith("Pearson") or fila["prueba"].endswith("sub_grade"):
        return abs(v) > 0.8
    return v < 0.05


def main():
    df = limpiar(cargar_datos_originales())
    train = df[particion_temporal(df["issue_d"]) == "train"]
    print(f"Créditos: todos {len(df):,} | entrenamiento {len(train):,}")
    todos, ent = pruebas(df), pruebas(train)
    tabla = todos.merge(ent, on=["variable", "prueba"], suffixes=("_todos", "_entrenamiento"))
    tabla["descartada_en_notebook"] = tabla["variable"].isin(DESCARTADAS_NOTEBOOK)
    tabla["se_descarta_todos"] = tabla.rename(columns={"valor_todos": "valor"}).apply(decidir, axis=1)
    tabla["se_descarta_entrenamiento"] = tabla.rename(columns={"valor_entrenamiento": "valor"}).apply(decidir, axis=1)
    salida = RAIZ / "resultados" / "parte1" / "seleccion_entrenamiento.csv"
    tabla.to_csv(salida, index=False)
    pd.set_option("display.width", 200)
    print(tabla.round(4).to_string(index=False))

    elegidas = set(COLUMNAS_NUMERICAS) | set(MAPEOS_ORDINALES) | set(COLUMNAS_ONE_HOT)
    desc_ent = set(tabla.loc[tabla["se_descarta_entrenamiento"], "variable"])
    print(f"\nDescartadas por el notebook (todos los datos): {sorted(DESCARTADAS_NOTEBOOK)}")
    print(f"Se descartarían con solo entrenamiento:        {sorted(desc_ent)}")
    print("La selección no cambia." if desc_ent == DESCARTADAS_NOTEBOOK and not (desc_ent & elegidas)
          else "La selección CAMBIA.")
    print(f"Guardado en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
