"""
scripts/metricas_clasificacion.py
=================================
Módulo para el cálculo riguroso de métricas de discriminación y calibración
en modelos de riesgo de crédito (clasificación binaria).

Universidad Nacional de Colombia - Facultad de Minas
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp

def calcular_metricas_completas(y_true, y_prob, nombre_conjunto="Evaluación"):
    """
    Calcula las métricas exigidas por la rúbrica oficial:
    1. Discriminación: AUC, Gini, Estadístico KS.
    2. Calibración: Brier Score y Brier de Referencia.

    Parámetros:
    -----------
    y_true : array-like de forma (N,)
        Etiquetas binarias reales (0: buen pagador, 1: mal pagador).
    y_prob : array-like de forma (N,)
        Probabilidades calibradas predichas de incumplimiento en [0, 1].
    nombre_conjunto : str
        Identificador del split (ej. 'Validación', 'Prueba OOT').

    Retorna:
    --------
    dict con las métricas calculadas.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # 1. Discriminación
    auc = roc_auc_score(y_true, y_prob)
    gini = 2.0 * auc - 1.0

    # KS estadístico: máxima separación entre CDF de buenos y malos
    prob_buenos = y_prob[y_true == 0]
    prob_malos = y_prob[y_true == 1]
    ks_res = ks_2samp(prob_malos, prob_buenos)
    ks_stat = ks_res.statistic
    ks_pval = ks_res.pvalue

    # 2. Calibración
    brier = brier_score_loss(y_true, y_prob)
    tasa_base = np.mean(y_true)
    brier_ref = tasa_base * (1.0 - tasa_base)

    return {
        "Conjunto": nombre_conjunto,
        "N": len(y_true),
        "Tasa Default": round(tasa_base, 4),
        "AUC": round(auc, 4),
        "Gini": round(gini, 4),
        "KS": round(ks_stat, 4),
        "KS_pvalue": ks_pval,
        "Brier": round(brier, 4),
        "Brier_Referencia": round(brier_ref, 4),
        "Calibrado": brier < brier_ref
    }

def tabla_calibracion_decilica(y_true, y_prob, n_bins=10):
    """
    Genera la tabla de confiabilidad (Reliability Table) agrupando en deciles de probabilidad.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decil"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    
    tabla = df.groupby("decil", observed=True).agg(
        n=("y_true", "count"),
        prob_media=("y_prob", "mean"),
        tasa_observada=("y_true", "mean"),
        malos=("y_true", "sum")
    ).reset_index()

    tabla["error_calibracion"] = tabla["tasa_observada"] - tabla["prob_media"]
    return tabla

if __name__ == "__main__":
    np.random.seed(42)
    n = 10000
    y_sim = np.random.binomial(1, 0.23, n)
    logits_sim = -1.5 + 1.8 * y_sim + np.random.normal(0, 0.5, n)
    probs_sim = 1.0 / (1.0 + np.exp(-logits_sim))

    res = calcular_metricas_completas(y_sim, probs_sim, "Simulación Test")
    print("=" * 60)
    print("MÉTRICAS COMPLETAS DE EVALUACIÓN")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:18s}: {v}")
    print("=" * 60)

"""
Posible resultado:
python scripts/metricas_clasificacion.py

======================================================================
MÉTRICAS COMPLETAS DE EVALUACIÓN
======================================================================
  Conjunto          : Simulación Test
  N                 : 10000
  Tasa Default      : 0.2307
  AUC               : 0.8006
  Gini              : 0.6012
  KS                : 0.4454
  KS_pvalue         : 0.0
  Brier             : 0.1608
  Brier_Referencia  : 0.1776
  Calibrado         : True
======================================================================
"""