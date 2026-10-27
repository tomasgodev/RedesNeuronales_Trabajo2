"""Pérdida esperada E[L] = PD · LGD · EAD en el conjunto de prueba, con un solo comando.

Uso:
    python scripts/perdida_esperada.py
    python scripts/perdida_esperada.py --modelo_pd logistica

Combina los modelos guardados, sin reentrenar:
- PD: red de notebooks/clasificacion.ipynb (o la logística de scripts/parte1_evaluacion.py),
  con la corrección del peso de clase.
- LGD: red de notebooks/regresion.ipynb (weights/modelo_lgd.pth).
- EAD al originar: loan_amnt por la fracción promedio pendiente al castigo según el plazo
  (factor_ead guardado en weights/regresion_lineal_lgd.joblib; docs/parte2_lgd.md, sección 8).

La pérdida observada solo se conoce en los créditos castigados (LGD · EAD reales). En los que están
en Default o Late (31-120 days) aún no se conoce, así que la pérdida observada es una cota inferior.

Salidas: resultados/perdida_esperada/*.csv y blog/figuras/fig17_perdida_esperada.png
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.modelos import (  # noqa: E402
    a_tensor, calcular_pos_weight, cargar_clasificador, cargar_factor_ead, cargar_modelo_lgd, datos_parte1,
    estimar_ead, predecir, predecir_proba,
)

SALIDA = RAIZ / "resultados" / "perdida_esperada"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo_pd", choices=["red", "logistica"], default="red")
    args = p.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)

    d = datos_parte1()
    test = d["df"][d["particion"] == "test"].copy()
    pos_weight = calcular_pos_weight(d["y_train"])

    if args.modelo_pd == "red":
        red, scaler, columnas = cargar_clasificador()
        test["pd"] = predecir_proba(red, a_tensor(scaler.transform(test[columnas])), pos_weight)
    else:
        rl = joblib.load(RAIZ / "weights" / "regresion_logistica_pd.joblib")
        logit = rl["modelo"].decision_function(rl["scaler"].transform(test[rl["columnas"]])) - rl["ln_peso_clase"]
        test["pd"] = 1 / (1 + np.exp(-logit))

    modelo_lgd, scaler_lgd, columnas_lgd, _ = cargar_modelo_lgd()
    test["lgd_pred"] = predecir(modelo_lgd, a_tensor(scaler_lgd.transform(test[columnas_lgd])))
    test["ead_pred"] = estimar_ead(test["loan_amnt"], test["term_ 60 months"], cargar_factor_ead())
    test["perdida_esperada"] = test["pd"] * test["lgd_pred"] * test["ead_pred"]
    # Pérdida observada: castigados -> LGD · EAD reales; buenos -> 0; Default/Late -> desconocida
    test["perdida_observada"] = np.where(test["incumple"] == 0, 0.0, test["lgd"] * test["ead"])
    sin_observar = (test["incumple"] == 1) & test["lgd"].isna()

    resumen = {
        "modelo_pd": args.modelo_pd,
        "creditos": int(len(test)),
        "monto_total": round(float(test["loan_amnt"].sum()), 0),
        "pd_media": round(float(test["pd"].mean()), 4),
        "tasa_incumplimiento_observada": round(float(test["incumple"].mean()), 4),
        "lgd_media_predicha": round(float(test["lgd_pred"].mean()), 4),
        "ead_media_estimada": round(float(test["ead_pred"].mean()), 0),
        "perdida_esperada_total": round(float(test["perdida_esperada"].sum()), 0),
        "perdida_esperada_media": round(float(test["perdida_esperada"].mean()), 0),
        "perdida_esperada_sobre_monto": round(float(test["perdida_esperada"].sum() / test["loan_amnt"].sum()), 4),
        "perdida_observada_castigados_total": round(float(test["perdida_observada"].sum()), 0),
        "creditos_castigados": int(test["lgd"].notna().sum()),
        "creditos_incumplidos_sin_perdida_observada": int(sin_observar.sum()),
        "monto_incumplidos_sin_perdida_observada": round(float(test.loc[sin_observar, "loan_amnt"].sum()), 0),
    }
    (SALIDA / f"resumen_{args.modelo_pd}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    # Por decil de PD: ¿la pérdida esperada ordena la pérdida observada?
    test["decil_pd"] = pd.qcut(test["pd"].rank(method="first"), 10, labels=range(1, 11))
    con_obs = test[~sin_observar]
    deciles = test.groupby("decil_pd", observed=True).agg(
        creditos=("pd", "size"), pd_media=("pd", "mean"), tasa_observada=("incumple", "mean"),
        lgd_predicha_media=("lgd_pred", "mean"), ead_estimada_media=("ead_pred", "mean"),
        perdida_esperada_media=("perdida_esperada", "mean"))
    deciles["perdida_observada_media"] = con_obs.groupby("decil_pd", observed=True)["perdida_observada"].mean()
    deciles["incumplidos_sin_perdida_observada"] = test[sin_observar].groupby("decil_pd", observed=True).size()
    deciles = deciles.fillna({"incumplidos_sin_perdida_observada": 0}).reset_index()
    deciles.to_csv(SALIDA / f"deciles_{args.modelo_pd}.csv", index=False)
    print(deciles.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, 11)
    ax.bar(x - 0.2, deciles["perdida_esperada_media"], width=0.4, color="#e8833a", label="Pérdida esperada (modelos)")
    ax.bar(x + 0.2, deciles["perdida_observada_media"], width=0.4, color="#3d3d3d",
           label="Pérdida observada (sin Default ni Late)")
    ax.set(xticks=x, xlabel="Decil de probabilidad de incumplimiento (1 = menor riesgo)",
           ylabel="USD por crédito", title="Pérdida esperada y observada por decil en prueba")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(RAIZ / "blog" / "figuras" / "fig17_perdida_esperada.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figura: blog/figuras/fig17_perdida_esperada.png")


if __name__ == "__main__":
    main()
