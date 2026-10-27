"""Parte 1: análisis descriptivo de las hipótesis H1 a H4 del blog (sección 2.4).

Uso:
    python scripts/parte1_descriptivo.py

Se calcula solo con el periodo de entrenamiento (2012-08 a 2014-04), para no mirar los datos de
validación ni de prueba. emp_length no está en los datos procesados (se descartó en la selección),
así que se toma del archivo original por `id`; si el original no se puede cargar, H4 se omite.
También cuenta los estados de loan_status (tabla 2 del blog) y la proporción de créditos excluidos
(NA) por trimestre de originación.

Salidas: resultados/parte1/descriptivo_*.csv y blog/figuras/fig15_hipotesis.png, fig16_excluidos_por_cosecha.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.preprocesamiento import MAPEO_INCUMPLIMIENTO, cargar_datos_originales, cargar_datos_procesados, particion_temporal  # noqa: E402

SALIDA = RAIZ / "resultados" / "parte1"
FIGURAS = RAIZ / "blog" / "figuras"
LETRAS = list("ABCDEFG")


def tasa_por(df, columna):
    return df.groupby(columna, observed=True)["incumple"].agg(creditos="size", tasa="mean").reset_index()


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = cargar_datos_procesados()
    df = df[particion_temporal(df["issue_d"]) == "train"].copy()
    print(f"Periodo de entrenamiento: {len(df):,} créditos, tasa de incumplimiento {df['incumple'].mean():.3f}")

    try:
        og = cargar_datos_originales()
    except Exception as e:  # sin internet ni caché
        print(f"No se pudo cargar el archivo original ({e}); se omite H4 y los conteos de loan_status")
        og = None

    # H1: dti por quintiles; H2: int_rate por quintiles y letra de sub_grade; H3: plazo; H4: emp_length
    df["dti_quintil"] = pd.qcut(df["dti"], 5)
    df["int_rate_quintil"] = pd.qcut(df["int_rate"], 5)
    df["letra"] = pd.Categorical([LETRAS[int(s) // 5] for s in df["sub_grade"]], categories=LETRAS)
    df["plazo"] = np.where(df["term_ 60 months"] == 1, "60 meses", "36 meses")
    tablas = {"dti_quintil": tasa_por(df, "dti_quintil"), "int_rate_quintil": tasa_por(df, "int_rate_quintil"),
              "letra_sub_grade": tasa_por(df, "letra"), "plazo": tasa_por(df, "plazo")}
    if og is not None:
        emp = og.set_index("id")["emp_length"]
        anos = emp.str.extract(r"(\d+)")[0].astype(float)
        anos[emp.str.startswith("<", na=False)] = 0
        df["emp_length"] = df["id"].map(anos)
        tablas["emp_length"] = tasa_por(df.dropna(subset=["emp_length"]), "emp_length")

    for nombre, t in tablas.items():
        t.to_csv(SALIDA / f"descriptivo_{nombre}.csv", index=False)
        print(f"\n{nombre}:\n{t.round(4).to_string(index=False)}")

    paneles = [("dti_quintil", "H1: quintil de dti"), ("int_rate_quintil", "H2: quintil de int_rate"),
               ("letra_sub_grade", "H2: letra de sub_grade"), ("plazo", "H3: plazo"), ("emp_length", "H4: años de empleo")]
    paneles = [p for p in paneles if p[0] in tablas]
    fig, axes = plt.subplots(1, len(paneles), figsize=(4 * len(paneles), 4), sharey=True)
    for ax, (nombre, titulo) in zip(axes, paneles):
        t = tablas[nombre]
        etiquetas = [f"Q{i + 1}" for i in range(len(t))] if "quintil" in nombre else t.iloc[:, 0].astype(str)
        if nombre == "emp_length":
            etiquetas = ["<1" if v == 0 else ("10+" if v == 10 else str(int(v))) for v in t["emp_length"]]
        ax.bar(range(len(t)), t["tasa"], color="#c0392b", alpha=0.75)
        ax.set(xticks=range(len(t)), xticklabels=etiquetas, title=titulo)
        ax.axhline(df["incumple"].mean(), color="k", linestyle=":", linewidth=1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Tasa de incumplimiento (entrenamiento)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "fig15_hipotesis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nfigura: blog/figuras/fig15_hipotesis.png")

    if og is not None:
        # Conteo de estados (tabla 2) y proporción excluida por cosecha, en el mismo rango de fechas del modelo
        conteo = og["loan_status"].value_counts().rename_axis("loan_status").reset_index(name="creditos")
        conteo["codificacion"] = conteo["loan_status"].map(MAPEO_INCUMPLIMIENTO).astype("Int64")
        conteo.to_csv(SALIDA / "descriptivo_loan_status.csv", index=False)
        print(f"\n{conteo.to_string(index=False)}")

        og = og.assign(issue_d=pd.to_datetime(og["issue_d"], format="%b-%Y"))
        og = og[og["issue_d"] >= "2012-08-01"]
        og["trimestre"] = og["issue_d"].dt.to_period("Q").dt.start_time
        og["incluido"] = og["loan_status"].isin(MAPEO_INCUMPLIMIENTO)
        og["incumple"] = og["loan_status"].map(MAPEO_INCUMPLIMIENTO)
        cosechas = og.groupby("trimestre").agg(creditos=("incluido", "size"), incluidos=("incluido", "mean"),
                                               tasa_incumplimiento=("incumple", "mean")).reset_index()
        cosechas.to_csv(SALIDA / "descriptivo_excluidos_por_cosecha.csv", index=False)
        print(f"\n{cosechas.round(4).to_string(index=False)}")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(cosechas["trimestre"], 1 - cosechas["incluidos"], "o-", color="#7f8c8d", label="Proporción excluida (NA)")
        ax.plot(cosechas["trimestre"], cosechas["tasa_incumplimiento"], "o-", color="#c0392b", label="Tasa de incumplimiento de los incluidos")
        for corte, texto in [("2014-05-01", "validación"), ("2014-11-01", "prueba")]:
            ax.axvline(pd.Timestamp(corte), color="#9a9a9a", linestyle="--", linewidth=1)
            ax.text(pd.Timestamp(corte), 1.0, f" {texto}", va="top", color="#5a5a5a", fontsize=9)
        ax.set(ylim=(0, 1.05), xlabel="Trimestre de originación", ylabel="Proporción",
               title="Créditos excluidos e incumplimiento por cosecha")
        ax.legend(frameon=False, loc="center left")
        ax.grid(alpha=0.3)
        fig.savefig(FIGURAS / "fig16_excluidos_por_cosecha.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("figura: blog/figuras/fig16_excluidos_por_cosecha.png")


if __name__ == "__main__":
    main()
