"""Prepara app/opciones_app.json: categorías, valores por defecto y rangos de las variables de entrada.

Uso:
    python scripts/preparar_app.py

Las categorías completas (incluida la que eliminó drop_first en src/preprocesamiento.py) se leen del
archivo original; los valores por defecto (mediana o moda) y los rangos (percentiles 1 y 99) se
calculan solo con el periodo de entrenamiento de los datos procesados. Así la aplicación no
necesita internet ni el archivo original para correr.
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.preprocesamiento import (  # noqa: E402
    COLUMNAS_NUMERICAS, COLUMNAS_ONE_HOT, COLUMNAS_SIN_NULOS, MAPEOS_ORDINALES, cargar_datos_originales,
    cargar_datos_procesados, construir_objetivos, particion_temporal,
)

ENTEROS = {"delinq_2yrs", "inq_last_6mths", "open_acc", "total_acc", "collections_12_mths_ex_med", "acc_now_delinq"}


def main():
    # Categorías: de todos los créditos procesados, igual que get_dummies en src/preprocesamiento.py
    # (algunas, como home_ownership = ANY, solo aparecen después del periodo de entrenamiento).
    # Valores por defecto y rangos: solo del periodo de entrenamiento.
    og_todo = construir_objetivos(cargar_datos_originales()).dropna(subset=COLUMNAS_SIN_NULOS)
    og = og_todo[particion_temporal(og_todo["issue_d"]) == "train"]
    df = cargar_datos_procesados()
    df = df[particion_temporal(df["issue_d"]) == "train"]

    opciones = {"numericas": {}, "categoricas": {}}
    for c in COLUMNAS_NUMERICAS:
        s = df[c]
        opciones["numericas"][c] = {
            "defecto": float(s.median()), "min": float(s.quantile(0.01)), "max": float(s.quantile(0.99)),
            "entero": c in ENTEROS,
        }
    for c in COLUMNAS_ONE_HOT:
        opciones["categoricas"][c] = {"opciones": sorted(og_todo[c].unique()), "defecto": og[c].value_counts().index[0]}
    for c, mapeo in MAPEOS_ORDINALES.items():
        conteo = og[c].value_counts()
        opciones["categoricas"][c] = {"opciones": list(mapeo), "defecto": conteo.index[0]}

    ruta = RAIZ / "app" / "opciones_app.json"
    ruta.write_text(json.dumps(opciones, indent=2, ensure_ascii=False))
    print(f"Guardado en {ruta}")
    print(json.dumps({c: v["defecto"] for c, v in opciones["categoricas"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
