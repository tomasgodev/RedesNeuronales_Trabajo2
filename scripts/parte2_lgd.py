"""Parte 2 (LGD) con un solo comando y sin internet. Reproduce notebooks/regresion.ipynb.

Uso:
    python scripts/parte2_lgd.py                  # evalúa los modelos guardados y reentrena la configuración final (≈ 40 s)
    python scripts/parte2_lgd.py --lr 0.001 --perdida bce --capas 32   # reentrena con otros hiperparámetros
    python scripts/parte2_lgd.py --demo           # además, búsqueda y semillas cortas (< 2 min)
    python scripts/parte2_lgd.py --busqueda       # además, búsqueda aleatoria (30) y 30 semillas (≈ 20 min)
    python scripts/parte2_lgd.py --descriptivo    # además, estadísticas de la LGD con el archivo original

Usa los datos procesados del repositorio, así que no necesita internet; solo --descriptivo necesita el
archivo original. No modifica nada del notebook: lee sus resultados (resultados/*_lgd.*) y sus modelos
(weights/modelo_lgd.pth y weights/regresion_lineal_lgd.joblib). Lo que entrena lo guarda en
resultados/parte2/ (o resultados/demo/parte2/). Cada sección indica la celda del notebook que reproduce.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.modelos import (  # noqa: E402
    RUTA_LINEAL_LGD, a_tensor, cargar_modelo_lgd, contar_parametros, entrenar_lgd, metricas_regresion, predecir,
)
from src.validacion import (  # noqa: E402
    entero_no_negativo, entero_positivo, fraccion, leer_config, no_negativo, revisar, revisar_espacio, tamano_lote,
)
from src.preprocesamiento import (  # noqa: E402
    INICIO_PRUEBA, INICIO_VALIDACION, cargar_datos_procesados, columnas_entrada, particion_temporal,
)

CONJUNTOS = ["train", "val", "test"]
COLORES = [("Observada", "#3d3d3d"), ("Regresión lineal", "#2a6fdb"), ("MLP", "#e8833a")]


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(RAIZ / "configs" / "parte2.json"))
    p.add_argument("--demo", action="store_true", help="búsqueda y semillas cortas (< 2 min)")
    p.add_argument("--busqueda", action="store_true", help="búsqueda aleatoria y 30 semillas completas")
    p.add_argument("--descriptivo", action="store_true", help="estadísticas de la LGD con el archivo original")
    g = p.add_argument_group("red que se reentrena (por defecto, resultados/config_final_lgd.json)")
    g.add_argument("--capas", type=entero_positivo, nargs="+")
    g.add_argument("--activacion", choices=["relu", "tanh", "leaky_relu"])
    g.add_argument("--dropout", type=fraccion, help="entre 0 y 1")
    g.add_argument("--lr", type=no_negativo)
    g.add_argument("--weight_decay", type=no_negativo)
    g.add_argument("--batch_size", type=tamano_lote)
    g.add_argument("--perdida", choices=["bce", "mse"])
    g.add_argument("--epocas", type=entero_positivo)
    g.add_argument("--paciencia", type=entero_no_negativo)
    g.add_argument("--semilla", type=entero_no_negativo)
    g.add_argument("--n_busqueda", type=entero_positivo)
    g.add_argument("--n_semillas", type=entero_positivo)
    return p.parse_args()


def guardar(fig, carpeta, nombre):
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {(carpeta / nombre).relative_to(RAIZ)}")


def descriptivo(salida, figuras):
    """Celdas 4 a 8: todos los castigados del archivo original (2007-2015)."""
    from src.preprocesamiento import cargar_datos_originales, construir_objetivos
    df_obj = construir_objetivos(cargar_datos_originales())
    estados = df_obj[df_obj["incumple"] == 1].groupby("loan_status").agg(
        n=("loan_status", "size"),
        pct_con_recuperacion=("recoveries", lambda s: round((s > 0).mean() * 100, 2)),
        out_prncp_promedio=("out_prncp", "mean"))
    print(f"\n[Celda 4] Incumplidos por estado:\n{estados.round(2).to_string()}")
    df_lgd = df_obj[df_obj["lgd"].notna()]
    sin_recorte = 1 - (df_lgd["recoveries"] - df_lgd["collection_recovery_fee"]) / df_lgd["ead"]
    print(f"\n[Celda 5] Castigados: {len(df_lgd):,} | LGD < 0 antes del recorte: {(sin_recorte < 0).sum()} | "
          f"LGD > 1: {(sin_recorte > 1).sum()} | LGD = 1: {(df_lgd['lgd'] == 1).mean():.2%}")
    print(df_lgd["lgd"].describe().round(4).to_string())
    por_anio = df_lgd.groupby(df_lgd["issue_d"].dt.year).agg(
        n=("lgd", "size"), lgd_promedio=("lgd", "mean"),
        pct_sin_recuperacion=("recoveries", lambda s: round((s == 0).mean() * 100, 2)))
    print(f"\n[Celda 8] LGD por año de originación:\n{por_anio.round(4).to_string()}")
    estados.to_csv(salida / "incumplidos_por_estado.csv")
    por_anio.to_csv(salida / "lgd_por_anio.csv")

    fig, ax = plt.subplots(figsize=(8, 4))  # celda 6
    ax.hist(df_lgd["lgd"], bins=50, color="#2a6fdb", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("LGD")
    ax.set_ylabel("Número de créditos")
    ax.set_title("Distribución de la LGD en créditos castigados")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    guardar(fig, figuras, "fig05_distribucion_lgd.png")


def comparar(preds_por_modelo, y_real):
    """Celda 28: sesgo = error de nivel; Spearman = capacidad de ordenar, independiente del nivel."""
    filas = []
    for modelo, preds in preds_por_modelo:
        for nombre in CONJUNTOS:
            y_, p_ = np.asarray(y_real[nombre]), preds[nombre]
            filas.append({
                "modelo": modelo, "conjunto": nombre, **metricas_regresion(y_, p_),
                "RMSE_sin_sesgo": np.sqrt(np.mean(((y_ - y_.mean()) - (p_ - p_.mean())) ** 2)),
                "sesgo": p_.mean() - y_.mean(),
                "spearman": spearmanr(y_, p_).statistic if len(np.unique(p_)) > 1 else np.nan,
            })
    return pd.DataFrame(filas)


def main():
    args = leer_argumentos()
    cfg = leer_config(args.config)
    revisar([("semilla", cfg["semilla"], entero_no_negativo), ("epocas", cfg["epocas"], entero_positivo),
             ("paciencia", cfg["paciencia"], entero_no_negativo), ("n_busqueda", cfg["n_busqueda"], entero_positivo),
             ("n_semillas", cfg["n_semillas"], entero_positivo), ("demo.n_busqueda", cfg["demo"]["n_busqueda"], entero_positivo),
             ("demo.n_semillas", cfg["demo"]["n_semillas"], entero_positivo), ("demo.epocas", cfg["demo"]["epocas"], entero_positivo)],
            Path(args.config).name)
    revisar_espacio(cfg["espacio_busqueda"], Path(args.config).name)
    t0 = time.perf_counter()
    salida = RAIZ / "resultados" / ("demo/parte2" if args.demo else "parte2")
    figuras = salida / "figuras" if args.demo else RAIZ / "blog" / "figuras"
    salida.mkdir(parents=True, exist_ok=True)

    if args.descriptivo:
        descriptivo(salida, figuras)

    # ------------------------------------------------ celdas 10 y 11: datos procesados y partición
    df = cargar_datos_procesados()
    df_modelo = df[df["lgd"].notna()].sort_values("issue_d").reset_index(drop=True)
    columnas_x = columnas_entrada(df)
    particion = particion_temporal(df_modelo["issue_d"])
    print(f"\n[Celda 10] Créditos castigados: {len(df_modelo):,} | Variables de entrada: {len(columnas_x)}")
    print(df_modelo.groupby(particion).agg(n=("lgd", "size"), desde=("issue_d", "min"), hasta=("issue_d", "max"),
                                           lgd_promedio=("lgd", "mean")).loc[CONJUNTOS].to_string())
    y_real = {n: df_modelo.loc[particion == n, "lgd"] for n in CONJUNTOS}
    scaler = StandardScaler().fit(df_modelo.loc[particion == "train", columnas_x])
    X = {n: scaler.transform(df_modelo.loc[particion == n, columnas_x]) for n in CONJUNTOS}
    X_t = {n: a_tensor(X[n]) for n in CONJUNTOS}
    y_t = {n: a_tensor(y_real[n]) for n in ["train", "val"]}

    # ------------------------------------------------ celda 13: regresión lineal (determinista)
    t = time.perf_counter()
    lineal = LinearRegression().fit(X["train"], y_real["train"])
    segundos_lineal = time.perf_counter() - t
    guardada = joblib.load(RUTA_LINEAL_LGD)
    dif = np.abs(lineal.predict(X["test"]) - guardada["modelo"].predict(guardada["scaler"].transform(
        df_modelo.loc[particion == "test", guardada["columnas"]]))).max()
    print(f"\n[Celda 13] Regresión lineal: {lineal.coef_.size + 1} parámetros, {segundos_lineal:.2f} s. "
          f"Diferencia máxima con la guardada en prueba: {dif:.1e}")

    # ------------------------------------------------ celdas 28 y 30: comparación con los modelos guardados
    mlp, scaler_g, columnas_g, config_g = cargar_modelo_lgd()
    assert columnas_g == columnas_x, "Las columnas del modelo guardado no coinciden"
    pred_lineal = {n: lineal.predict(X[n]) for n in CONJUNTOS}
    pred_mlp = {n: predecir(mlp, a_tensor(scaler_g.transform(df_modelo.loc[particion == n, columnas_x]))) for n in CONJUNTOS}
    pred_media = {n: np.full(len(y_real[n]), y_real["train"].mean()) for n in CONJUNTOS}
    tabla = comparar([("Media de entrenamiento", pred_media), ("Regresión lineal", pred_lineal), ("MLP", pred_mlp)], y_real)
    tabla.to_csv(salida / "comparacion_lgd.csv", index=False)
    print("\n[Celda 28] Comparación (red guardada en weights/modelo_lgd.pth):")
    print(tabla.set_index(["modelo", "conjunto"]).round(4).to_string())

    cuantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    dist = pd.DataFrame({
        "Observada": y_real["test"].quantile(cuantiles).to_numpy(),
        "Regresión lineal": np.quantile(pred_lineal["test"], cuantiles),
        "MLP": np.quantile(pred_mlp["test"], cuantiles),
    }, index=[f"p{int(q * 100)}" for q in cuantiles]).T.assign(
        desv_estandar=[y_real["test"].std(), pred_lineal["test"].std(), pred_mlp["test"].std()])
    dist.to_csv(salida / "distribucion_prueba.csv")
    print(f"\n[Celda 30] Distribución en prueba (LGD = 1 en el {(y_real['test'] == 1).mean():.1%} de los créditos):")
    print(dist.round(4).to_string())

    # ------------------------------------------------ celdas 29 y 31: figuras con los modelos guardados
    X_todo = scaler.transform(df_modelo[columnas_x])
    por_trimestre = pd.DataFrame({
        "Observada": df_modelo["lgd"], "Regresión lineal": lineal.predict(X_todo),
        "MLP": predecir(mlp, a_tensor(scaler_g.transform(df_modelo[columnas_x]))),
    }).groupby(df_modelo["issue_d"].dt.to_period("Q").dt.start_time).mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for columna, color in COLORES:
        ax.plot(por_trimestre.index, por_trimestre[columna], label=columna, color=color, linewidth=2, marker="o", markersize=4)
    for corte, texto in [(pd.Timestamp(INICIO_VALIDACION), "validación"), (pd.Timestamp(INICIO_PRUEBA), "prueba")]:
        ax.axvline(corte, color="#9a9a9a", linestyle="--", linewidth=1)
        ax.text(corte, ax.get_ylim()[1], f" {texto}", va="top", color="#5a5a5a", fontsize=9)
    ax.set_xlabel("Trimestre de originación")
    ax.set_ylabel("LGD promedio")
    ax.set_title("LGD promedio por trimestre: observada vs. predicha")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    guardar(fig, figuras, "fig07_lgd_por_trimestre.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (etiqueta, color), valores in zip(COLORES, [y_real["test"].to_numpy(), pred_lineal["test"], pred_mlp["test"]]):
        x = np.sort(valores)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=etiqueta, color=color, linewidth=2)
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("LGD")
    ax.set_ylabel("Proporción acumulada de créditos")
    ax.set_title("Distribución acumulada de la LGD en prueba: observada vs. predicha")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    guardar(fig, figuras, "fig08_ecdf_lgd.png")

    # ------------------------------------------------ celdas 33 y 34: EAD y pérdida esperada con PD = 1
    plazo_60 = df_modelo["term_ 60 months"] == 1
    fraccion = df_modelo["ead"] / df_modelo["loan_amnt"]
    factor_ead = fraccion[particion == "train"].groupby(plazo_60[particion == "train"]).mean()
    factor_ead.index = factor_ead.index.map({False: "36 meses", True: "60 meses"})
    ead_estimada = df_modelo["loan_amnt"] * np.where(plazo_60, factor_ead["60 meses"], factor_ead["36 meses"])
    en_test = particion == "test"
    perdida_real = df_modelo.loc[en_test, "lgd"] * df_modelo.loc[en_test, "ead"]
    perdida_estimada = 1.0 * pred_mlp["test"] * ead_estimada[en_test]
    print(f"\n[Celda 33] Factor de EAD: {factor_ead.round(4).to_dict()} "
          f"(guardado: {guardada['factor_ead'].round(4).to_dict()})")
    print(f"[Celda 34] Prueba con PD = 1: pérdida observada {perdida_real.sum():,.0f} USD; "
          f"estimada con la red y la EAD estimada {perdida_estimada.sum():,.0f} USD")

    # ------------------------------------------------ celdas 24 y 25: reentrenar la configuración final
    config_final = json.loads((RAIZ / "resultados" / "config_final_lgd.json").read_text())
    cambios = {k: v for k, v in {
        "capas_ocultas": args.capas, "activacion": args.activacion, "dropout": args.dropout, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size, "perdida": args.perdida,
        "epocas": args.epocas, "paciencia": args.paciencia}.items() if v is not None}
    config = {**config_final, **cambios}
    if args.demo and args.epocas is None:
        config["epocas"] = cfg["demo"]["epocas"]
    semilla = args.semilla if args.semilla is not None else cfg["semilla"]
    claves = ["capas_ocultas", "activacion", "dropout", "weight_decay", "lr", "batch_size", "perdida", "epocas", "paciencia"]
    print(f"\n[Celda 24] Reentrenando con semilla {semilla}: {({k: config[k] for k in claves})}"
          + (f"\n           Cambios frente a resultados/config_final_lgd.json: {cambios}" if cambios else ""))
    t = time.perf_counter()
    modelo, historia = entrenar_lgd(config, semilla, X_t["train"], y_t["train"], X_t["val"], y_t["val"], verbose=True)
    m = {n: metricas_regresion(y_real[n], predecir(modelo, X_t[n])) for n in ["val", "test"]}
    fila_guardada = tabla[(tabla["modelo"] == "MLP") & (tabla["conjunto"] == "test")].iloc[0]
    print(f"Red reentrenada: {contar_parametros(modelo):,} parámetros, {len(historia['val_loss'])} épocas, "
          f"{time.perf_counter() - t:.1f} s")
    print(f"  validación: MAE {m['val']['MAE']:.4f} | RMSE {m['val']['RMSE']:.4f}")
    print(f"  prueba:     MAE {m['test']['MAE']:.4f} | RMSE {m['test']['RMSE']:.4f} "
          f"(red guardada: MAE {fila_guardada['MAE']:.4f} | RMSE {fila_guardada['RMSE']:.4f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(historia["train_loss"], label="Entrenamiento", color="#2a6fdb", linewidth=2)
    ax.plot(historia["val_loss"], label="Validación", color="#e8833a", linewidth=2)
    ax.set_xlabel("Época")
    ax.set_ylabel(f"Pérdida ({config['perdida'].upper()})")
    ax.set_title("Curvas de entrenamiento del MLP final")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    oficial = not cambios and not args.demo and semilla == cfg["semilla"]
    guardar(fig, figuras if oficial else salida / "figuras", "fig06_curvas_mlp_lgd.png")
    resumen = {"red_reentrenada": {"config": {k: config[k] for k in claves}, "semilla": semilla,
                                   "parametros": contar_parametros(modelo), "epocas": len(historia["val_loss"]),
                                   **{f"{k}_{n}": round(v, 4) for n in m for k, v in m[n].items()}}}

    # ------------------------------------------------ celdas 19 a 22: búsqueda aleatoria y semillas
    if args.busqueda or args.demo:
        espacio = cfg["espacio_busqueda"]
        n_busqueda = args.n_busqueda or (cfg["demo"]["n_busqueda"] if args.demo else cfg["n_busqueda"])
        n_semillas = args.n_semillas or (cfg["demo"]["n_semillas"] if args.demo else cfg["n_semillas"])
        epocas = args.epocas or (cfg["demo"]["epocas"] if args.demo else cfg["epocas"])
        base = {"epocas": epocas, "paciencia": cfg["paciencia"]}
        rng = np.random.default_rng(cfg["semilla"])
        filas = []
        print(f"\n[Celda 19] Búsqueda aleatoria: {n_busqueda} configuraciones, máximo {epocas} épocas")
        for i in range(n_busqueda):
            # Misma lógica que muestrear_config de la celda 17
            config_i = {**base, **{k: v[rng.integers(len(v))] for k, v in espacio.items()}}
            t = time.perf_counter()
            modelo_i, historia_i = entrenar_lgd(config_i, cfg["semilla"], X_t["train"], y_t["train"],
                                                X_t["val"], y_t["val"], verbose=False)
            filas.append({**{k: config_i[k] for k in espacio}, "parametros": contar_parametros(modelo_i),
                          "epocas": len(historia_i["val_loss"]),
                          **{f"{k}_val": v for k, v in metricas_regresion(y_real["val"], predecir(modelo_i, X_t["val"])).items()},
                          "segundos": time.perf_counter() - t})
            print(f"{i + 1:02d}/{n_busqueda} | RMSE val: {filas[-1]['RMSE_val']:.4f} | {filas[-1]['segundos']:.1f} s")
        busqueda = pd.DataFrame(filas).sort_values("RMSE_val").reset_index(drop=True)
        busqueda.to_csv(salida / "busqueda_lgd.csv", index=False)
        mejor = {**base, **{k: busqueda.iloc[0][k] for k in espacio}}
        mejor = {k: (v.item() if hasattr(v, "item") else v) for k, v in mejor.items()}
        (salida / "config_final_lgd.json").write_text(json.dumps(mejor, indent=2))
        print(f"Búsqueda: {busqueda['segundos'].sum() / 60:.1f} min. Mejor (celda 20): "
              f"{({k: mejor[k] for k in espacio})}")

        filas = []
        for s in range(n_semillas):  # celda 22
            t = time.perf_counter()
            modelo_s, historia_s = entrenar_lgd(mejor, s, X_t["train"], y_t["train"], X_t["val"], y_t["val"], verbose=False)
            filas.append({"semilla": s, "epocas": len(historia_s["val_loss"]),
                          **{f"{k}_val": v for k, v in metricas_regresion(y_real["val"], predecir(modelo_s, X_t["val"])).items()},
                          **{f"{k}_test": v for k, v in metricas_regresion(y_real["test"], predecir(modelo_s, X_t["test"])).items()},
                          "segundos": time.perf_counter() - t})
        semillas = pd.DataFrame(filas)
        semillas.to_csv(salida / "semillas_lgd.csv", index=False)
        res = semillas[["MAE_val", "RMSE_val", "MAE_test", "RMSE_test"]].agg(["mean", "std", "min", "max"])
        res.index = ["media", "desv. estándar", "mejor", "peor"]
        print(f"\n[Celda 22] {n_semillas} semillas ({semillas['segundos'].sum() / 60:.1f} min):\n{res.round(4).to_string()}")
        resumen_busqueda = {"configuraciones": n_busqueda, "minutos": round(busqueda["segundos"].sum() / 60, 1),
                            "mejor": {k: mejor[k] for k in espacio},
                            "semillas": n_semillas, "minutos_semillas": round(semillas["segundos"].sum() / 60, 1)}
        (salida / "resumen_busqueda_lgd.json").write_text(json.dumps(resumen_busqueda, indent=2, ensure_ascii=False))

    (salida / "resumen_parte2.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nResultados en {salida.relative_to(RAIZ)} | tiempo total: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
