"""Parte 1: búsqueda aleatoria de hiperparámetros y semillas de la red de clasificación.

Uso:
    python scripts/parte1_busqueda.py                  # búsqueda + 30 semillas (≈ 45 min en CPU)
    python scripts/parte1_busqueda.py --demo           # versión corta para la sustentación (< 2 min)
    python scripts/parte1_busqueda.py --solo_semillas  # usa resultados/config_final_pd.json y corre solo las semillas
    python scripts/parte1_busqueda.py --n_configuraciones 10 --n_semillas 10 --epocas 20

Todos los parámetros vienen de configs/parte1.json y se pueden cambiar con argumentos.
La búsqueda reutiliza el espacio y el muestreo de notebooks/regresion.ipynb (Parte 2), sin la
opción de pérdida, porque en clasificación la pérdida es siempre la entropía cruzada ponderada.
La configuración de Tomás (notebooks/clasificacion.ipynb) entra como candidata número 0.

Salidas: resultados/busqueda_pd.csv, resultados/config_final_pd.json, resultados/semillas_pd.csv
y weights/modelo_pd_busqueda.pth. En modo demo se escriben en resultados/demo/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.modelos import (  # noqa: E402
    a_tensor, calcular_pos_weight, construir_clasificador, contar_parametros, datos_parte1,
    entrenar_clasificador, predecir_proba,
)
from metricas_clasificacion import calcular_metricas_completas  # noqa: E402  (scripts/metricas_clasificacion.py)
from src.validacion import (  # noqa: E402
    entero_no_negativo, entero_positivo, leer_config, no_negativo, revisar, revisar_espacio, salir, tamano_lote,
)

METRICAS = ["AUC", "Gini", "KS", "Brier"]


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(RAIZ / "configs" / "parte1.json"))
    p.add_argument("--demo", action="store_true", help="muestra de entrenamiento y pocas corridas (< 2 min)")
    p.add_argument("--solo_semillas", action="store_true", help="salta la búsqueda y usa la configuración final guardada")
    p.add_argument("--n_configuraciones", type=entero_no_negativo, help="configuraciones aleatorias de la búsqueda")
    p.add_argument("--n_semillas", type=entero_positivo, help="semillas de la configuración final")
    p.add_argument("--epocas", type=entero_positivo, help="máximo de épocas por entrenamiento")
    p.add_argument("--semilla", type=entero_no_negativo, help="semilla de la búsqueda")
    return p.parse_args()


def muestrear_config(rng, espacio):
    # Misma lógica que muestrear_config de notebooks/regresion.ipynb (celda 17)
    return {k: v[rng.integers(len(v))] for k, v in espacio.items()}


def describir(config):
    if config.get("modelo", "CreditRiskNN") == "CreditRiskNN":
        return {"capas_ocultas": "[64, 32, 16]", "activacion": "relu", "dropout": "0.2 (0 en la última capa)"}
    return {"capas_ocultas": str(config["capas_ocultas"]), "activacion": config["activacion"], "dropout": config["dropout"]}


def evaluar(model, X_t, y, pos_weight, nombre):
    m = calcular_metricas_completas(y, predecir_proba(model, X_t, pos_weight), nombre)
    return {f"{k}_{nombre}": float(m[k]) for k in METRICAS}


def main():
    args = leer_argumentos()
    cfg = leer_config(args.config)
    nombre_cfg, b, base_cfg, demo_cfg = Path(args.config).name, cfg["busqueda"], cfg["modelo_base"], cfg["demo"]
    revisar([("semilla", cfg["semilla"], entero_no_negativo), ("n_semillas", cfg["n_semillas"], entero_positivo),
             ("busqueda.n_configuraciones", b["n_configuraciones"], entero_no_negativo),
             ("busqueda.epocas", b["epocas"], entero_positivo), ("busqueda.paciencia", b["paciencia"], entero_no_negativo),
             ("modelo_base.lr", base_cfg["lr"], no_negativo), ("modelo_base.weight_decay", base_cfg["weight_decay"], no_negativo),
             ("modelo_base.batch_size", base_cfg["batch_size"], tamano_lote),
             ("demo.muestra_entrenamiento", demo_cfg["muestra_entrenamiento"], entero_positivo),
             ("demo.n_configuraciones", demo_cfg["n_configuraciones"], entero_no_negativo),
             ("demo.n_semillas", demo_cfg["n_semillas"], entero_positivo), ("demo.epocas", demo_cfg["epocas"], entero_positivo)],
            nombre_cfg)
    revisar_espacio(b["espacio"], nombre_cfg)
    semilla = args.semilla if args.semilla is not None else cfg["semilla"]
    n_conf = cfg["busqueda"]["n_configuraciones"]
    n_sem = cfg["n_semillas"]
    epocas = cfg["busqueda"]["epocas"]
    if args.demo:
        n_conf, n_sem, epocas = cfg["demo"]["n_configuraciones"], cfg["demo"]["n_semillas"], cfg["demo"]["epocas"]
    n_conf = args.n_configuraciones if args.n_configuraciones is not None else n_conf
    n_sem = args.n_semillas if args.n_semillas is not None else n_sem
    epocas = args.epocas if args.epocas is not None else epocas

    salida = RAIZ / "resultados" / ("demo" if args.demo else "")
    salida.mkdir(parents=True, exist_ok=True)
    inicio_total = time.perf_counter()

    d = datos_parte1()
    X_train, y_train = d["X_train"], d["y_train"]
    if args.demo:
        if cfg["demo"]["muestra_entrenamiento"] > len(X_train):
            salir(f"demo.muestra_entrenamiento ({cfg['demo']['muestra_entrenamiento']}) es mayor que los créditos de entrenamiento ({len(X_train):,})")
        idx = np.random.default_rng(semilla).choice(len(X_train), cfg["demo"]["muestra_entrenamiento"], replace=False)
        X_train, y_train = X_train.iloc[idx], y_train[idx]
    scaler = StandardScaler().fit(X_train)  # ajustado solo con entrenamiento, como en el notebook
    X_train_t, X_val_t, X_test_t = (a_tensor(scaler.transform(X)) for X in (X_train, d["X_val"], d["X_test"]))
    y_train_t = a_tensor(y_train)
    pos_weight = calcular_pos_weight(y_train)
    print(f"Entrenamiento: {len(y_train):,} créditos | pos_weight = {pos_weight.item():.4f} | "
          f"configuraciones: {n_conf} + base | semillas: {n_sem} | máx. épocas: {epocas}")

    base = {**cfg["modelo_base"], "epocas": epocas, "paciencia": cfg["busqueda"]["paciencia"]}
    ruta_final = salida / "config_final_pd.json"

    # 1. Búsqueda aleatoria: la configuración base de Tomás es la candidata 0
    if args.solo_semillas:
        if not ruta_final.exists():
            salir(f"no existe {ruta_final.relative_to(RAIZ)}: corra primero la búsqueda (sin --solo_semillas)"
                  + (" con --demo" if args.demo else ""))
        final = json.loads(ruta_final.read_text())
        print(f"Configuración final leída de {ruta_final}")
    else:
        rng = np.random.default_rng(semilla)
        candidatas = [base] + [
            {"modelo": "LGDNN", **muestrear_config(rng, cfg["busqueda"]["espacio"]),
             "epocas": epocas, "paciencia": cfg["busqueda"]["paciencia"]}
            for _ in range(n_conf)
        ]
        filas, mejor_modelo, mejor_auc = [], None, -1.0
        for i, c in enumerate(candidatas):
            t = time.perf_counter()
            model, hist = entrenar_clasificador(c, X_train_t, y_train_t, X_val_t, d["y_val"], pos_weight, semilla=semilla)
            fila = {"candidata": i, "origen": "base (Tomás)" if i == 0 else "aleatoria", **describir(c),
                    "weight_decay": c["weight_decay"], "lr": c["lr"], "batch_size": c["batch_size"],
                    "parametros": contar_parametros(model), "epocas": len(hist["val_auc"]),
                    **evaluar(model, X_val_t, d["y_val"], pos_weight, "val"), "segundos": time.perf_counter() - t}
            filas.append(fila)
            print(f"{i:02d}/{n_conf} | AUC val: {fila['AUC_val']:.4f} | {fila['capas_ocultas']} {fila['activacion']} "
                  f"| lr={c['lr']} | {fila['segundos']:.1f} s")
            if fila["AUC_val"] > mejor_auc:
                mejor_auc, mejor_modelo, final = fila["AUC_val"], model, c
        busqueda = pd.DataFrame(filas).sort_values("AUC_val", ascending=False)
        busqueda.to_csv(salida / "busqueda_pd.csv", index=False)
        ruta_final.write_text(json.dumps(final, indent=2))
        torch.save({"state_dict": mejor_modelo.state_dict(), "config": final, "columnas": d["features"], "scaler": scaler},
                   RAIZ / "weights" / ("modelo_pd_busqueda_demo.pth" if args.demo else "modelo_pd_busqueda.pth"))
        print(f"\nBúsqueda: {len(filas)} configuraciones en {busqueda['segundos'].sum() / 60:.1f} min")
        print(busqueda.head(5).drop(columns=["segundos"]).round(4).to_string(index=False))

    # 2. Semillas de la configuración final y, si es distinta, también de la base
    configs_semillas = [("final", final)] + ([] if final == base else [("base (Tomás)", base)])
    filas = []
    for nombre, c in configs_semillas:
        for s in range(n_sem):
            t = time.perf_counter()
            model, hist = entrenar_clasificador(c, X_train_t, y_train_t, X_val_t, d["y_val"], pos_weight, semilla=s)
            filas.append({"configuracion": nombre, "semilla": s, "epocas": len(hist["val_auc"]),
                          **evaluar(model, X_val_t, d["y_val"], pos_weight, "val"),
                          **evaluar(model, X_test_t, d["y_test"], pos_weight, "test"),
                          "segundos": time.perf_counter() - t})
            print(f"[{nombre}] semilla {s:02d} | AUC val: {filas[-1]['AUC_val']:.4f} | AUC prueba: {filas[-1]['AUC_test']:.4f}")
    semillas = pd.DataFrame(filas)
    semillas.to_csv(salida / "semillas_pd.csv", index=False)

    for nombre, grupo in semillas.groupby("configuracion", sort=False):
        print(f"\nResumen de {len(grupo)} semillas — configuración {nombre} "
              f"({grupo['segundos'].sum() / 60:.1f} min):")
        columnas = [f"{m}_{k}" for k in ["val", "test"] for m in METRICAS]
        resumen = grupo[columnas].agg(["mean", "std", "max", "min"])
        resumen.index = ["media", "desv. estándar", "máximo", "mínimo"]
        print(resumen.round(4).to_string())
    print(f"\nTiempo total: {(time.perf_counter() - inicio_total) / 60:.1f} min. Resultados en {salida}")


if __name__ == "__main__":
    main()
