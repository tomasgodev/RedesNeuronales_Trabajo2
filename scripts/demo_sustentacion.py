"""Demostración en vivo para la sustentación (< 2 minutos): entrena la red de la Parte 1 con los
parámetros que se pidan, la compara con la regresión logística, calcula la scorecard y la pérdida esperada.

Uso:
    python scripts/demo_sustentacion.py
    python scripts/demo_sustentacion.py --lr 0.01 --dropout 0.5
    python scripts/demo_sustentacion.py --capas 128 64 --activacion tanh --epocas 10
    python scripts/demo_sustentacion.py --muestra 20000 --semilla 7
    python scripts/demo_sustentacion.py --pdo 40 --puntaje_base 700 --odds_base 20
    python scripts/demo_sustentacion.py --sin_peso_clase      # entrena sin pos_weight

Todo se calcula de verdad con los datos procesados. Sin --capas se usa la arquitectura de
notebooks/clasificacion.ipynb (CreditRiskNN); con --capas se usa la clase parametrizada LGDNN.
La LGD y la EAD vienen de los modelos guardados de la Parte 2.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.modelos import (  # noqa: E402
    a_tensor, calcular_pos_weight, cargar_factor_ead, cargar_modelo_lgd, contar_parametros, datos_parte1,
    entrenar_clasificador, estimar_ead, predecir, predecir_proba,
)
from metricas_clasificacion import calcular_metricas_completas  # noqa: E402
from scorecard_engine import CreditScorecard  # noqa: E402


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("red neuronal")
    g.add_argument("--capas", type=int, nargs="+", help="neuronas por capa oculta, p. ej. --capas 128 64")
    g.add_argument("--activacion", choices=["relu", "tanh", "leaky_relu"], help="por defecto relu")
    g.add_argument("--dropout", type=float, help="por defecto 0.2")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-4, help="penalización L2")
    g.add_argument("--batch_size", type=int, default=512)
    g.add_argument("--epocas", type=int, default=15)
    g.add_argument("--paciencia", type=int, default=10)
    g.add_argument("--semilla", type=int, default=56)
    g.add_argument("--muestra", type=int, help="créditos de entrenamiento a usar (por defecto, todos)")
    g.add_argument("--sin_peso_clase", action="store_true", help="entrena sin pos_weight")
    s = p.add_argument_group("scorecard")
    s.add_argument("--puntaje_base", type=float, default=600)
    s.add_argument("--odds_base", type=float, help="por defecto, las odds de la cartera de entrenamiento (N0/N1)")
    s.add_argument("--pdo", type=float, default=20)
    return p.parse_args()


def linea(texto=""):
    print(texto)


def main():
    args = leer_argumentos()
    t0 = time.perf_counter()
    linea("=" * 72)
    linea("TRABAJO 2 · ¿ME PRESTAS? · DEMOSTRACIÓN CON DATOS Y MODELOS REALES")
    linea("=" * 72)

    d = datos_parte1()
    X_train, y_train = d["X_train"], d["y_train"]
    if args.muestra:
        idx = np.random.default_rng(args.semilla).choice(len(X_train), args.muestra, replace=False)
        X_train, y_train = X_train.iloc[idx], y_train[idx]
    scaler = StandardScaler().fit(X_train)
    X_tr, X_va, X_te = (a_tensor(scaler.transform(X)) for X in (X_train, d["X_val"], d["X_test"]))
    pos_weight = calcular_pos_weight(y_train)
    w_entrenamiento = a_tensor([1.0]) if args.sin_peso_clase else pos_weight
    linea(f"Datos: entrenamiento {len(y_train):,} | validación {len(d['y_val']):,} | prueba {len(d['y_test']):,} "
          f"| 82 variables | tasa de incumplimiento en entrenamiento {y_train.mean():.1%}")

    if args.capas or args.activacion or args.dropout is not None:
        capas = args.capas or [64, 32, 16]
        activacion = args.activacion or "relu"
        dropout = 0.2 if args.dropout is None else args.dropout
        config = {"modelo": "LGDNN", "capas_ocultas": capas, "activacion": activacion, "dropout": dropout}
        arquitectura = f"82 → {' → '.join(map(str, capas))} → 1, {activacion}, dropout {dropout} en todas las capas"
    else:
        config = {"modelo": "CreditRiskNN"}
        arquitectura = "82 → 64 → 32 → 16 → 1, relu, dropout 0.2 (arquitectura de clasificacion.ipynb)"
    config |= {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
               "epocas": args.epocas, "paciencia": args.paciencia}
    linea(f"\n[1/4] Entrenando la red: {arquitectura}")
    linea(f"      lr {args.lr} | weight_decay {args.weight_decay} | lote {args.batch_size} | máx. {args.epocas} épocas "
          f"| semilla {args.semilla} | pos_weight {'1 (sin peso)' if args.sin_peso_clase else f'{pos_weight.item():.2f}'}")
    t = time.perf_counter()
    red, hist = entrenar_clasificador(config, X_tr, a_tensor(y_train), X_va, d["y_val"], w_entrenamiento,
                                      semilla=args.semilla, verbose=True)
    linea(f"      {contar_parametros(red):,} parámetros | {len(hist['val_auc'])} épocas | {time.perf_counter() - t:.1f} s")

    rl = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42).fit(scaler.transform(X_train), y_train)
    ln_w = float(torch.log(pos_weight))

    linea("\n[2/4] Métricas (probabilidades corregidas por el peso de clase)")
    linea(f"      {'Modelo':<22}{'Conjunto':<12}{'AUC':>8}{'Gini':>8}{'KS':>8}{'Brier':>8}{'Brier ref.':>12}")
    probs = {}
    for k, X_t, X_df in [("validación", X_va, d["X_val"]), ("prueba", X_te, d["X_test"])]:
        y = d["y_val"] if k == "validación" else d["y_test"]
        probs[("red", k)] = predecir_proba(red, X_t, w_entrenamiento)
        logit_rl = rl.decision_function(scaler.transform(X_df)) - ln_w
        probs[("rl", k)] = 1 / (1 + np.exp(-logit_rl))
        for nombre, clave in [("Red neuronal", "red"), ("Regresión logística", "rl")]:
            m = calcular_metricas_completas(y, probs[(clave, k)], k)
            linea(f"      {nombre:<22}{k:<12}{m['AUC']:>8.4f}{m['Gini']:>8.4f}{m['KS']:>8.4f}{m['Brier']:>8.4f}{m['Brier_Referencia']:>12.4f}")

    odds_base = args.odds_base if args.odds_base is not None else float((y_train == 0).sum() / (y_train == 1).sum())
    sc = CreditScorecard(score_base=args.puntaje_base, odds_base=odds_base, pdo=args.pdo)
    puntajes = sc.probabilidad_a_score(probs[("red", "prueba")])
    linea(f"\n[3/4] Scorecard: base {args.puntaje_base:g} | odds {odds_base:.3f}:1 | PDO {args.pdo:g} "
          f"→ Factor {sc.factor:.4f} | Offset {sc.offset:.4f}")
    linea(f"      Puntaje = {sc.offset:.2f} − {sc.factor:.2f} · ln(p / (1 − p))")
    linea(f"      Prueba: media {puntajes.mean():.1f} | desv. {puntajes.std():.1f} | percentiles 5/50/95: "
          f"{np.percentile(puntajes, 5):.0f} / {np.percentile(puntajes, 50):.0f} / {np.percentile(puntajes, 95):.0f}")
    for p in [0.05, 0.10, 0.25, 0.50]:
        linea(f"      PD {p:>4.0%} → {sc.probabilidad_a_score(p)} puntos")

    test = d["df"][d["particion"] == "test"]
    modelo_lgd, scaler_lgd, columnas_lgd, _ = cargar_modelo_lgd()
    lgd = predecir(modelo_lgd, a_tensor(scaler_lgd.transform(test[columnas_lgd])))
    ead = estimar_ead(test["loan_amnt"], test["term_ 60 months"], cargar_factor_ead())
    el = probs[("red", "prueba")] * lgd * ead
    linea("\n[4/4] Pérdida esperada en prueba, E[L] = PD · LGD · EAD")
    linea(f"      PD media {probs[('red', 'prueba')].mean():.1%} (observada {d['y_test'].mean():.1%}) | "
          f"LGD media {lgd.mean():.1%} | EAD media {ead.mean():,.0f} USD")
    linea(f"      E[L] total {el.sum():,.0f} USD | por crédito {el.mean():,.0f} USD | "
          f"{el.sum() / test['loan_amnt'].sum():.1%} del monto prestado")

    linea("\n" + "=" * 72)
    linea(f"Tiempo total: {time.perf_counter() - t0:.1f} s")
    linea("=" * 72)


if __name__ == "__main__":
    main()
