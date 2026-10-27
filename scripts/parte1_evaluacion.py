"""Parte 1: evaluación de los dos modelos, scorecard e interpretabilidad, con un solo comando.

Uso:
    python scripts/parte1_evaluacion.py
    python scripts/parte1_evaluacion.py --puntaje_base 650 --odds_base 30 --pdo 25
    python scripts/parte1_evaluacion.py --rapido      # menos repeticiones y muestra para la sustentación

Qué hace:
1. Reentrena la regresión logística con la misma configuración de notebooks/regresion_logistica.ipynb
   (determinista) y la guarda en weights/regresion_logistica_pd.joblib.
2. Carga la red guardada por notebooks/clasificacion.ipynb (no la reentrena).
3. Evalúa ambos modelos en validación y prueba con calcular_metricas_completas
   (scripts/metricas_clasificacion.py), después de corregir el efecto del peso de clase.
4. Calcula los puntajes con CreditScorecard (scripts/scorecard_engine.py).
5. Interpretabilidad: importancia por permutación por grupo de variables (Breiman, 2001;
   Fisher et al., 2019) y gradientes integrados para repartir el puntaje de la red por variable
   (Sundararajan et al., 2017).

Salidas: resultados/parte1/*.csv y *.json; figuras en blog/figuras/.
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
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
from src.modelos import (  # noqa: E402
    a_tensor, calcular_pos_weight, cargar_clasificador, datos_parte1, predecir_proba,
)
from metricas_clasificacion import calcular_metricas_completas, tabla_calibracion_decilica  # noqa: E402
from scorecard_engine import CreditScorecard  # noqa: E402

SALIDA = RAIZ / "resultados" / "parte1"
FIGURAS = RAIZ / "blog" / "figuras"
COLORES = {"Red neuronal": "#e8833a", "Regresión logística": "#2a6fdb"}
# Variables originales y prefijos de sus indicadoras (src/preprocesamiento.py)
GRUPOS_ONE_HOT = ["term", "home_ownership", "purpose", "addr_state"]


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(RAIZ / "configs" / "parte1.json"))
    p.add_argument("--puntaje_base", type=float)
    p.add_argument("--odds_base", type=float)
    p.add_argument("--pdo", type=float)
    p.add_argument("--rapido", action="store_true", help="menos repeticiones de permutación y muestra en gradientes integrados")
    return p.parse_args()


def grupos_de_variables(features):
    """Agrupa las columnas por variable original: las indicadoras de una misma variable van juntas."""
    grupos = {}
    for col in features:
        base = next((g for g in GRUPOS_ONE_HOT if col.startswith(g + "_")), col)
        grupos.setdefault(base, []).append(col)
    return grupos


def importancia_permutacion(predecir, X, y, grupos, repeticiones, semilla):
    """Caída del AUC al permutar conjuntamente las columnas de cada grupo (misma permutación de filas)."""
    rng = np.random.default_rng(semilla)
    auc_base = roc_auc_score(y, predecir(X))
    filas = []
    for nombre, cols in grupos.items():
        idx = [X.columns.get_loc(c) for c in cols]
        caidas = []
        for _ in range(repeticiones):
            Xp = X.to_numpy().copy()
            Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
            caidas.append(auc_base - roc_auc_score(y, predecir(pd.DataFrame(Xp, columns=X.columns))))
        filas.append({"grupo": nombre, "columnas": len(cols), "caida_auc": np.mean(caidas), "desv": np.std(caidas)})
    return pd.DataFrame(filas).sort_values("caida_auc", ascending=False), auc_base


def gradientes_integrados(model, X_t, pasos, lote=2048):
    """Atribución del logit respecto a la línea base x̃ = 0 (el promedio de entrenamiento).

    Por la propiedad de completitud, la suma de las atribuciones de un crédito es f(x) - f(0).
    La integral se aproxima con la regla del punto medio.
    """
    model.eval()
    alphas = ((torch.arange(pasos, dtype=torch.float32) + 0.5) / pasos).to(X_t.device)
    atribuciones = []
    for inicio in range(0, len(X_t), lote):
        x = X_t[inicio:inicio + lote]
        gradiente = torch.zeros_like(x)
        for a in alphas:
            xa = (a * x).requires_grad_(True)
            model(xa).sum().backward()
            gradiente += xa.grad
        atribuciones.append((x * gradiente / pasos).detach())
    return torch.cat(atribuciones).cpu().numpy()


def guardar(fig, nombre):
    fig.savefig(FIGURAS / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {(FIGURAS / nombre).relative_to(RAIZ)}")


def main():
    global SALIDA, FIGURAS
    args = leer_argumentos()
    if args.rapido:  # no sobrescribe los resultados completos ni las figuras del blog
        SALIDA = RAIZ / "resultados" / "demo" / "parte1"
        FIGURAS = RAIZ / "resultados" / "demo" / "figuras"
    cfg = json.loads(Path(args.config).read_text())
    sc_cfg = cfg["scorecard"]
    puntaje_base = args.puntaje_base if args.puntaje_base is not None else sc_cfg["puntaje_base"]
    odds_base = args.odds_base if args.odds_base is not None else sc_cfg["odds_base"]
    pdo = args.pdo if args.pdo is not None else sc_cfg["pdo"]
    repeticiones = 2 if args.rapido else cfg["interpretabilidad"]["repeticiones_permutacion"]
    pasos = cfg["interpretabilidad"]["pasos_gradientes_integrados"]
    SALIDA.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ---------------------------------------------------------------- datos y modelos
    d = datos_parte1()
    features = d["features"]
    red, scaler_red, columnas_red = cargar_clasificador()
    assert columnas_red == features, "Las columnas del modelo guardado no coinciden con los datos procesados"
    pos_weight = calcular_pos_weight(d["y_train"])
    ln_w = float(torch.log(pos_weight))
    print(f"pos_weight = {pos_weight.item():.4f} (ln = {ln_w:.4f})")

    # Regresión logística: misma configuración de notebooks/regresion_logistica.ipynb (celdas 6 y 7)
    t = time.perf_counter()
    scaler_rl = StandardScaler().fit(d["X_train"])
    rl = LogisticRegression(**cfg["regresion_logistica"]).fit(scaler_rl.transform(d["X_train"]), d["y_train"])
    segundos_rl = time.perf_counter() - t
    joblib.dump({"modelo": rl, "scaler": scaler_rl, "columnas": features, "ln_peso_clase": ln_w},
                RAIZ / "weights" / "regresion_logistica_pd.joblib")
    print(f"Regresión logística ajustada en {segundos_rl:.1f} s ({rl.coef_.size + 1} parámetros)")

    # class_weight='balanced' pesa los incumplimientos N0/N1 veces más que los buenos: el mismo
    # desplazamiento del logit que pos_weight, así que se corrige igual (blog, sección 7.3)
    def proba_rl(X, corregir=True):
        logit = rl.decision_function(scaler_rl.transform(X))
        return 1 / (1 + np.exp(-(logit - (ln_w if corregir else 0.0))))

    def proba_red(X):
        return predecir_proba(red, a_tensor(scaler_red.transform(X)), pos_weight)

    modelos = {"Regresión logística": proba_rl, "Red neuronal": proba_red}
    prob = {(m, k): f(d[f"X_{k}"]) for m, f in modelos.items() for k in ["val", "test"]}

    # ---------------------------------------------------------------- métricas
    filas = []
    for (m, k), p in prob.items():
        r = calcular_metricas_completas(d[f"y_{k}"], p, k)
        filas.append({"modelo": m, "conjunto": k, **{c: r[c] for c in ["N", "Tasa Default", "AUC", "Gini", "KS", "Brier", "Brier_Referencia"]}})
    for k in ["val", "test"]:  # la logística sin corregir, para mostrar el efecto de class_weight
        r = calcular_metricas_completas(d[f"y_{k}"], proba_rl(d[f"X_{k}"], corregir=False), k)
        filas.append({"modelo": "Regresión logística sin corregir", "conjunto": k,
                      **{c: r[c] for c in ["N", "Tasa Default", "AUC", "Gini", "KS", "Brier", "Brier_Referencia"]}})
    metricas = pd.DataFrame(filas)
    metricas.to_csv(SALIDA / "metricas_pd.csv", index=False)
    print("\nMétricas de la Parte 1:")
    print(metricas.to_string(index=False))

    for m in modelos:
        tabla_calibracion_decilica(d["y_test"], prob[(m, "test")]).to_csv(
            SALIDA / f"calibracion_prueba_{'red' if m == 'Red neuronal' else 'logistica'}.csv", index=False)

    # ---------------------------------------------------------------- figuras de evaluación
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, k, titulo in [(axes[0], "val", "Validación"), (axes[1], "test", "Prueba")]:
        for m in modelos:
            frac, media = calibration_curve(d[f"y_{k}"], prob[(m, k)], n_bins=10, strategy="quantile")
            ax.plot(media, frac, "o-", color=COLORES[m], label=m)
        frac, media = calibration_curve(d[f"y_{k}"], proba_rl(d[f"X_{k}"], corregir=False), n_bins=10, strategy="quantile")
        ax.plot(media, frac, "o--", color=COLORES["Regresión logística"], alpha=0.4, label="Regresión logística sin corregir")
        ax.plot([0, 1], [0, 1], "k:", label="Calibración perfecta")
        ax.set(xlim=(0, 0.9), ylim=(0, 0.9), xlabel="Probabilidad predicha (promedio del decil)",
               ylabel="Tasa observada de incumplimiento", title=titulo)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    guardar(fig, "fig09_calibracion_modelos.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in modelos:
        fpr, tpr, _ = roc_curve(d["y_test"], prob[(m, "test")])
        ax.plot(fpr, tpr, color=COLORES[m], label=f"{m} (AUC = {roc_auc_score(d['y_test'], prob[(m, 'test')]):.4f})")
    ax.plot([0, 1], [0, 1], "k:", label="Al azar")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC en prueba")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    guardar(fig, "fig10_roc_prueba.png")

    p, y = prob[("Red neuronal", "test")], d["y_test"]
    s = np.sort(np.unique(p))
    F1 = np.searchsorted(np.sort(p[y == 1]), s, side="right") / (y == 1).sum()
    F0 = np.searchsorted(np.sort(p[y == 0]), s, side="right") / (y == 0).sum()
    i_ks = np.argmax(np.abs(F0 - F1))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(s, F0, color="#2a9d8f", label="Buenos pagadores ($F_0$)")
    ax.plot(s, F1, color="#c0392b", label="Malos pagadores ($F_1$)")
    ax.vlines(s[i_ks], F1[i_ks], F0[i_ks], color="k", linestyle="--", label=f"KS = {abs(F0[i_ks] - F1[i_ks]):.4f}")
    ax.set(xlabel="Probabilidad de incumplimiento predicha", ylabel="Proporción acumulada",
           title="Estadístico KS de la red neuronal en prueba")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    guardar(fig, "fig11_ks_red_prueba.png")

    # ---------------------------------------------------------------- scorecard
    sc = CreditScorecard(score_base=puntaje_base, odds_base=odds_base, pdo=pdo)
    resumen_sc = sc.obtener_resumen_calibracion()
    print(f"\nScorecard: base {puntaje_base:g}, odds {odds_base:g}:1, PDO {pdo:g} -> "
          f"Factor {resumen_sc['factor']}, Offset {resumen_sc['offset']}")
    puntajes = {m: sc.probabilidad_a_score(prob[(m, "test")]) for m in modelos}
    pts = puntajes["Red neuronal"]

    dist = pd.Series(pts).value_counts().sort_index().rename_axis("puntaje").reset_index(name="creditos")
    dist["proporcion_acumulada"] = dist["creditos"].cumsum() / dist["creditos"].sum()
    dist.to_csv(SALIDA / "distribucion_puntajes_prueba.csv", index=False)

    deciles = pd.DataFrame({"puntaje": pts, "pd": prob[("Red neuronal", "test")], "incumple": d["y_test"]})
    deciles["decil"] = pd.qcut(deciles["puntaje"].rank(method="first"), 10, labels=range(1, 11))
    tabla_deciles = deciles.groupby("decil", observed=True).agg(
        puntaje_min=("puntaje", "min"), puntaje_max=("puntaje", "max"), creditos=("puntaje", "size"),
        pd_media=("pd", "mean"), tasa_observada=("incumple", "mean")).reset_index()
    tabla_deciles.to_csv(SALIDA / "deciles_puntaje_prueba.csv", index=False)
    print("\nDeciles de puntaje en prueba (red neuronal):")
    print(tabla_deciles.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(pts.min() - 0.5, pts.max() + 1.5, 4)
    axes[0].hist(pts[d["y_test"] == 0], bins=bins, alpha=0.6, color="#2a9d8f", label="Buenos pagadores")
    axes[0].hist(pts[d["y_test"] == 1], bins=bins, alpha=0.6, color="#c0392b", label="Malos pagadores")
    axes[0].axvline(puntaje_base, color="k", linestyle="--", linewidth=1, label=f"Puntaje base ({puntaje_base:g})")
    axes[0].set(xlabel="Puntaje", ylabel="Créditos", title="Distribución de puntajes en prueba")
    axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(1, 11)
    axes[1].bar(x, tabla_deciles["tasa_observada"], color="#c0392b", alpha=0.75, label="Tasa observada")
    axes[1].plot(x, tabla_deciles["pd_media"], "ko-", label="PD media predicha")
    axes[1].set(xticks=x, xlabel="Decil de puntaje (1 = menor puntaje)", ylabel="Incumplimiento",
                title="Incumplimiento por decil de puntaje en prueba")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    guardar(fig, "fig12_puntajes_prueba.png")

    # Scorecard continua de la logística: con variables estandarizadas el logit es una suma, así que
    # Puntaje = [Offset - Factor (b0 - ln w)] - sum_j Factor * b_j * (x_j - media_j) / desv_j
    puntos_rl = pd.DataFrame({
        "variable": features, "coef_estandarizado": rl.coef_[0],
        "media_entrenamiento": scaler_rl.mean_, "desv_entrenamiento": scaler_rl.scale_,
    })
    puntos_rl["puntos_por_unidad"] = -sc.factor * puntos_rl["coef_estandarizado"] / puntos_rl["desv_entrenamiento"]
    puntos_rl["puntos_por_desviacion"] = -sc.factor * puntos_rl["coef_estandarizado"]
    puntos_rl = puntos_rl.reindex(puntos_rl["puntos_por_desviacion"].abs().sort_values(ascending=False).index)
    puntos_rl.to_csv(SALIDA / "scorecard_logistica_puntos.csv", index=False)
    base_rl = sc.offset - sc.factor * (rl.intercept_[0] - ln_w)
    # Verificación: la suma de puntos reproduce el puntaje calculado desde la probabilidad
    Xs = scaler_rl.transform(d["X_test"])
    suma = base_rl - sc.factor * (Xs @ rl.coef_[0])
    directo = sc.offset - sc.factor * np.log(prob[("Regresión logística", "test")] / (1 - prob[("Regresión logística", "test")]))
    print(f"\nScorecard de la logística: puntaje del cliente promedio = {base_rl:.2f}; "
          f"diferencia máxima suma de puntos vs. fórmula = {np.abs(suma - directo).max():.2e}")

    # ---------------------------------------------------------------- interpretabilidad
    grupos = grupos_de_variables(features)
    grupos_perm = {**grupos, "int_rate + sub_grade": ["int_rate", "sub_grade"]}
    imp = []
    for m, f in modelos.items():
        tabla, auc_base = importancia_permutacion(f, d["X_test"], d["y_test"], grupos_perm, repeticiones, cfg["semilla"])
        imp.append(tabla.assign(modelo=m, auc_prueba=auc_base))
    imp = pd.concat(imp)
    imp.to_csv(SALIDA / "importancia_permutacion_prueba.csv", index=False)
    print(f"\nImportancia por permutación (caída del AUC en prueba, {repeticiones} repeticiones):")
    print(imp.pivot(index="grupo", columns="modelo", values="caida_auc").sort_values("Red neuronal", ascending=False).round(4).to_string())

    n_ig = 1000 if args.rapido else 5000
    idx = np.random.default_rng(cfg["semilla"]).choice(len(d["X_test"]), n_ig, replace=False)
    X_ig = a_tensor(scaler_red.transform(d["X_test"].iloc[idx]))
    ig = gradientes_integrados(red, X_ig, pasos)
    with torch.no_grad():
        f_x = red(X_ig).cpu().numpy()
        f_0 = red(torch.zeros(1, X_ig.shape[1], device=X_ig.device)).item()
    error_completitud = np.abs(ig.sum(axis=1) - (f_x - f_0))
    base_red = sc.offset - sc.factor * (f_0 - ln_w)
    print(f"\nGradientes integrados ({n_ig} créditos de prueba, {pasos} pasos): logit de la línea base = {f_0:.4f}, "
          f"puntaje de la línea base = {base_red:.2f}; error de completitud medio = {error_completitud.mean():.4f}, "
          f"máximo = {error_completitud.max():.4f}")
    puntos_red = pd.DataFrame({g: -sc.factor * ig[:, [features.index(c) for c in cols]].sum(axis=1)
                               for g, cols in grupos.items()})
    imp_ig = puntos_red.abs().mean().sort_values(ascending=False).rename_axis("grupo").reset_index(name="puntos_abs_medios")
    imp_ig.to_csv(SALIDA / "importancia_gradientes_integrados.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"wspace": 0.55})
    orden = imp[imp["modelo"] == "Red neuronal"].set_index("grupo")["caida_auc"].sort_values().index
    y_pos = np.arange(len(orden))
    for j, m in enumerate(modelos):
        t = imp[imp["modelo"] == m].set_index("grupo").loc[orden]
        axes[0].barh(y_pos + (j - 0.5) * 0.4, t["caida_auc"], height=0.4, xerr=t["desv"], color=COLORES[m], label=m)
    axes[0].set(yticks=y_pos, yticklabels=orden, xlabel="Caída del AUC en prueba al permutar",
                title="Importancia por permutación")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    t = imp_ig.set_index("grupo")["puntos_abs_medios"].sort_values()
    axes[1].barh(t.index, t.values, color=COLORES["Red neuronal"])
    axes[1].set(xlabel="Puntos absolutos promedio por crédito", title="Gradientes integrados (red neuronal)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    guardar(fig, "fig13_importancia_variables.png")

    # Ejemplo: el crédito de la muestra con la PD mediana
    p_ig = prob[("Red neuronal", "test")][idx]
    k = int(np.argsort(p_ig)[len(p_ig) // 2])
    ejemplo = puntos_red.iloc[k].sort_values(key=np.abs, ascending=False)
    top = ejemplo.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color=["#2a9d8f" if v > 0 else "#c0392b" for v in top.values])
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set(xlabel="Puntos frente al cliente promedio",
           title=f"Puntos por variable de un crédito de prueba\n(PD = {p_ig[k]:.1%}, puntaje = {sc.probabilidad_a_score(p_ig[k])}, "
                 f"línea base = {base_red:.0f})")
    ax.grid(axis="x", alpha=0.3)
    guardar(fig, "fig14_puntos_ejemplo.png")

    # ---------------------------------------------------------------- resumen
    resumen = {
        "pos_weight": round(pos_weight.item(), 4), "ln_pos_weight": round(ln_w, 4),
        "segundos_regresion_logistica": round(segundos_rl, 2),
        "parametros_regresion_logistica": int(rl.coef_.size + 1),
        "scorecard": {**{k: resumen_sc[k] for k in ["score_base", "odds_base", "pdo", "factor", "offset"]},
                      "puntaje_cliente_promedio_logistica": round(base_rl, 2),
                      "puntaje_linea_base_red": round(base_red, 2)},
        "puntajes_prueba_red": {"media": round(float(pts.mean()), 1), "desv": round(float(pts.std()), 1),
                                "min": int(pts.min()), "max": int(pts.max()),
                                **{f"p{q}": float(np.percentile(pts, q)) for q in [1, 5, 25, 50, 75, 95, 99]}},
        "gradientes_integrados": {"creditos": n_ig, "pasos": pasos, "logit_linea_base": round(f_0, 4),
                                  "error_completitud_medio": round(float(error_completitud.mean()), 5),
                                  "error_completitud_max": round(float(error_completitud.max()), 5)},
        "segundos_total": round(time.perf_counter() - t0, 1),
    }
    (SALIDA / "resumen_parte1.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nResumen guardado en resultados/parte1/resumen_parte1.json | tiempo total: {resumen['segundos_total']} s")


if __name__ == "__main__":
    main()
