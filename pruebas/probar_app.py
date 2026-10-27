"""Casos de prueba: ¿la aplicación (app/app_riesgo.py) usa los modelos entrenados por el equipo?

Uso:
    python pruebas/probar_app.py

No necesita internet. Compara lo que calcula el motor de la app (app/motor_riesgo.py) con los modelos
guardados por los notebooks:
  1. Los pesos que carga la app son idénticos a weights/modelo_clasificador.pth y weights/modelo_lgd.pth.
  2. Para créditos reales de prueba, la PD y la LGD de la app son las mismas que dan los modelos guardados.
  3. Sobre todo el conjunto de prueba, la PD de la app reproduce el AUC del notebook (0.7033), y la LGD,
     el MAE y el RMSE del notebook de regresión (0.0594 y 0.0802).
  4. Si se alteran los pesos de la red, el resultado de la app cambia: no hay valores escritos a mano.
  5. El puntaje de la app es el de la scorecard: 600 - 28.85 * logit de la red.
  6. Los cambios en una variable mueven el puntaje en la dirección esperada.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score, root_mean_squared_error

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "app"))
from motor_riesgo import MotorRiesgo  # noqa: E402
from scorecard_engine import CreditScorecard  # noqa: E402
from src.modelos import (  # noqa: E402
    RUTA_CLASIFICADOR, RUTA_LGD, a_tensor, cargar_clasificador, cargar_modelo_lgd, datos_parte1, predecir,
    predecir_proba,
)
from src.preprocesamiento import COLUMNAS_NUMERICAS, COLUMNAS_ONE_HOT, MAPEOS_ORDINALES, particion_temporal  # noqa: E402

resultados = []


def caso(nombre, ok, detalle):
    resultados.append(ok)
    print(f"[{'PASA' if ok else 'FALLA'}] {nombre}\n        {detalle}")


def a_solicitud(fila, opciones):
    """Reconstruye lo que una persona escribiría en la app a partir de una fila de los datos procesados."""
    cliente = {c: float(fila[c]) for c in COLUMNAS_NUMERICAS}
    for c, mapeo in MAPEOS_ORDINALES.items():
        inverso = {v: k for k, v in mapeo.items()}
        cliente[c] = inverso[int(fila[c])]
    for c in COLUMNAS_ONE_HOT:
        cats = opciones["categoricas"][c]["opciones"]
        activa = [k for k in cats if f"{c}_{k}" in fila.index and fila[f"{c}_{k}"] == 1]
        cliente[c] = activa[0] if activa else cats[0]  # sin indicadora activa: la categoría eliminada
    return cliente


def main():
    print("Cargando el motor de la app y los modelos guardados...\n")
    motor = MotorRiesgo()
    sc = CreditScorecard(score_base=600, odds_base=motor.odds_cartera, pdo=20)
    d = datos_parte1()
    red_archivo, scaler_archivo, columnas = cargar_clasificador()
    lgd_archivo, scaler_lgd, columnas_lgd, _ = cargar_modelo_lgd()

    # 1. Pesos idénticos a los archivos
    pesos_pd = torch.load(RUTA_CLASIFICADOR, map_location="cpu")
    pesos_lgd = torch.load(RUTA_LGD, map_location="cpu", weights_only=False)["state_dict"]
    iguales_pd = all(torch.equal(v, pesos_pd[k]) for k, v in motor.red.state_dict().items())
    iguales_lgd = all(torch.equal(v, pesos_lgd[k]) for k, v in motor.modelo_lgd.state_dict().items())
    caso("1. La app carga los pesos entrenados", iguales_pd and iguales_lgd,
         f"red de PD = weights/modelo_clasificador.pth: {iguales_pd} | red de LGD = weights/modelo_lgd.pth: {iguales_lgd}")

    # 2. Créditos reales de prueba, ingresados como lo haría una persona en la app
    test = d["df"][d["particion"] == "test"]
    muestra = test.sample(500, random_state=7)
    salidas = [motor.evaluar(a_solicitud(f, motor.opciones), sc) for _, f in muestra.iterrows()]
    pd_app, lgd_app = np.array([s["pd"] for s in salidas]), np.array([s["lgd"] for s in salidas])
    pw = motor.pos_weight
    pd_modelo = predecir_proba(red_archivo, a_tensor(scaler_archivo.transform(muestra[columnas])), pw)
    lgd_modelo = predecir(lgd_archivo, a_tensor(scaler_lgd.transform(muestra[columnas_lgd])))
    dif_pd, dif_lgd = np.abs(pd_app - pd_modelo).max(), np.abs(lgd_app - lgd_modelo).max()
    caso("2. En 500 créditos reales de prueba, la app da la misma PD y LGD que los modelos",
         dif_pd < 1e-5 and dif_lgd < 1e-5,
         f"diferencia máxima de PD: {dif_pd:.1e} | diferencia máxima de LGD: {dif_lgd:.1e}")

    # 3. Métricas del notebook con la PD y la LGD de la app
    pd_red_app = predecir_proba(motor.red, a_tensor(motor.scaler.transform(d["X_test"])), pw)
    auc_directo = roc_auc_score(d["y_test"], pd_red_app)
    poblacion_ok = np.allclose(np.sort(pd_red_app), motor.pd_poblacion)  # la población del percentil sale del modelo
    castigados = test[test["lgd"].notna()]
    lgd_pred = predecir(motor.modelo_lgd, a_tensor(motor.scaler_lgd.transform(castigados[motor.columnas_lgd])))
    mae, rmse = mean_absolute_error(castigados["lgd"], lgd_pred), root_mean_squared_error(castigados["lgd"], lgd_pred)
    caso("3. Con la red de la app se reproducen las métricas de los notebooks",
         round(auc_directo, 4) == 0.7033 and poblacion_ok and round(mae, 4) == 0.0594 and round(rmse, 4) == 0.0802,
         f"AUC en prueba: {auc_directo:.4f} (notebook 0.7033) | LGD: MAE {mae:.4f} (0.0594), RMSE {rmse:.4f} (0.0802) | "
         f"población del percentil = PD del modelo en prueba: {poblacion_ok}")

    # 4. Si se alteran los pesos, el resultado cambia
    cliente = motor.cliente_por_defecto()
    antes = motor.evaluar(cliente, sc)
    respaldo = {k: v.clone() for k, v in motor.red.state_dict().items()}
    with torch.no_grad():
        motor.red.network[-1].weight.mul_(0.5)
    despues = motor.evaluar(cliente, sc)
    motor.red.load_state_dict(respaldo)
    caso("4. Si se alteran los pesos de la red, el resultado de la app cambia",
         abs(antes["pd"] - despues["pd"]) > 1e-3,
         f"PD con los pesos entrenados: {antes['pd']:.4f} | con la última capa a la mitad: {despues['pd']:.4f}")

    # 5. El puntaje es el de la scorecard
    X = motor._x(cliente)
    with torch.no_grad():
        logit = motor.red(a_tensor(motor.scaler.transform(X))).item()
    esperado = 600 - (20 / np.log(2)) * logit
    caso("5. El puntaje de la app es 600 - 28.85 * logit de la red",
         abs(antes["puntaje"] - esperado) <= 0.5,
         f"puntaje de la app: {antes['puntaje']} | 600 - 28.85 * ({logit:.4f}) = {esperado:.2f}")

    # 6. Dirección de los cambios
    cambios = {"plazo de 60 meses": {"term": " 60 months"}, "subcalificación E3 y tasa de 20 %": {"sub_grade": "E3", "int_rate": 20.0},
               "relación deuda/ingreso de 30": {"dti": 30.0}, "propósito pequeño negocio": {"purpose": "small_business"}}
    filas = {k: motor.evaluar(cliente | v, sc)["puntaje"] for k, v in cambios.items()}
    caso("6. Los cambios que aumentan el riesgo bajan el puntaje",
         all(p < antes["puntaje"] for p in filas.values()),
         f"cliente típico: {antes['puntaje']} | " + " | ".join(f"{k}: {p}" for k, p in filas.items()))

    # 7. La interfaz no se cae con valores extremos
    from streamlit.testing.v1 import AppTest
    errores, pruebas = [], 0
    extremos = {"f_annual_inc": [0.0, 1e9], "f_loan_amnt": [0.0, 1e7], "f_dti": [0.0, 999.0], "f_int_rate": [0.0, 99.0],
                "f_revol_util": [0.0, 500.0], "f_tot_cur_bal": [0.0, 1e9], "f_revol_bal": [0.0, 1e8],
                "f_delinq_2yrs": [0, 50], "f_inq_last_6mths": [0, 50], "f_open_acc": [0, 200], "f_total_acc": [0, 500],
                "f_collections_12_mths_ex_med": [0, 20], "f_acc_now_delinq": [0, 20]}
    barra = {"sc_pdo": [5.0, 100.0], "sc_odds": [0.5, 500.0], "sc_base": [300, 900]}
    at = AppTest.from_file(str(RAIZ / "app" / "app_riesgo.py"), default_timeout=180).run()
    for clave, valores in list(extremos.items()) + list(barra.items()):
        for valor in valores:
            at.number_input(key=clave).set_value(valor).run(); pruebas += 1
            if at.exception:
                errores.append(f"{clave}={valor}: {at.exception[0].value}")
            at = AppTest.from_file(str(RAIZ / "app" / "app_riesgo.py"), default_timeout=180).run()
    variables = list(motor.opciones["numericas"]) + list(motor.opciones["categoricas"])  # valores internos de las opciones
    for opcion in variables:
        at.selectbox(key="wi_variable").set_value(opcion).run(); pruebas += 1
        if at.exception:
            errores.append(f"simular {opcion}: {at.exception[0].value}")
    for clave in ["f_term", "f_purpose", "f_home_ownership", "f_addr_state", "f_sub_grade", "f_verification_status"]:
        opciones = motor.opciones["categoricas"][clave[2:]]["opciones"]
        for opcion in opciones[:: max(1, len(opciones) // 4)]:
            at.selectbox(key=clave).set_value(opcion).run(); pruebas += 1
            if at.exception:
                errores.append(f"{clave}={opcion}: {at.exception[0].value}")
    caso("7. La interfaz no se cae con valores extremos ni con ninguna opción",
         not errores, f"{pruebas} interacciones probadas" + (f"; errores: {errores[:3]}" if errores else ", sin errores"))

    print(f"\n{sum(resultados)} de {len(resultados)} casos pasan.")
    sys.exit(0 if all(resultados) else 1)


if __name__ == "__main__":
    main()
