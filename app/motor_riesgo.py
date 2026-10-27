"""Motor de la aplicación: usa los modelos guardados del equipo, sin reentrenar.

- PD: red de notebooks/clasificacion.ipynb, con la corrección del peso de clase.
- Codificación de lo que ingresa el usuario: codificar() de src/preprocesamiento.py.
- Puntaje: CreditScorecard de scripts/scorecard_engine.py.
- Percentil: frente a las PD de los créditos de prueba.
- Puntos por variable: gradientes integrados frente al cliente promedio (scripts/parte1_evaluacion.py).
- LGD: red de notebooks/regresion.ipynb; EAD: factor por plazo de la Parte 2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "scripts"))
from src.modelos import (  # noqa: E402
    a_tensor, calcular_pos_weight, cargar_clasificador, cargar_factor_ead, cargar_modelo_lgd, datos_parte1,
    estimar_ead, predecir, predecir_proba,
)
from src.preprocesamiento import codificar  # noqa: E402
from parte1_evaluacion import gradientes_integrados, grupos_de_variables  # noqa: E402
from scorecard_engine import CreditScorecard  # noqa: E402

PASOS_IG = 256


class MotorRiesgo:
    def __init__(self):
        d = datos_parte1()
        self.red, self.scaler, self.columnas = cargar_clasificador()
        self.pos_weight = calcular_pos_weight(d["y_train"])
        self.ln_w = float(torch.log(self.pos_weight))
        # Odds de la cartera de entrenamiento (N0/N1): con ellas, el puntaje base es el cliente promedio
        self.odds_cartera = float((d["y_train"] == 0).sum() / (d["y_train"] == 1).sum())
        self.modelo_lgd, self.scaler_lgd, self.columnas_lgd, _ = cargar_modelo_lgd()
        self.factor_ead = cargar_factor_ead()
        self.opciones = json.loads((RAIZ / "app" / "opciones_app.json").read_text())
        self.grupos = grupos_de_variables(self.columnas)
        # PD de los créditos de prueba: la población contra la que se ubica cada solicitante
        self.pd_poblacion = np.sort(predecir_proba(self.red, a_tensor(self.scaler.transform(d["X_test"])), self.pos_weight))
        with torch.no_grad():
            self.logit_base = self.red(torch.zeros(1, len(self.columnas))).item()

    def cliente_por_defecto(self):
        cliente = {c: v["defecto"] for c, v in self.opciones["numericas"].items()}
        cliente |= {c: v["defecto"] for c, v in self.opciones["categoricas"].items()}
        return cliente

    def _x(self, cliente):
        """Codifica un solicitante con codificar() de src/preprocesamiento.py.

        Con una sola fila, get_dummies(drop_first=True) elimina la única categoría presente y todas las
        indicadoras quedan en 0: cualquier solicitante quedaría como plazo 36, vivienda ANY, propósito
        car y estado AK. Para evitarlo se agregan filas de relleno con todas las categorías (así
        drop_first elimina la misma categoría que en entrenamiento) y se conserva solo la primera fila.
        """
        cats = {c: v["opciones"] for c, v in self.opciones["categoricas"].items()}
        n = max(len(v) for v in cats.values())
        relleno = pd.DataFrame([cliente] * n)
        for c, valores in cats.items():
            relleno[c] = [valores[i % len(valores)] for i in range(n)]
        X = codificar(pd.concat([pd.DataFrame([cliente]), relleno], ignore_index=True), self.columnas)
        return X.iloc[[0]]

    def probabilidad(self, cliente):
        X = self._x(cliente)
        return float(predecir_proba(self.red, a_tensor(self.scaler.transform(X)), self.pos_weight)[0])

    def percentil(self, pd_cliente):
        """Porcentaje de la población con un puntaje menor o igual (es decir, con PD mayor o igual)."""
        return 100.0 * (1 - np.searchsorted(self.pd_poblacion, pd_cliente, side="left") / len(self.pd_poblacion))

    def evaluar(self, cliente, scorecard: CreditScorecard):
        X = self._x(cliente)
        Xt = a_tensor(self.scaler.transform(X))
        pd_cliente = float(predecir_proba(self.red, Xt, self.pos_weight)[0])
        lgd = float(predecir(self.modelo_lgd, a_tensor(self.scaler_lgd.transform(X[self.columnas_lgd])))[0])
        ead = float(estimar_ead([cliente["loan_amnt"]], X["term_ 60 months"].to_numpy(), self.factor_ead)[0])
        return {
            "pd": pd_cliente, "puntaje": int(scorecard.probabilidad_a_score(pd_cliente)),
            "percentil": self.percentil(pd_cliente), "lgd": lgd, "ead": ead, "perdida_esperada": pd_cliente * lgd * ead,
        }

    def puntos_por_variable(self, cliente, scorecard: CreditScorecard):
        """Puntos que suma o resta cada variable frente al cliente promedio (línea base x̃ = 0)."""
        Xt = a_tensor(self.scaler.transform(self._x(cliente)))
        ig = gradientes_integrados(self.red, Xt, PASOS_IG)[0]
        puntos = {g: -scorecard.factor * ig[[self.columnas.index(c) for c in cols]].sum() for g, cols in self.grupos.items()}
        base = scorecard.offset - scorecard.factor * (self.logit_base - self.ln_w)
        return base, pd.Series(puntos).sort_values(key=np.abs, ascending=False)

    def puntajes_poblacion(self, scorecard: CreditScorecard):
        return scorecard.probabilidad_a_score(self.pd_poblacion)
