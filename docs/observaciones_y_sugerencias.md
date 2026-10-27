# Observaciones y Sugerencias Técnicas para el Equipo
**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02  
**Trabajo 2:** Redes neuronales para datos tabulares (¿me prestas?)  
**Profesor:** Juan David Ospina Arango | **Monitor:** Andrés Mauricio Zapata Rincón  
**Universidad Nacional de Colombia — Facultad de Minas**

---

En cumplimiento de las normas de trabajo colaborativo y para evitar cualquier conflicto en Git con los archivos sobre los que los compañeros se encuentran trabajando, **no se ha modificado ningún archivo preexistente**. En su lugar, se registran aquí las observaciones detalladas, la ubicación exacta y las sugerencias técnicas para consideración y revisión del equipo antes de la sustentación oral.

---

### Observación 1: Calibración de Probabilidades en el `classification_report` de Clasificación

* **Archivo y ubicación:** `notebooks/clasificacion.ipynb`, celdas 48 a 50.
* **Descripción del hallazgo:**  
  En la celda 45, se implementó de forma matemáticamente impecable la función de desesgo para compensar el factor de ponderación de clases `pos_weight = 3.39`:
  ```python
  def predecir_proba(model, X_t, pos_weight):
      model.eval()
      with torch.no_grad():
          logits = model(X_t) - torch.log(pos_weight)   # deshace el efecto de pos_weight
          return torch.sigmoid(logits).cpu().numpy()
  ```
  Sin embargo, en la celda 48, al calcular las predicciones para el `classification_report`, se omitió esta corrección:
  ```python
  # Código actual en celda 48:
  logits = model(X_val_t)
  y_prob_val = torch.sigmoid(logits).cpu().numpy().flatten()
  y_pred = (y_prob_val > best_threshold).astype(int)
  ```
* **Impacto en los resultados:**  
  Al no restar $\ln(\text{pos\_weight}) \approx 1.221$ del logit, las probabilidades predichas quedan infladas artificialmente hacia 1. Como consecuencia, el reporte de clasificación muestra un *recall* distorsionado de $0.99$ con una precisión de solo $0.27$, reflejando una tasa de falsos positivos excesivamente alta.
* **Corrección sugerida para el notebook:**  
  Reemplazar la llamada directa en la celda 48 por el vector desesgado ya calculado en la celda 45:
  ```python
  # Sugerencia:
  y_prob_val = p_val.flatten()  # p_val ya tiene el desesgo de pos_weight aplicado
  y_pred = (y_prob_val > best_threshold).astype(int)
  ```

---

### Observación 2: Estandarización y Serialización del `StandardScaler` de Clasificación

* **Archivo y ubicación:** `notebooks/clasificacion.ipynb`, celdas 37 y 51.
* **Descripción del hallazgo:**  
  El objeto `StandardScaler` se ajusta correctamente solo con los datos de entrenamiento (`scaler.fit_transform(X_train)`), y los pesos del modelo se guardan en `../weights/modelo_clasificador.pth`. No obstante, la línea de serialización del escalador (`pickle.dump(scaler, open("scaler.pkl", "wb"))`) quedó comentada en la celda 51.
* **Impacto en los resultados:**  
  Para que un usuario externo, un script CLI (`run_demo.py`) o la aplicación web (`app/app.py`) puedan realizar inferencia en tiempo real sobre nuevas observaciones ingresadas por un cliente sin depender de reprocesar el dataset de 40 MB, se requiere tener guardado el objeto `scaler.pkl` o sus medias y varianzas de entrenamiento.
* **Corrección sugerida para el notebook:**  
  Descomentar el guardado del escalador en la carpeta `weights/`:
  ```python
  import joblib
  joblib.dump(scaler, "../weights/scaler_clasificador.joblib")
  ```

---

### Observación 3: Aislamiento de la Partición Temporal Fuera de Tiempo (OOT)

* **Archivo y ubicación:** `notebooks/clasificacion.ipynb` vs `notebooks/entrenamiento_validacion.ipynb`.
* **Descripción del hallazgo:**  
  En `notebooks/clasificacion.ipynb` (celda 37), el equipo implementó con gran acierto una partición temporal pura en 3 bloques cronológicos sobre `issue_d`:
  - **Entrenamiento (60%):** 2012-08 a 2014-04 (115,830 créditos, $PD = 21.6\%$).
  - **Validación (20%):** 2014-04 a 2014-10 (38,610 créditos, $PD = 26.3\%$).
  - **Prueba OOT (20%):** 2014-10 a 2015-12 (38,610 créditos, $PD = 25.3\%$).
  Sin embargo, en `notebooks/entrenamiento_validacion.ipynb` (que parece ser un borrador previo), se realizaba un `train_test_split` aleatorio estratificado al 80/20 con `random_state=56`.
* **Sugerencia:**  
  Para el reporte técnico y la defensa oral frente al profesor, defender categóricamente la **partición temporal de `clasificacion.ipynb`**, ya que en riesgo de crédito bancario el estándar regulatorio y técnico exige evaluar *Out-of-Time* (créditos originados en el futuro con respecto a la ventana de entrenamiento), lo que demuestra solidez frente a cambios macroeconómicos y maduración de cartera.

---

### Observación 4: Consistencia de la Definición de EAD para Pérdida Esperada ($E[L]$)

* **Archivo y ubicación:** `docs/parte2_lgd.md`, Sección 8.
* **Descripción del hallazgo:**  
  En la Sección 8 de `docs/parte2_lgd.md` se define la exposición estimada al originar como $\widehat{EAD}_i = \text{funded\_amnt}_i \cdot \bar{f}_{\text{plazo}}$, donde $\bar{f}_{36} = 0.670$ y $\bar{f}_{60} = 0.846$.
* **Sugerencia técnica:**  
  En la sustentación, destacar ante el profesor la distinción conceptual entre:
  1. **EAD contable expost (en castigo):** $\text{funded\_amnt} - \text{total\_rec\_prncp}$, utilizada retrospectivamente para calcular la LGD observada de la muestra.
  2. **EAD prospectiva en originación ($t_0$):** Antes del desembolso, el cliente no ha pagado ninguna cuota; por ende, la exposición crediticia máxima comprometida por el banco es el monto desembolsado total (`funded_amnt`), o alternativamente ajustada por el factor de supervivencia de saldo según el plazo. Presentar ambas opciones demuestra dominio total del negocio de crédito bancario.
