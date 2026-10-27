# ¿Me prestas? Redes Neuronales y Riesgo de Crédito en Datos Tabulares: De la Teoría a la Decisión Bancaria

**Autores:** Equipo de Trabajo — Curso de Redes Neuronales y Algoritmos Bioinspirados  
**Profesor:** Juan David Ospina Arango | **Monitor:** Andrés Mauricio Zapata Rincón  
**Universidad Nacional de Colombia — Facultad de Minas**  
**Semestre:** 2026-02  

---

### Enlaces del Proyecto
* 💻 **Repositorio de GitHub:** [https://github.com/](https://github.com/) *(Reproducibilidad completa)*
* 🌐 **Aplicación Web Interactiva:** [https://share.streamlit.io/](https://share.streamlit.io/) *(Scorecard, Percentil y Simulador What-If)*
* 🎥 **Video Promocional del Producto (Pitch):** [https://youtube.com/](https://youtube.com/) *(Solución al negocio bancario)*

---

## Resumen Ejecutivo

Antes de otorgar un crédito de consumo, una entidad financiera prudente debe responder con rigor a dos preguntas fundamentales:
1. **¿Qué tan probable es que el solicitante incumpla con sus obligaciones?** (Probabilidad de Incumplimiento o $PD$, problema de **clasificación binaria**).
2. **Si el cliente incumple, ¿qué fracción del dinero prestado no podrá recuperarse?** (Pérdida Dado el Incumplimiento o $LGD$, problema de **regresión**).

A partir de 887,379 registros históricos de préstamos originados entre junio de 2007 y diciembre de 2015 en la plataforma *Lending Club*, construimos, optimizamos y evaluamos modelos basados en Perceptrones Multicapa (MLP) comparados de forma justa frente a modelos de baja complejidad (Regresión Logística y Regresión Lineal MCO). Los modelos fueron evaluados sobre **particiones temporales puras** (*Out-of-Time*, OOT) y validados estocásticamente a lo largo de **30 semillas aleatorias independientes**. 

Asimismo, transformamos el modelo de probabilidad en una **Scorecard Crediticia calibrada** (Score Base 600 puntos, Odds 50:1, PDO 20 puntos) y estimamos la **Pérdida Esperada** prospectiva ($E[L] = PD \cdot LGD \cdot EAD$). Los resultados evidencian que, si bien la red neuronal logra un desempeño marginalmente superior en discriminación de $PD$ ($\Delta \text{AUC} \approx +0.003$), en la estimación de $LGD$ el comportamiento individual está dominado por la censura temporal del recobro y políticas operativas de cobranza, donde la red neuronal no aporta información discriminante adicional frente al modelo lineal clásico.

---

## 1. El Caso de Uso Bancario y Frontera de Decisión

Todo proceso de modelado predictivo en banca debe estar estrictamente anclado a un caso de uso operativo para evitar el error metodológico más destructivo en machine learning: la **fuga de datos** (*data leakage*).

* **¿Quién usa el modelo?:** La Dirección de Riesgo de Crédito Minorista y los analistas del motor de suscripción (*Credit Underwriting Engine*).
* **¿Para qué decisión?:** 
  1. Aprobación o rechazo automatizado de la solicitud de crédito de consumo.
  2. Asignación de tasa de interés ajustada por riesgo (*risk-based pricing*).
  3. Cálculo de provisiones contables y requerimientos de capital económico bajo el marco de Basilea.
* **¿En qué momento ($t_0$)?:** En el instante exacto en que el cliente diligencia su solicitud en la plataforma web, **antes** de que cualquier desembolso ocurra.
* **Información disponible en $t_0$:** Únicamente datos sociodemográficos verificados (ingreso anual, tenencia de vivienda, antigüedad laboral), atributos de la solicitud (monto financiado, plazo pactado a 36 o 60 meses, tasa de interés ofrecida) y variables del historial crediticio extraídas del buró de crédito en la fecha de consulta (cuentas abiertas, delincuencias previas, saldo en líneas rotativas).
* **Restricción estricta anti-fuga:** Se excluyen de raíz todas las variables generadas *expost* durante la vida del crédito (`recoveries`, `collection_recovery_fee`, `total_rec_prncp`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`). Como se ilustra en la Sección de la Cacería de la Alucinación, incluir estas columnas genera modelos con desempeño aparentemente perfecto en entrenamiento pero completamente inútiles en la realidad operativa.

---

## 2. Datos, Análisis Descriptivo e Hipótesis Iniciales

### 2.1. Fuente de Datos y Diccionario Oficial
Trabajamos con el conjunto oficial *Credit Risk Analysis* de Kaggle (`ranadeep/credit-risk-dataset`), correspondiente a microdatos de Lending Club. Las definiciones textuales provienen del diccionario institucional `LCDataDictionary.xlsx`:
- `loan_status`: Estado final o actual de pago del crédito.
- `funded_amnt`: Monto total comprometido y desembolsado para el préstamo.
- `int_rate`: Tasa de interés pactada para el crédito.
- `dti`: Relación deuda/ingreso (*debt-to-income ratio*), calculada dividiendo los pagos mensuales de deuda excluyendo hipoteca entre los ingresos mensuales brutos declarados.
- `annual_inc`: Ingreso bruto anual declarado por el solicitante.
- `emp_length`: Duración del empleo en años (valores de 0 a 10+).

### 2.2. Codificación de la Variable Objetivo de Clasificación
En cumplimiento de la directriz oficial, se filtraron los registros ambiguos o con maduración incompleta, codificando la variable objetivo binaria $y \in \{0, 1\}$ según la Tabla 1.

#### Tabla 1: Codificación de `loan_status` para Clasificación de Incumplimiento
| Categoría Original en Lending Club | Codificación | Justificación de Negocio |
| :--- | :---: | :--- |
| `Fully Paid` | 0 | Buen pagador confirmado. Crédito liquidado en su totalidad. |
| `Does not meet the credit policy. Status:Fully Paid` | 0 | Buen pagador confirmado (a pesar de excepciones de política inicial). |
| `Charged Off` | 1 | Mal pagador. Crédito reconocido contablemente como incobrable. |
| `Default` | 1 | Mal pagador. Impago grave confirmado. |
| `Late (31-120 days)` | 1 | Cartera vencida mayor a 30 días; alta probabilidad de incobrabilidad. |
| `Does not meet the credit policy. Status:Charged Off` | 1 | Mal pagador y no cumplió requisitos. |
| `Current`, `Issued`, `In Grace Period`, `Late (16-30 days)` | **Excluidos (NA)** | No se incluye en el entrenamiento por maduración incierta. |

*Fuente: Elaboración propia a partir del enunciado y lineamientos del Banco.*

Tras el filtrado y preprocesamiento (eliminación de valores nulos y codificación *one-hot* con eliminación de la primera columna para evitar multicolinealidad), se consolidó una matriz final con **193,050 registros** originados entre agosto de 2012 y diciembre de 2015, con **80 variables predictoras** y una tasa base promedio de incumplimiento del **23.3 %**.

### 2.3. Formulación de Hipótesis Iniciales de Riesgo
Antes de entrenar los modelos, el equipo formuló cinco hipótesis explícitas:
* **Hipótesis 1 (Capacidad de Endeudamiento):** A mayor relación deuda/ingreso (`dti`), mayor probabilidad de default ($PD$), debido a la menor holgura financiera del cliente ante imprevistos.
* **Hipótesis 2 (Calificación y Tasa de Interés):** Créditos con tasas de interés (`int_rate`) más elevadas y subgrados más riesgosos (`sub_grade`) reflejarán la mayor tasa de default, evidenciando que el mecanismo de precios de Lending Club captura riesgo real.
* **Hipótesis 3 (Plazo del Crédito):** Los créditos pactados a 60 meses presentarán mayor probabilidad de default y mayor severidad de pérdida ($LGD$) que los de 36 meses debido a una amortización de capital significativamente más lenta.
* **Hipótesis 4 (Estabilidad Laboral):** Solicitantes con antigüedad laboral superior a 10 años (`emp_length = 10+`) tendrán una probabilidad de default sustancialmente menor que aquellos con menos de un año.
* **Hipótesis 5 (Predecibilidad de la LGD):** Las características socioeconómicas del cliente al originar permitirán estimar con alta precisión ($R^2 > 0.50$, Spearman $> 0.50$) el porcentaje final que se perderá en caso de impago ($LGD$).

---

## 3. Parte 1: Modelado de Probabilidad de Incumplimiento (PD)

### 3.1. Partición Temporal Fuera de Tiempo (Out-of-Time, OOT)
En riesgo de crédito, una partición aleatoria convencional (*k-fold cross-validation* mezclada) es conceptualmente defectuosa, pues evalúa el modelo sobre clientes contemporáneos al conjunto de entrenamiento. Para simular el entorno bancario real, donde un modelo entrenado hoy califica solicitudes de meses futuros, ordenamos el conjunto cronológicamente por la fecha de emisión (`issue_d`) y realizamos la división 60% entrenamiento, 20% validación y 20% prueba OOT, detallada en la Tabla 2.

#### Tabla 2: Partición Temporal Cronológica de Clasificación
| Conjunto | Rango Temporal (`issue_d`) | Número de Créditos ($n$) | Tasa de Incumplimiento |
| :--- | :---: | :---: | :---: |
| **Entrenamiento** | 2012-08 $\to$ 2014-04 | 115,830 | 21.6 % |
| **Validación** | 2014-04 $\to$ 2014-10 | 38,610 | 26.3 % |
| **Prueba OOT** | 2014-10 $\to$ 2015-12 | 38,610 | 25.3 % |

*Fuente: Elaboración propia a partir de datos procesados de Lending Club.*

> [!NOTE]
> Obsérvese que la tasa de incumplimiento aumenta en validación y prueba (de 21.6 % a más de 25 %). Esto refleja el envejecimiento natural de las cosechas y cambios de política de colocación, constituyendo una prueba de estrés rigurosa para los modelos.

### 3.2. Modelos Evaluados y Arquitectura
1. **Modelo de Referencia (Regresión Logística):** Modelo lineal generalizado con enlace logit, penalización $L_2$ ligera y ponderación balanceada de clases (`class_weight='balanced'`). Cuenta con **81 parámetros entrenables** (80 coeficientes y 1 intercepto).
2. **Red Neuronal Artificial (Perceptrón Multicapa, MLP):** 
   - Entrada: 80 neuronas.
   - Capa Oculta 1: Linear(80 $\to$ 64) $\to$ BatchNorm1d(64) $\to$ ReLU $\to$ Dropout($p=0.2$).
   - Capa Oculta 2: Linear(64 $\to$ 32) $\to$ BatchNorm1d(32) $\to$ ReLU $\to$ Dropout($p=0.2$).
   - Capa Oculta 3: Linear(32 $\to$ 16) $\to$ BatchNorm1d(16) $\to$ ReLU.
   - Capa de Salida: Linear(16 $\to$ 1) con función de pérdida `BCEWithLogitsLoss(pos_weight=3.39)`.
   - Optimización: Adam con tasa de aprendizaje inicial $\eta = 10^{-3}$, decaimiento de pesos $L_2 = 10^{-4}$, programador `ReduceLROnPlateau(factor=0.5, patience=3)` y parada temprana con paciencia de 10 épocas monitoreando el AUC de validación.
   - Cuenta con **8,033 parámetros entrenables**.

### 3.3. Resultados de Discriminación y Calibración
La evaluación se realizó aplicando el desesgo analítico de probabilidades $\sigma(\hat{z} - \ln(w_{\text{pos}}))$ sobre los conjuntos de validación y prueba, reportado en la Tabla 3.

#### Tabla 3: Métricas de Evaluación de Modelos de Clasificación de Default (PD)
| Modelo | Conjunto | AUC-ROC | Índice de Gini | Estadístico KS | Brier Score | Brier de Referencia |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | Validación | 0.7075 | 0.4150 | 0.3082 | 0.1741 | 0.1890 |
| **Regresión Logística** | Prueba OOT | 0.7071 | 0.4142 | 0.3075 | 0.1743 | 0.1891 |
| **Red Neuronal (MLP)** | Validación | **0.7104** | **0.4209** | **0.3120** | **0.1726** | 0.1890 |
| **Red Neuronal (MLP)** | Prueba OOT | **0.7105** | **0.4210** | **0.3117** | **0.1725** | 0.1891 |

*Fuente: Elaboración propia a partir de resultados de entrenamiento y validación.*

Como se observa en la Tabla 3, ambos modelos son ampliamente informativos ($BS < BS_{\text{ref}}$) y logran superar el umbral bancario tradicional de $KS > 0.30$. No obstante, el MLP supera a la regresión logística por un margen extremadamente estrecho de **+0.0034 en AUC**.

### 3.4. Scorecard Crediticia Formal
A partir de la probabilidad de incumplimiento calibrada $p$, derivamos matemáticamente la scorecard bancaria bajo los parámetros regulatorios fijados:
- Puntaje Base: $\text{Score}_0 = 600$ puntos.
- Odds de Referencia: $\text{Odds}_0 = 50:1$ (50 buenos por cada mal pagador).
- Puntos para Duplicar las Odds: $\text{PDO} = 20$ puntos.

Aplicando las ecuaciones derivadas formalmente en `docs/fundamentacion_matematica.md`:
$$\text{Factor} = \frac{\text{PDO}}{\ln(2)} = \frac{20}{\ln(2)} \approx 28.8539$$
$$\text{Offset} = \text{Score}_0 - \text{Factor} \cdot \ln(\text{Odds}_0) = 600 - 28.8539 \cdot \ln(50) \approx 487.1227$$
$$\text{Score}(p) = 487.1227 - 28.8539 \cdot \ln\left(\frac{p}{1 - p}\right)$$

#### Tabla 4: Puntos de Calibración de la Scorecard de Consumo
| Probabilidad de Impago ($PD$) | Odds de Buen Pagador ($\frac{1-p}{p}$) | Puntaje Resultante (Score) | Calificación Operativa |
| :---: | :---: | :---: | :---: |
| 0.50 % | 199.0 : 1 | 640 puntos | Excelente (Bajo Riesgo) |
| 1.96 % | 50.0 : 1 | **600 puntos** | **Punto Base de Referencia** |
| 3.85 % | 25.0 : 1 | 580 puntos ($-20$ pts por mitad de odds) | Riesgo Medio-Bajo |
| 7.41 % | 12.5 : 1 | 560 puntos | Riesgo Medio |
| 23.30 % | 3.29 : 1 | 521 puntos | Tasa Promedio Poblacional |
| 50.00 % | 1.0 : 1 | 487 puntos | Alto Riesgo (Indiferencia) |
| 80.00 % | 0.25 : 1 | 447 puntos | Muy Alto Riesgo (Rechazo) |

*Fuente: Elaboración propia a partir del motor de scoring (`scripts/scorecard_engine.py`).*

### 3.5. Variables de Riesgo e Interpretabilidad
Para interpretar las variables que determinan el riesgo, calculamos la importancia por permutación (*Permutation Feature Importance*, Breiman, 2001) sobre el conjunto de prueba y analizamos los coeficientes de la regresión logística. Como se observa en la Tabla 5, las variables con mayor capacidad predictiva corresponden a la tasa de interés pactada, el subgrado de riesgo, el plazo a 60 meses y el DTI.

#### Tabla 5: Top 6 Variables de Mayor Riesgo en el Otorgamiento
| Variable | Signo del Impacto | Interpretación Económica |
| :--- | :---: | :--- |
| `int_rate` | Positivo ($+$ Riesgo) | Mayor tasa refleja mayor riesgo percibido inicialmente por la plataforma. |
| `sub_grade` | Positivo ($+$ Riesgo) | Calificaciones internas desfavorables (D, E, F, G) multiplican el riesgo. |
| `term_ 60 months` | Positivo ($+$ Riesgo) | Los créditos a 5 años acumulan mayor exposición temporal a shocks de liquidez. |
| `dti` | Positivo ($+$ Riesgo) | Mayor porcentaje de ingresos comprometido en deudas previas reduce margen de pago. |
| `annual_inc` | Negativo ($-$ Riesgo) | Mayores ingresos brutos proporcionan un amortiguador financiero protector. |
| `inq_last_6mths` | Positivo ($+$ Riesgo) | Múltiples consultas recientes en buró denotan búsqueda desesperada de liquidez. |

*Fuente: Elaboración propia a partir de coeficientes e importancias estandarizadas.*

---

## 4. Parte 2: Pérdida Dado el Incumplimiento (LGD) y Pérdida Esperada

### 4.1. Definición y Construcción de la Variable Objetivo LGD
Para los créditos que efectivamente entraron en castigo contable (`Charged Off`, 46,009 operaciones), la pérdida efectiva representa la porción de la exposición no recuperada a través de gestiones de cobro judicial o extrajudicial:
$$EAD_i = \text{funded\_amnt}_i - \text{total\_rec\_prncp}_i$$
$$R_i = \text{recoveries}_i - \text{collection\_recovery_fee}_i$$
$$LGD_i = \min\left(1, \max\left(0, 1 - \frac{R_i}{EAD_i}\right)\right)$$

#### Limitaciones Fundamentales de la LGD Observada:
1. **Censura temporal a la derecha:** La cobranza judicial de un crédito en mora toma entre 2 y 5 años. En las cosechas recientes (2014-2015), los créditos que recién cayeron en castigo apenas están iniciando el proceso de cobro, por lo que casi no tienen recuperaciones registradas ($recoveries = 0$), inflando la LGD observada al 98 % en 2015 frente al 91 % en 2012.
2. **Ausencia de descuento financiero:** No se dispone de las fechas exactas de los flujos de cobro, impidiendo descontar los pagos al valor presente de la fecha de mora.
3. **Costos indirectos no capturados:** Solo se restan los honorarios directos de agencia (`collection_recovery_fee`), omitiendo costos legales internos y administrativos del banco.

### 4.2. Modelos de LGD y Evaluación
Se comparó una Regresión Lineal MCO de 42 parámetros frente a un Perceptrón Multicapa [64, 32, 16] con activación tanh, regularización dropout (0.2), salida sigmoide y función de pérdida MSE (**5,537 parámetros**). La búsqueda aleatoria de 30 configuraciones y el posterior entrenamiento con **30 semillas aleatorias independientes** arrojaron los resultados presentados en la Tabla 6.

#### Tabla 6: Desempeño Comparativo de Modelos de LGD (Conjunto de Prueba OOT)
| Modelo | MAE | RMSE | RMSE sin Sesgo | Sesgo de Nivel | Coeficiente Spearman | Parámetros |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Media Histórica (Base)** | 0.0814 | 0.0919 | 0.0765 | $-0.0509$ | — | 1 |
| **Regresión Lineal MCO** | 0.0740 | 0.0878 | 0.0773 | $-0.0418$ | **0.140** | 42 |
| **Red Neuronal (MLP LGD)** | **0.0630** | **0.0800** | **0.0766** | **$-0.0232$** | 0.106 | 5,537 |

*Fuente: Elaboración propia a partir de `docs/parte2_lgd.md` y `resultados/semillas_lgd.csv`.*

### 4.3. Diagnóstico Crítico: El Triunfo Ilusorio del MLP en LGD
A primera vista, la red neuronal parecería superar a la regresión lineal porque reduce el RMSE de 0.0878 a 0.0800. Sin embargo, un análisis econométrico riguroso desmonta esta conclusión:
1. **El error sin sesgo es idéntico:** Al substraer el error de nivel medio, el RMSE sin sesgo de la red neuronal (**0.0766**) empata técnicamente con el de la regresión lineal (**0.0773**).
2. **El modelo lineal ordena mejor:** La correlación de rangos de Spearman de la regresión lineal es superior (**0.140 vs 0.106**).
3. **El efecto de la censura:** La red neuronal "gana" en RMSE simplemente porque su nivel medio de predicción se infló hacia arriba (+0.026 en entrenamiento), acercándose artificialmente a la media inflada por censura del conjunto de prueba reciente (donde el 81% de los créditos tiene LGD = 1.0 exacta).
4. **Ningún modelo reproduce la bimodalidad:** La LGD observada es fuertemente bimodal (se concentra en 0 o en 1). Ambos modelos actúan como estimadores de la media condicional condesando las predicciones en un rango estrecho entre 0.90 y 0.96.

### 4.4. Integración de la Pérdida Esperada ($E[L]$)
Para estimar la pérdida prospectiva esperada al momento de la originación ($t_0$), definimos la exposición esperada al incumplimiento ($\widehat{EAD}$) a partir del saldo promedio remanente al momento del castigo:
$$\widehat{EAD}_i = \text{funded\_amnt}_i \cdot \bar{f}_{\text{plazo}(i)}, \quad \bar{f}_{36} = 0.670, \quad \bar{f}_{60} = 0.846$$
La pérdida esperada final se calcula e implementa en el motor analítico como:
$$E[L]_i = PD_i \cdot \widehat{LGD}_i \cdot \widehat{EAD}_i$$

---

## 5. Regla de Comparación Justa y Discusión: ¿Qué Aportaron las Redes Neuronales?

Para dar estricto cumplimiento a la regla de comparación justa, resumimos en la Tabla 7 los presupuestos de búsqueda, tiempos de cómputo y variabilidad entre 30 semillas independientes.

#### Tabla 7: Matriz Integral de Comparación Justa entre Familias de Modelos
| Tarea | Modelo | Parámetros Entrenables | Configuraciones Exploradas | Tiempo de Búsqueda | Tiempo 30 Semillas | Métrica Principal (Media $\pm$ Std) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clasificación (PD)** | Regresión Logística | 81 | 1 | 0.8 s | Determinista | AUC = 0.7071 |
| **Clasificación (PD)** | Red Neuronal (MLP) | 8,033 | 25 épocas con Early Stop | 1.8 min | 4.2 min | AUC = $0.7105 \pm 0.0009$ |
| **Regresión (LGD)** | Regresión Lineal MCO | 42 | 1 | 0.05 s | Determinista | RMSE = 0.0878 |
| **Regresión (LGD)** | Red Neuronal (MLP) | 5,537 | 30 configuraciones aleatorias | 5.7 min | 13.6 min | RMSE = $0.0796 \pm 0.0005$ |

*Fuente: Elaboración propia a partir de registros en `resultados/busqueda_lgd.csv` y `resultados/semillas_lgd.csv`.*

### Aprendizajes Fundamentales sobre el Problema:
1. **La brecha de complejidad vs ganancia en datos tabulares:** Multiplicar por 100 el número de parámetros (de 81 a 8,033) en clasificación de default solo produjo una mejora de +0.0034 en AUC. En datos tabulares con variables heterogéneas, ruido de captura y baja dimensionalidad geométrica, las redes neuronales estándar sufren para superar a los modelos lineales bien regularizados (Grinsztajn et al., 2022).
2. **Interpretabilidad regulatoria:** La regresión logística ofrece una relación directa entre coeficientes y probabilidades mediante razones de momios (*odds ratios*), facilitando la justificación de decisiones ante reguladores y clientes. En el MLP, la no linealidad requiere recurrir a aproximaciones como SHAP o permutaciones, aumentando el costo operativo de auditoría.
3. **La ilusión de la métrica en presencia de censura:** En la tarea de LGD aprendimos que una red neuronal puede parecer superior según métricas de error cuadrático medio (RMSE) no porque comprenda mejor los determinantes del cliente, sino porque su mayor flexibilidad le permite mimetizar el sesgo de nivel introducido por la maduración incompleta de las cosechas recientes.
4. **Verificación de hipótesis iniciales:**
   - *Hipótesis 1 (DTI), 2 (Tasa de interés) y 3 (Plazo a 60 meses):* **Confirmadas ampliamente.** Constituyen los tres pilares del riesgo de la cartera.
   - *Hipótesis 4 (Antigüedad laboral):* **Parcialmente refutada.** Su capacidad de separación fue mínima frente a variables financieras.
   - *Hipótesis 5 (LGD predecible con datos de originación):* **Rotundamente refutada.** La LGD individual depende de la política de cobro posterior y ventas de cartera, no del perfil socioeconómico inicial del solicitante.

---

## 6. Sección de Uso de Herramientas de Inteligencia Artificial

Durante el desarrollo del trabajo, el equipo utilizó herramientas de Inteligencia Artificial Generativa (asistentes de código basados en LLMs). A continuación, se detallan los prompts clave utilizados y el impacto de su utilización en el proyecto:

1. **Prompt de Estructuración de la Scorecard:**  
   *Texto:* `"Deriva las ecuaciones para transformar probabilidades predichas a una escala de scorecard bancaria con puntos para duplicar las odds (PDO), Odds base y Score base."`  
   *Impacto:* Permitió formalizar la derivación matemática paso a paso. Sin embargo, requirió revisión crítica inmediata debido a la alucinación sobre el signo del factor documentada en el Hallazgo 1.
2. **Prompt de Implementación en PyTorch:**  
   *Texto:* `"Genera una clase nn.Module en PyTorch para un MLP con capas densas, BatchNorm1d, Dropout y BCEWithLogitsLoss con pos_weight para balancear clases."`  
   *Impacto:* Aceleró la escritura del código base de entrenamiento y optimización en `notebooks/clasificacion.ipynb`.
3. **Prompt de Interfaz de Usuario en Streamlit:**  
   *Texto:* `"Crea una aplicación interactiva en Streamlit que reciba los inputs del cliente, calcule el score en la scorecard y grafique un histograma comparativo."`  
   *Impacto:* Proveyó el esqueleto visual de la aplicación web, el cual fue refinado para incorporar el motor de $E[L]$ y el simulador de sensibilidad (*What-If*).

---

## 7. Cacería de la Alucinación 🎯 (Resumen de Hallazgos)

En estricto cumplimiento de la rúbrica, documentamos cinco hallazgos donde la IA generó afirmaciones incorrectas o engañosas:
* **Hallazgo 1 (Matemáticas):** Inversión analítica de la fórmula del Factor de la scorecard ($\text{Factor} = \ln(2)/\text{PDO}$ en lugar de $\text{PDO}/\ln(2)$), lo que causaba que a mayor probabilidad de impago aumentara el puntaje crediticio.
* **Hallazgo 2 (Datos del Mundo Real):** Sugerencia de incluir variables de cobranza posterior (`recoveries`, `collection_recovery_fee`) en el clasificador de originación, generando una fuga masiva de datos (*target leakage*).
* **Hallazgo 3 (Código):** Invocación de `scipy.stats.ks_2samp(y_true, y_prob)` comparando etiquetas binarias contra probabilidades continuas, en lugar de contrastar las probabilidades de buenos contra malos.
* **Hallazgo 4 (Bibliografía):** Fabricación de un paper inexistente en el *Journal of Financial Data Science* con un DOI falso (`10.3905/jfds.2021.1.078`) atribuyendo coautoría al profesor del curso.
* **Hallazgo 5 (Conceptos):** Afirmación de que el Teorema de Aproximación Universal garantiza que una red neuronal profunda siempre superará a un modelo lineal o árbol en datos tabulares finitos.

*(El detalle exhaustivo con los 6 campos por hallazgo se encuentra compilado en [`docs/caceria_alucinaciones.md`](../docs/caceria_alucinaciones.md)).*

---

## 8. Referencias Bibliográficas (Normas APA, 7.ª edición)

* Anderson, R. (2007). *The credit scoring toolkit: Theory and practice for retail credit risk management and decision automation*. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199226405.001.0001
* Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*(10), 281–305. https://jmlr.org/papers/v13/bergstra12a.html
* Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2022). Deep neural networks and tabular data: A survey. *IEEE Transactions on Neural Networks and Learning Systems*, 1–21. https://doi.org/10.1109/TNNLS.2022.3229161
* Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
* Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems (NeurIPS 2021)*, 34, 18932–18943. https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229a-Abstract.html
* Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35, 507–520. https://openreview.net/forum?id=Fp7__phQszn
* Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks, 2*(5), 359–366. https://doi.org/10.1016/0893-6080(89)90020-8
* Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*. https://doi.org/10.48550/arXiv.1412.6980
* Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765–4774. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
* Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1
* Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011
* Siddiqi, N. (2012). *Credit risk scorecards: Developing and implementing intelligent credit scoring*. John Wiley & Sons. https://doi.org/10.1002/9781119201731
