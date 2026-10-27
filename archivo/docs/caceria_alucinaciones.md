# Cacería de la Alucinación 🎯
**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02  
**Trabajo 2:** Redes neuronales para datos tabulares (¿me prestas?)  
**Profesor:** Juan David Ospina Arango | **Monitor:** Andrés Mauricio Zapata Rincón  
**Universidad Nacional de Colombia — Facultad de Minas**

---

Las herramientas de inteligencia artificial generativa son aceleradores de desarrollo extraordinarios, pero producen con frecuencia errores conceptuales, matemáticos y de código redactados con absoluta convicción. A continuación, se documentan **cinco hallazgos genuinos** identificados durante el desarrollo del trabajo, cubriendo cinco categorías diferentes: **Matemáticas, Datos del Mundo Real, Código, Bibliografía y Conceptos**.

---

## Hallazgo 1: Matemáticas — Derivación invertida del Factor de la Scorecard

1. **Prompt utilizado (literal):**  
   > *"Dame la fórmula para calcular el Factor y el Offset de una scorecard crediticia si quiero un Score Base de 600 puntos, Odds de referencia de 50:1 y un PDO de 20 puntos. ¿Cómo es la ecuación final para transformar la probabilidad de incumplimiento p en puntaje?"*

2. **Parte relevante de la respuesta de la IA:**  
   > *"Para implementar la scorecard, las ecuaciones fundamentales son:*  
   > *$$\text{Factor} = \frac{\ln(2)}{\text{PDO}}$$*  
   > *$$\text{Offset} = \text{Score}_0 - \text{Factor} \cdot \ln(\text{Odds}_0)$$*  
   > *Y el puntaje de cualquier cliente con probabilidad de impago $p$ se obtiene directamente sumando:*  
   > *$$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{p}{1-p}\right)$$*  
   > *Con $\text{PDO}=20$, el Factor es $0.0346$."*

3. **Cómo se sospechó que había un error:**  
   - Si $\text{Factor} = \frac{\ln(2)}{20} \approx 0.0346$, cualquier variación en las probabilidades del cliente apenas movería el puntaje unos cuantos centésimos de punto, inutilizando la escala crediticia.
   - Además, en la ecuación dada, a mayor probabilidad de impago $p$, el término $\ln(p / (1-p))$ crece, lo que **aumentaría** el puntaje del cliente malo. En el negocio de crédito, un puntaje alto siempre debe representar menor riesgo y mayor solvencia.

4. **Evidencia que demuestra el error:**  
   Por definición estándar de riesgo de crédito (Siddiqi, 2012; Anderson, 2007), las odds se definen como la razón de buenos a malos: $\text{Odds} = \frac{1-p}{p}$. Al duplicar las odds de buen pagador ($2 \cdot \text{Odds}$), el puntaje debe incrementarse exactamente en $\text{PDO}$ puntos:
   $$\text{Score}(\text{Odds}) = \text{Offset} + \text{Factor} \cdot \ln(\text{Odds})$$
   $$\text{Score}(2 \cdot \text{Odds}) - \text{Score}(\text{Odds}) = \text{PDO}$$
   $$\text{Factor} \cdot \ln(2 \cdot \text{Odds}) - \text{Factor} \cdot \ln(\text{Odds}) = \text{PDO} \implies \text{Factor} \cdot \ln(2) = \text{PDO}$$
   $$\implies \text{Factor} = \frac{\text{PDO}}{\ln(2)}$$

5. **Corrección:**  
   $$\text{Factor} = \frac{\text{PDO}}{\ln(2)} = \frac{20}{\ln(2)} \approx 28.8539$$
   $$\text{Offset} = \text{Score}_0 - \text{Factor} \cdot \ln(\text{Odds}_0) = 600 - 28.8539 \cdot \ln(50) \approx 487.13$$
   Y la fórmula para evaluar a partir de la probabilidad de default $p$:
   $$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{1-p}{p}\right) = \text{Offset} - \text{Factor} \cdot \ln\left(\frac{p}{1-p}\right)$$

6. **Lección aprendida:**  
   Hacer siempre un análisis dimensional de las unidades y verificar la monotonicidad económica: un modelo de crédito nunca puede asignar mayor puntaje al mayor riesgo.

---

## Hallazgo 2: Datos del Mundo Real — Fuga de información (*Data Leakage*) con variables de cobranza

1. **Prompt utilizado (literal):**  
   > *"Estoy seleccionando variables de Lending Club para predecir si un cliente va a entrar en default (loan_status). ¿Cuáles son las variables numéricas más correlacionadas con el default que debería incluir en el modelo de clasificación para maximizar el AUC?"*

2. **Parte relevante de la respuesta de la IA:**  
   > *"Para obtener un AUC superior al 95 %, debes incluir sin falta las siguientes variables que tienen la mayor correlación con el default:*  
   > *- `recoveries`: dinero bruto recuperado tras la pérdida.*  
   > *- `collection_recovery_fee`: honorarios pagados a la agencia de cobranzas.*  
   > *- `total_rec_prncp`: total de principal pagado.*  
   > *- `last_pymnt_amnt`: monto del último pago recibido.*  
   > *Estas variables separan perfectamente a los créditos pagados de los castigados."*

3. **Cómo se sospechó que había un error:**  
   El caso de uso establecido en el enunciado es la **originación de crédito** ($t_0$). Si un cliente está solicitando un crédito hoy, no es posible saber cuánto dinero va a recuperar una agencia de cobranzas en tres años si no paga. Incluir variables del futuro destruye la validez operativa del modelo.

4. **Evidencia que demuestra el error:**  
   El diccionario oficial de Lending Club (`LCDataDictionary.xlsx`) describe explícitamente:
   - `recoveries`: *"post charge off gross recovery"* (recuperación posterior al reconocimiento del castigo contable).
   - `collection_recovery_fee`: *"post charge off collection fee"*.  
   En el momento de otorgamiento, el valor de estas variables es desconocido o idénticamente cero para cualquier nuevo solicitante. Utilizarlas en clasificación constituye una fuga masiva de datos (*target leakage*).

5. **Corrección:**  
   Se excluyeron estrictamente todas las columnas de comportamiento posterior al desembolso del modelo de clasificación de $PD$. Dichas variables (`recoveries`, `collection_recovery_fee`, `total_rec_prncp`, `funded_amnt`) se reservaron **única y exclusivamente** para la construcción matemática de la variable objetivo observada en la Parte 2 ($LGD$).

6. **Lección aprendida:**  
   La definición del caso de uso es la frontera temporal del modelado: ninguna variable observada posterior al momento de la decisión ($t_0$) puede ser una covariable de entrada.

---

## Hallazgo 3: Código — Invocación errónea del estadístico Kolmogorov-Smirnov ($KS$)

1. **Prompt utilizado (literal):**  
   > *"Escribe una función en Python para calcular el estadístico KS de un modelo de clasificación binaria usando scipy.stats.ks_2samp recibiendo y_true y las probabilidades predichas y_prob."*

2. **Parte relevante de la respuesta de la IA:**  
   > *"El estadístico de Kolmogorov-Smirnov evalúa la calidad del modelo comparando las predicciones con las etiquetas reales. Puedes calcularlo directamente así:*  
   > ```python
   > from scipy.stats import ks_2samp
   > 
   > def calcular_ks(y_true, y_prob):
   >     stat, p_val = ks_2samp(y_true, y_prob)
   >     return stat
   > ```
   > *Un valor de stat cercano a 1 indica excelente discriminación."*

3. **Cómo se sospechó que había un error:**  
   `y_true` es un vector discreto de etiquetas $\{0, 1\}$, mientras que `y_prob` es una variable continua en el intervalo $[0, 1]$. La prueba de dos muestras de Kolmogorov-Smirnov (`ks_2samp`) evalúa si dos conjuntos de datos continuos provienen de la misma distribución continua subyacente. Comparar $\{0, 1\}$ directamente con probabilidades continuas carece de sentido matemático en riesgo crediticio.

4. **Evidencia que demuestra el error:**  
   En la industria bancaria y en la estadística de riesgo, el estadístico $KS$ mide la máxima distancia vertical entre las funciones de distribución acumulada de las probabilidades predichas para la población de "buenos" ($y=0$) y la población de "malos" ($y=1$):
   $$KS = \sup_{s \in [0, 1]} \left| F_{\hat{p} \mid Y=0}(s) - F_{\hat{p} \mid Y=1}(s) \right|$$
   Al ejecutar la función de la IA sobre un modelo sintético perfecto, arrojaba valores absurdos ($KS \approx 0.50$ en vez de $1.0$).

5. **Corrección:**  
   Separar el vector de probabilidades según la etiqueta observada antes de llamar a `ks_2samp`:
   ```python
   from scipy.stats import ks_2samp

   def calcular_ks(y_true, y_prob):
       prob_buenos = y_prob[y_true == 0]
       prob_malos = y_prob[y_true == 1]
       stat = ks_2samp(prob_malos, prob_buenos).statistic
       return stat
   ```

6. **Lección aprendida:**  
   Comprender a fondo la definición matemática y económica de las métricas antes de delegar su implementación en librerías de terceros.

---

## Hallazgo 4: Bibliografía — Fabricación de referencias y DOI inexistentes

1. **Prompt utilizado (literal):**  
   > *"Proporcióname referencias bibliográficas en formato APA 7 sobre la superioridad de modelos basados en árboles (Gradient Boosting, LightGBM, XGBoost) frente a redes neuronales en datos tabulares de crédito, con autores reconocidos y DOI funcional."*

2. **Parte relevante de la respuesta de la IA:**  
   > *"Una referencia fundamental en este campo es:*  
   > *Ospina-Arango, J. D., & Martinez, C. (2021). Neural Networks vs Gradient Boosting in Consumer Credit Risk: An Empirical Study on Tabular Data. Journal of Financial Data Science, 3(4), 45-59. https://doi.org/10.3905/jfds.2021.1.078"*

3. **Cómo se sospechó que había un error:**  
   La IA intentó congraciarse incluyendo el nombre del profesor del curso (*Juan David Ospina Arango*) como coautor en una revista extranjera especializada en un tema específico de datos tabulares.

4. **Evidencia que demuestra el error:**  
   - Se consultó la API oficial de resolución de DOI (`https://doi.org/10.3905/jfds.2021.1.078`), la cual arrojó un error `404 Not Found` (el identificador no está registrado en CrossRef).
   - Se realizaron búsquedas exhaustivas en Google Scholar, DBLP y Scopus con las palabras clave del título; el artículo no existe en ninguna base de datos académica indexada.

5. **Corrección:**  
   Se eliminó la referencia fabricada y se sustituyó por los estudios seminales, verificables y arbitrados que realmente lideran la discusión científica actual sobre datos tabulares:
   - Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35, 507–520. https://openreview.net/forum?id=Fp7__phQszn
   - Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems (NeurIPS 2021)*, 34, 18932–18943.

6. **Lección aprendida:**  
   Toda referencia generada por IA debe ser verificada de manera obligatoria e independiente en bases de datos indexadas y comprobar la resolución activa de su DOI antes de citarla en un reporte técnico.

---

## Hallazgo 5: Conceptos — Falacia del Teorema de Aproximación Universal en Datos Tabulares

1. **Prompt utilizado (literal):**  
   > *"Explica por qué un perceptrón multicapa profundo siempre superará a una regresión logística o un árbol de decisión en datos tabulares bancarios."*

2. **Parte relevante de la respuesta de la IA:**  
   > *"El Perceptrón Multicapa profundo supera invariablemente a la regresión logística y a los árboles de decisión porque el Teorema de Aproximación Universal (Hornik et al., 1989) demuestra matemáticamente que una red neuronal con suficientes capas y neuronas puede aproximar cualquier función continua compleja con un error arbitrariamente pequeño. Por tanto, la red neuronal siempre aprenderá mejores relaciones no lineales que cualquier modelo tabular clásico, sin importar el ruido o las correlaciones de los datos."*

3. **Cómo se sospechó que había un error:**  
   El Teorema de Aproximación Universal es un teorema de **representación asintótica**, no de **estimación ni de generalización estadística** fuera de muestra. Afirmar que garantiza superioridad empírica en cualquier problema tabular ignora décadas de investigación en aprendizaje automático.

4. **Evidencia que demuestra el error:**  
   - La literatura moderna (Borisov et al., 2022; Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022) demuestra que en conjuntos de datos tabulares las redes neuronales densas carecen del sesgo inductivo adecuado (como la invariancia a traslaciones de las CNNs en imágenes o la causalidad secuencial de los Transformers en texto).
   - En nuestro propio experimento empírico de crédito:
     - Regresión Logística (81 parámetros): AUC = **0.7075**.
     - MLP clasificador (8,033 parámetros, 100 veces más pesado): AUC = **0.7105**.
     - La mejora marginal fue de apenas $\Delta \text{AUC} = +0.003$, a costa de perder interpretabilidad directa e incrementar el costo computacional.
     - En la tarea de LGD, el RMSE sin sesgo del MLP en prueba fue idéntico al de la regresión lineal (**0.0766 vs 0.0773**), y la regresión lineal ordenó mejor los créditos (Spearman de **0.140 vs 0.106**).

5. **Corrección:**  
   El teorema garantiza capacidad teórica de representación, pero no garantiza convergencia al mínimo global ni capacidad de generalización en muestras finitas ruidosas con frontera de decisión irregular. En datos tabulares con variables heterogéneas, los modelos lineales regularizados y los árboles de decisión suelen ser iguales o superiores a las redes densas estándar.

6. **Lección aprendida:**  
   No confundir la capacidad teórica de representación con el rendimiento empírico de generalización estadística. La evidencia empírica rigurosa prima siempre sobre citas teóricas descontextualizadas.
