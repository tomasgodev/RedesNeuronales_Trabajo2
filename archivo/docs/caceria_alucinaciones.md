# Cacería de la alucinación 🎯

**Curso:** Redes neuronales y algoritmos bioinspirados — Semestre 2026-02
**Trabajo 2:** Redes neuronales para datos tabulares (¿me prestas?)
**Profesor:** Juan David Ospina Arango | **Monitor:** Andrés Mauricio Zapata Rincón
**Universidad Nacional de Colombia — Facultad de Minas**



---

## 0. Resumen y transparencia

Documentamos **seis hallazgos** (errores distintos) que provienen de **cuatro conversaciones** con una IA generativa durante el desarrollo. Todas se hicieron con **Gemini** el **26 de octubre de 2026**. Los prompts están copiados literalmente, con la respuesta tal como la recibimos.

| # | Hallazgo | Categoría | Conversación | Gravedad |
|---|---|---|---|---|
| 1 | Codificación invertida del objetivo (1 = buen pagador) | Código | A | Alta |
| 2 | El código descarta las variantes `Does not meet the credit policy` que el texto pedía incluir | Código, Datos del mundo real | A | Media |
| 3 | Suma `total_rec_late_fee` a la recuperación posterior al incumplimiento | Datos del mundo real | B | Media |
| 4 | Presenta `out_prncp` como equivalente al monto original; con él, la LGD sería 1 para todos | Datos del mundo real | B | Alta (silenciosa) |
| 5 | Imprecisiones sobre Brier, normativa y la prueba de Hosmer-Lemeshow | Conceptos | C | Menor |
| 6 | Criterio incorrecto para comparar dos AUC (solapamiento de intervalos) y validación cruzada aleatoria como remedio en datos con dependencia temporal | Conceptos | D | Menor |

Categorías cubiertas: **Código, Datos del mundo real y Conceptos**. No cubrimos Matemáticas ni Bibliografía: no encontramos allí errores que podamos demostrar, y no queremos inventarlos.

Dos aclaraciones:

- Los hallazgos 1 y 2 salen de la misma respuesta, igual que los 3 y 4. Los contamos por separado porque son errores distintos, con evidencia distinta y consecuencias distintas.
- Los hallazgos 5 y 6 son **menores**: la idea central de cada respuesta es correcta. Los documentamos porque las imprecisiones son reales y se pueden verificar, pero no los presentamos como errores graves. Los cuatro primeros son los que habrían afectado el resultado.

---

## 1. Conversaciones

### Conversación A — Clasificación de `loan_status` (Gemini, 26/oct/2026)

**Prompt (literal):**

> Estoy haciendo un modelo para predecir si una persona va a incumplir con el pago de un crédito usando datos de Lending Club. ¿Cómo debería clasificar los diferentes estados de loan_status entre buenos y malos pagadores?

**Fragmentos relevantes de la respuesta:**

> Buenos Pagadores (Clase `1`): `Fully Paid` … Malos Pagadores (Clase `0`): `Charged Off`, `Default`, `Late (31-120 days)` … *Variantes:* `Does not meet the credit policy. Status:Charged Off` (si tu dataset incluye esta categoría histórica, debe ir con los malos).

```python
estados_validos = ['Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)']
df = df[df['loan_status'].isin(estados_validos)].copy()
# 1 = Buen pagador, 0 = Mal pagador (Default / Castigado / Con retraso severo)
def clasificar_estado(status):
  if status == 'Fully Paid':
    return 1
  else:
    return 0
```

Además, la respuesta presenta el enfoque de excluir `Current` y trabajar solo con créditos terminados como "el estándar de oro".

### Conversación B — Fórmula de la LGD (Gemini, 26/oct/2026)

**Prompt (literal):**

> Estoy trabajando con datos de créditos y necesito calcular la LGD, que representa cuánto dinero se pierde cuando alguien incumple. ¿Qué fórmula podría usar con las variables que tiene el dataset de Lending Club?

**Fragmentos relevantes de la respuesta:**

> `funded_amnt` (o `out_prncp` al momento del default): El monto original prestado.

> Recuperación Neta = recoveries − collection_recovery_fee + total_rec_late_fee

```python
df_default = df[df['loan_status'] == 'Charged Off'].copy()
df_default['EAD'] = df_default['funded_amnt'] - df_default['total_rec_prncp']
...
df_default['recovery_rate'] = np.where(df_default['EAD'] > 0,
                                       df_default['net_recoveries'] / df_default['EAD'], 0)
df_default['LGD'] = 1 - df_default['recovery_rate']
```

### Conversación C — AUC y calibración (Gemini, 26/oct/2026)

**Prompt (literal):**

> Estoy evaluando un modelo de riesgo de crédito y obtuve un AUC alto. ¿Eso significa que mi modelo predice bien las probabilidades de incumplimiento?

**Fragmentos relevantes de la respuesta:**

> En la gestión de riesgos y marcos regulatorios (como IFRS 9 o Basilea), la calibración exacta de la **PD** es obligatoria para calcular las pérdidas esperadas y el capital de reserva.

> **Brier Score** … Cuanto más cercano a 0, mayor precisión y calibración tiene el modelo.

> **Test de Hosmer-Lemeshow** … Un p-valor alto (superior a 0.05) sugiere que no hay discrepancias significativas entre predicción y realidad.

### Conversación D — Comparación de dos AUC (Gemini, 26/oct/2026)

**Prompt (literal):**

> Mi regresión logística obtuvo un AUC de 0.78 y mi red neuronal uno de 0.81. ¿Puedo decir que la red neuronal es mejor?

Los valores 0.78 y 0.81 eran hipotéticos; no son nuestros resultados (en prueba, la logística obtuvo 0.7033 y la red base 0.7031 en promedio).

**Fragmentos relevantes de la respuesta:**

> Bootstrapping: Para generar múltiples submuestras y calcular intervalos de confianza, evaluando si los rangos de AUC se solapan o si la diferencia es real.

> Al usar validación cruzada, puedes comparar tanto la media como la desviación estándar del AUC en múltiples iteraciones.

> ¿Justifica un incremento de 0.03 en el AUC la pérdida total de interpretabilidad y el aumento en la complejidad del sistema?

---

## 2. Hallazgos

### Hallazgo 1 — Codificación invertida del objetivo

*Categoría: Código. Gravedad: alta. Herramienta: Gemini. Fecha: 26/oct/2026.*

1. **Prompt utilizado (literal):** el de la Conversación A.
2. **Parte relevante de la respuesta:** la IA define "1 = Buen pagador, 0 = Mal pagador" y su función devuelve 1 solo para `Fully Paid`. Esto contradice la propia pregunta, que pedía predecir quién **incumple**.
3. **Cómo sospechamos que había un error:** al contrastar el código sugerido con el enunciado del trabajo, que establece que "los modelos de riesgo de crédito estiman la probabilidad de ser malo (valor 1), no la de ser bueno", y con la formulación de nuestro propio prompt.
4. **Evidencia:**
   - Enunciado del trabajo, Parte 1, sección "Variable objetivo".
   - Nuestro `src/preprocesamiento.py` (líneas 21 a 26) codifica `Fully Paid → 0` y `Charged Off → 1`.
   - Con la codificación de la IA, el modelo estimaría P(bueno). Se invertirían la corrección del `pos_weight` (3.607 está definido sobre la clase mala), la scorecard (el puntaje sube con menor riesgo, y las odds se definen como buenos:malos) y la PD que entra en E[L] = PD · LGD · EAD, que quedaría como 1 − PD.
5. **Corrección:** `Fully Paid` y su variante → 0; `Charged Off`, `Default`, `Late (31-120 days)` y la variante de `Charged Off` → 1; `Current`, `Issued`, `In Grace Period` y `Late (16-30 days)` → NA, excluidos del entrenamiento.
6. **Lección aprendida:** definir primero el evento de interés (clase positiva = incumplimiento) y revisar que cualquier código externo lo respete, aunque el resto de la respuesta sea razonable.

### Hallazgo 2 — El código descarta categorías que el texto pedía incluir

*Categoría: Código, Datos del mundo real. Gravedad: media. Herramienta: Gemini. Fecha: 26/oct/2026.*

1. **Prompt utilizado (literal):** el de la Conversación A.
2. **Parte relevante de la respuesta:** el texto dice que `Does not meet the credit policy. Status:Charged Off` "debe ir con los malos", pero la lista `estados_validos` no la contiene, así que `isin` la elimina en silencio. Tampoco menciona la variante `…Status:Fully Paid`.
3. **Cómo sospechamos que había un error:** al comparar la lista `estados_validos` con el `value_counts()` de `loan_status` y con la instrucción del propio texto. 
4. **Evidencia:** `resultados/parte1/descriptivo_loan_status.csv` muestra 1,988 créditos `…Status:Fully Paid` y 761 `…Status:Charged Off`. Son 2,749 filas, el **1.02 %** de los 268,530 créditos etiquetados. Nuestro preprocesamiento sí las incluye.
5. **Corrección:** mapear explícitamente las diez categorías de `loan_status`, y verificar con `value_counts()` que ninguna quede sin asignar (a 0, a 1 o a NA).
6. **Lección aprendida:** comparar la lista de filtrado con todas las categorías observadas, y probar que el conteo final cuadra con el total menos las exclusiones declaradas.

**Observación adicional (omisión, no error):** la respuesta llama "estándar de oro" a excluir `Current` y menciona la censura, pero no advierte el sesgo de selección que produce. En `resultados/parte1/descriptivo_excluidos_por_cosecha.csv` la proporción de créditos incluidos baja de 93.8 % (2012-T3) a 1.6 % (2015-T4), y la tasa de incumplimiento observada baja a 9.7 % en esa última cohorte porque los créditos buenos siguen vigentes. Lo analizamos en la Figura 4 del reporte (`blog/figuras/fig16_excluidos_por_cosecha.png`).

### Hallazgo 3 — Recuperación que incluye cargos cobrados antes del incumplimiento

*Categoría: Datos del mundo real. Gravedad: media. Herramienta: Gemini. Fecha: 26/oct/2026.*

1. **Prompt utilizado (literal):** el de la Conversación B.
2. **Parte relevante de la respuesta:** `Recuperación Neta = recoveries − collection_recovery_fee + total_rec_late_fee`.
3. **Cómo sospechamos que había un error:** al leer la definición oficial de cada campo. Dos son posteriores al castigo y uno no. 
4. **Evidencia:** `datasets/LCDataDictionary.xlsx` define:
   - `recoveries`: *post charge off gross recovery*
   - `collection_recovery_fee`: *post charge off collection fee*
   - `total_rec_late_fee`: *Late fees received to date*

   Los cargos por mora se cobran **durante la vida del crédito**, antes del castigo. Sumarlos a una "tasa de recuperación" mezcla flujos anteriores y posteriores al incumplimiento. Además, el numerador queda inconsistente con el denominador: la EAD solo contiene capital, y los cargos nunca formaron parte de ella. El efecto es subestimar la LGD.

   **Efecto medido** sobre los 46,009 castigados del archivo original (`loan.csv`, versión 3), con la misma EAD y el mismo recorte a $[0, 1]$ de `docs/parte2_lgd.md`. 4,004 créditos (8.7 %) tienen `total_rec_late_fee` > 0, con una mediana de 21.84 USD y un máximo de 358.68 USD. Sumar esos cargos a la recuperación cambia la LGD de 4,003 créditos. La LGD media casi no se mueve (0.9296 sin los cargos, 0.9290 con ellos) y la mediana pasa de 0.981 a 0.975. El efecto visible está en los créditos sin recuperación: la proporción con LGD = 1 baja de 46.4 % a 42.9 %. Son 1,575 créditos de los que no se recuperó nada después del castigo, y que con la fórmula de la IA mostrarían una "recuperación" que en realidad son cargos por mora cobrados antes.
5. **Corrección:** `R = recoveries − collection_recovery_fee`, que es la fórmula de `docs/parte2_lgd.md`, sección 2.2.
6. **Lección aprendida:** antes de sumar un campo a una fórmula, leer su definición y confirmar que corresponde al mismo momento y a la misma base que el resto de los términos.

### Hallazgo 4 — `out_prncp` presentado como si fuera el monto original

*Categoría: Datos del mundo real. Gravedad: alta (el error sería silencioso). Herramienta: Gemini. Fecha: 26/oct/2026.*

1. **Prompt utilizado (literal):** el de la Conversación B.
2. **Parte relevante de la respuesta:** "`funded_amnt` (o `out_prncp` al momento del default): El monto original prestado." La respuesta además asigna LGD = 1 cuando EAD ≤ 0 (`np.where(EAD > 0, …, 0)`).
3. **Cómo sospechamos que había un error:** al verificar qué valor toma `out_prncp` en los créditos castigados. 
4. **Evidencia:**
   - El diccionario define `out_prncp` como *Remaining outstanding principal for total amount funded*: saldo pendiente, no monto original.
   - En la tabla de la sección 2.1 de `docs/parte2_lgd.md` (Tabla 12 del reporte), `out_prncp` promedia **0** en los castigados, porque es el saldo a la fecha de corte del archivo.
   - Si alguien usara `out_prncp` como EAD, resultaría EAD = 0 para todos y el código asignaría `recovery_rate = 0`, es decir, **LGD = 1 para todos los créditos**. El modelo aprendería una constante y no habría ningún error visible.
5. **Corrección:** EAD = `funded_amnt − total_rec_prncp`; verificar que EAD > 0 (se cumple en todos los castigados); un crédito con EAD ≤ 0 se excluye, no se le asigna LGD = 1.
6. **Lección aprendida:** revisar la distribución de cada variable antes de usarla y leer su definición exacta. Un valor constante o cero es una señal de alarma.

**Observación adicional (omisión, no error):** la respuesta no menciona ninguna limitación de la LGD. El enunciado exige discutirlas. Nuestro documento incluye la censura a la derecha (la LGD media pasa de 0.911 en 2012 a 0.981 en 2015 y el porcentaje sin recuperación de 25.2 % a 89.0 %) y la ausencia de descuento temporal (`docs/parte2_lgd.md`, sección 2.4).

### Hallazgo 5 — Imprecisiones sobre Brier, normativa y Hosmer-Lemeshow (menor)

*Categoría: Conceptos. Gravedad: menor. Herramienta: Gemini. Fecha: 26/oct/2026.*

1. **Prompt utilizado (literal):** el de la Conversación C.
2. **Parte relevante de la respuesta:** tres afirmaciones, citadas en la Conversación C: calibración "exacta" obligatoria en IFRS 9 y Basilea; Brier "cercano a 0" como medida de precisión y calibración; Hosmer-Lemeshow con p > 0.05 como "meta ideal". 
3. **Cómo sospechamos que había un error:** al comparar la respuesta con nuestros propios resultados, donde un cambio solo de calibración movió el Brier sin mover el AUC, y con las tablas de calibración por deciles. 
4. **Evidencia:**
   - **La idea central es correcta.** La logística con y sin corrección del peso de clase tiene el mismo AUC en prueba (0.7033) y distinto Brier (0.2209 frente a 0.1722). Un AUC no dice nada sobre calibración (`resultados/parte1/metricas_pd.csv`).
   - **Brier.** El puntaje (Brier, 1950) no mide solo calibración: se descompone en confiabilidad, resolución e incertidumbre (Murphy, 1973), y su valor depende de la tasa base. Por eso se compara con p̄(1−p̄) = 0.1886, no con 0. Nuestro Brier de 0.1709 está lejos de 0 y es adecuado. Además, un Brier por debajo de la referencia no garantiza calibración: la red predice en promedio 0.2374 frente a una tasa observada de 0.2522 (subestima 1.5 puntos), y la logística 0.2245 (2.8 puntos) (`resultados/parte1/calibracion_prueba_red.csv` y `calibracion_prueba_logistica.csv`).
   - **Hosmer-Lemeshow.** Con N = 36,378 en prueba, la estadística calculada sobre esas mismas tablas por deciles, con 8 grados de libertad, es 75.9 para la red (p ≈ 3·10⁻¹³) y 306.1 para la logística (p ≈ 2·10⁻⁶¹). Los errores por decil van de 0.4 a 3.1 puntos porcentuales en la red y llegan a 7.1 en la logística. La prueba es la de Hosmer y Lemeshow (1980); con muestras grandes tiene tanta potencia que rechaza ante desviaciones pequeñas y sin importancia práctica (Paul et al., 2013), y la IA no lo advierte.
   - **Normativa.** "Calibración exacta" exagera, y la respuesta junta marcos con objetivos distintos. IFRS 9 pide medir las pérdidas esperadas con un monto insesgado y ponderado por probabilidad, determinado al evaluar un rango de resultados posibles (International Accounting Standards Board [IASB], 2014, párr. 5.5.17). El enfoque IRB de Basilea pide que la PD de cada grado sea un promedio de largo plazo de las tasas de incumplimiento a un año, y deja las exposiciones minoristas, como estos créditos de consumo, a requisitos propios (Basel Committee on Banking Supervision [BCBS], 2006, párr. 447). Además, Basilea exige agregar a las estimaciones un margen de conservadurismo (BCBS, 2006, párr. 451), lo contrario de una calibración "exacta". También confunde provisiones (IFRS 9) con capital (Basilea).
5. **Corrección:** evaluar la calibración con la curva por deciles, el sesgo medio (predicho frente a observado) y el Brier comparado con su referencia; no usar HL como criterio único con N grande; citar la normativa desde su fuente primaria.
6. **Lección aprendida:** una métrica resumen (AUC, Brier o un p-valor) no sustituye mirar cómo se comportan las probabilidades por tramos de riesgo, y las afirmaciones normativas deben tener cita.

### Hallazgo 6 — Comparación de AUC por solapamiento de intervalos y validación cruzada aleatoria (menor)

*Categoría: Conceptos. Gravedad: menor. Herramienta: Gemini. Fecha: 26/oct/2026.*

1. **Prompt utilizado (literal):** el de la Conversación D.
2. **Parte relevante de la respuesta:** propone evaluar "si los rangos de AUC se solapan o si la diferencia es real" y recomienda validación cruzada como remedio a una sola partición. También habla de una "pérdida total de interpretabilidad" de la red.
3. **Cómo sospechamos que había un error:** al contrastar esas recomendaciones con nuestro propio protocolo (30 semillas, partición temporal por fecha de originación y comparación sobre el mismo conjunto de prueba).
4. **Evidencia:**
   - **Solapamiento de intervalos.** Los dos AUC se calculan sobre los mismos créditos, por lo que están fuertemente correlacionados. En una simulación (n = 1,500, semilla fija) con AUC de 0.782 y 0.811, los intervalos de confianza por separado **se solapan** ([0.757, 0.809] y [0.787, 0.835]), pero el bootstrap **pareado** de la diferencia (0.028) da un intervalo de [0.0225, 0.0347], que excluye 0. La correlación entre los dos AUC remuestreados fue 0.97. La comparación correcta es la de la diferencia con una prueba pareada (DeLong et al., 1988) o con bootstrap pareado.
   - **Validación cruzada.** Un k-fold aleatorio mezcla cohortes de originación, algo inadecuado en riesgo de crédito, donde el enunciado exige una partición temporal (validación desde 2014-05 y prueba desde 2014-11 en `src/preprocesamiento.py`). Además, no captura la variabilidad por inicialización de la red: con 30 semillas, la red base promedia un AUC de 0.7031 con desviación 0.0011, y la logística (0.7033) cae dentro de ese rango (`resultados/parte1/semillas_pd.csv`, configuración `base`).
   - **Interpretabilidad.** "Pérdida total" exagera: explicamos la red con gradientes integrados y con importancia por permutación, y derivamos una scorecard a partir de ella. Es cierto que la logística es más directa de interpretar.
5. **Corrección:** comparar los modelos sobre el mismo conjunto de prueba temporal, con DeLong o bootstrap pareado de la diferencia, y repetir el entrenamiento de la red con varias semillas.
6. **Lección aprendida:** que dos intervalos se solapen no implica que la diferencia no sea significativa; con datos pareados se prueba la diferencia, no cada intervalo por separado. Y el método de validación debe respetar la estructura temporal de los datos.

---

## 3. Verificaciones sin error (auditoría del 27/oct/2026)

El 27 de octubre de 2026 hicimos una segunda ronda de prompts, **después** de terminar el desarrollo, para comprobar cuatro puntos técnicos donde sospechábamos que la IA podía fallar. Se hicieron con Gemini. Los prompts de esta ronda fueron redactados con apoyo de Claude (Anthropic). **No los contamos como hallazgos**, porque en lo esencial la IA acertó. Los reportamos para que quede claro lo que verificamos y cómo.

| # | Tema | Prompt (resumen) | Resultado | Evidencia |
|---|---|---|---|---|
| V1 | Corrección de logits tras entrenar con `pos_weight` | Cómo corregir los logits para recuperar probabilidades y cuál es el signo | **Correcto:** `z_corr = z − ln(pos_weight)` | Simulación: intercepto con pesos menos ln(w) = −1.2909, igual al sin pesos (−1.2907). Con los valores guardados, restar ln(3.607) reproduce el puntaje 612.74 de `resultados/parte1/resumen_parte1.json`; sumar daría 538.70 |
| V2 | Odds de referencia de la scorecard | Qué odds elegir con 25 % de incumplimiento, PDO 20 y puntaje base 600 | **Correcto**, con dos imprecisiones menores: da A ≈ 568.31 (el valor exacto es 568.30, por redondear el factor) y afirma que las odds "se calculan directamente" de la tasa y luego que "se eligen" | Cálculo directo: Offset = 600 − (20 / ln 2) · ln 3 = 568.30 |
| V3 | Estadístico KS | Función para calcularlo y qué significa KS = 0.30 | **Correcto:** `max\|TPR − FPR\|` es el KS; coincide con `ks_2samp(malos, buenos)` incluso con empates. La tabla de "qué tan bueno es un KS" es heurística, no estándar | Definición del KS en la Sección 7.7 del reporte (`ks_2samp` sobre las probabilidades de malos y buenos) |
| V4 | Gradientes integrados y completitud | Qué línea base usar, cuántos pasos y cómo verificar la completitud | **Correcto:** la fórmula y la propiedad de completitud coinciden con Sundararajan et al. (2017). Requiere adaptar el código a un solo logit. Trae restos de copiado (títulos de imágenes y un enlace con parámetro de seguimiento) que eliminamos | Revisión del texto frente al artículo original y a `error_completitud_medio` (0.0012) y `error_completitud_max` (0.019) de `resultados/parte1/resumen_parte1.json` |

**Lección de las verificaciones:** que una respuesta sea correcta no exime de comprobarla. Los resultados de V1 a V4 los reprodujimos con código antes de usarlos en el reporte.

---

## Referencias
Basel Committee on Banking Supervision. (2006). *International convergence of capital measurement and capital standards: A revised framework (comprehensive version)*. Bank for International Settlements. https://www.bis.org/publ/bcbs128.pdf

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics, 44*(3), 837–845. https://doi.org/10.2307/2531595

Hosmer, D. W., & Lemeshow, S. (1980). Goodness of fit tests for the multiple logistic regression model. *Communications in Statistics — Theory and Methods, 9*(10), 1043–1069. https://doi.org/10.1080/03610928008827941

International Accounting Standards Board. (2014). *IFRS 9 Financial instruments*. IFRS Foundation.

Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology, 12*(4), 595–600. https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2

Paul, P., Pennell, M. L., & Lemeshow, S. (2013). Standardizing the power of the Hosmer-Lemeshow goodness of fit test in large data sets. *Statistics in Medicine, 32*(1), 67–80. https://doi.org/10.1002/sim.5525

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. En *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 3319–3328). PMLR. https://proceedings.mlr.press/v70/sundararajan17a.html
