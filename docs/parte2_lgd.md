# Parte 2: Pérdida dado incumplimiento (LGD)

Notas de trabajo de la Parte 2. Código en [`notebooks/regresion.ipynb`](../notebooks/regresion.ipynb).

## 1. Datos

- Fuente: *Credit Risk Analysis* (Kaggle, `ranadeep/credit-risk-dataset`, versión 3), archivo `loan/loan.csv`.
- 887,379 registros y 74 variables, con créditos originados entre 2007-06 y 2015-12 (`issue_d`).
- Se parte del archivo original, no de `df_procesado.csv`, porque este último elimina por fuga de datos las columnas con las que se construye la LGD. Esas columnas se usan **solo para construir el objetivo**, nunca como variables de entrada.

## 2. Variable objetivo

### 2.1 ¿Qué créditos entran?

| `loan_status` | n | % con `recoveries > 0` | `out_prncp` promedio |
|---|---:|---:|---:|
| Charged Off | 45,248 | 53.45 | 0 |
| Does not meet the credit policy. Status:Charged Off | 761 | 64.39 | 0 |
| Default | 1,219 | 0.00 | 11,595 |
| Late (31-120 days) | 11,591 | 0.00 | 11,690 |

`Default` y `Late (31-120 days)` todavía no se han castigado: no tienen recuperaciones y aún tienen saldo pendiente, así que su LGD **no es observable**. Si se incluyeran, todos saldrían con LGD = 1. El modelo se entrena solo con los **castigados** (46,009 créditos) y luego se aplica a cualquier crédito, porque estima la pérdida *si* llegara a incumplir.

### 2.2 Fórmula

Las definiciones vienen de `LCDataDictionary.xlsx`:

| Campo | Definición (diccionario) | Rol |
|---|---|---|
| `funded_amnt` | *The total amount committed to that loan at that point in time.* | Monto desembolsado |
| `total_rec_prncp` | *Principal received to date.* | Capital pagado antes del castigo |
| `recoveries` | *post charge off gross recovery* | Recuperación bruta |
| `collection_recovery_fee` | *post charge off collection fee* | Costo de cobranza |

$$EAD_i = \text{funded\_amnt}_i - \text{total\_rec\_prncp}_i$$

$$R_i = \text{recoveries}_i - \text{collection\_recovery\_fee}_i$$

$$LGD_i = \min\!\left(1,\ \max\!\left(0,\ 1 - \frac{R_i}{EAD_i}\right)\right)$$

Verificaciones: $EAD > 0$ y `out_prncp = 0` en todos los castigados. Solo 38 casos dan LGD < 0 antes del recorte, porque lo recuperado puede cubrir también intereses y cargos. Ningún caso da LGD > 1.

### 2.3 Distribución

- Media 0.930, mediana 0.981 y desviación estándar 0.098.
- El **46.4 %** de los créditos tiene LGD = 1 (ninguna recuperación).
- Hay **picos**: `recoveries / EAD` se concentra en ~5.7 % y ~14 %, y `collection_recovery_fee / recoveries` en ~1 % y ~18 %. Esto sugiere que la LGD está dominada por **reglas operativas del recobro**, como la venta de cartera a precio fijo o tarifas de agencias, y no por el perfil del cliente. ⚠️ *Hipótesis sin verificar con fuentes.*

| Año de originación | n | LGD media | % sin recuperación |
|---|---:|---:|---:|
| 2007 | 158 | 0.921 | 36.7 |
| 2008 | 496 | 0.931 | 44.2 |
| 2009 | 723 | 0.926 | 34.0 |
| 2010 | 1,753 | 0.925 | 25.6 |
| 2011 | 3,205 | 0.922 | 22.7 |
| 2012 | 8,084 | 0.911 | 25.2 |
| 2013 | 14,836 | 0.918 | 42.9 |
| 2014 | 13,981 | 0.945 | 62.7 |
| 2015 | 2,773 | 0.981 | 89.0 |

### 2.4 Limitaciones

1. **Recobro incompleto (censura a la derecha).** El conjunto es una foto en un momento dado. En las cosechas recientes la cobranza no ha terminado, por lo que la LGD está sobreestimada (ver 2014–2015 en la tabla).
2. **Sin descuento temporal.** El conjunto no trae fechas de cada recuperación, así que no se descuentan al momento del incumplimiento y la pérdida económica real es mayor que la calculada.
3. **EAD solo con capital.** No incluye intereses causados no pagados.
4. **Se excluyen `Default` y `Late`.** Se supone que, al castigarse, se comportarán como los ya castigados.
5. **Costos indirectos.** `collection_recovery_fee` solo cubre el costo de la agencia de cobranza.

## 3. Variables de entrada

- Mismas 29 variables de originación y mismas codificaciones que `procesamiento.ipynb`: ordinales para `grade`, `sub_grade`, `verification_status` y `emp_length`, y one-hot con `drop_first` para las demás categóricas.
- Al eliminar filas con nulos quedan **33,110** créditos, originados entre 2012-08 y 2015-09. Las variables `tot_cur_bal`, `tot_coll_amt` y `total_rev_hi_lim` no existen para las cosechas anteriores.
- `pymnt_plan` y `application_type` son constantes en los castigados y se eliminan. Quedan **41 variables**.
- Escalado a $[-1, 1]$ con mínimo y máximo **de entrenamiento** para no filtrar información de validación y prueba.
- Datos procesados guardados en `datasets/df_lgd_procesado.csv.gz` (sin escalar, con `id`, `issue_d` y `ead`; `ead` se usa solo en la sección 8, no como variable de entrada).

## 4. Partición temporal

Se ordena por `issue_d` y se divide en 70/15/15. Lo más reciente queda en prueba, coherente con un modelo que se aplica a créditos futuros.

| Conjunto | n | Desde | Hasta | LGD media |
|---|---:|---|---|---:|
| Entrenamiento | 22,372 | 2012-08 | 2014-04 | 0.919 |
| Validación | 5,511 | 2014-05 | 2014-09 | 0.949 |
| Prueba | 5,227 | 2014-10 | 2015-09 | 0.970 |

La LGD media sube en validación y prueba por la censura de la sección 2.4. Esto penaliza a cualquier modelo entrenado con cosechas más antiguas.

## 5. Modelos

### 5.1 Regresión lineal (referencia)

Mínimos cuadrados ordinarios (`sklearn.linear_model.LinearRegression`) con 42 parámetros (41 coeficientes y el intercepto). En prueba ninguna predicción cae fuera de $[0, 1]$, pero el modelo no lo garantiza.

### 5.2 Perceptrón multicapa

- **Arquitectura:** igual a la de clasificación (`Linear → BatchNorm → activación → Dropout` por capa oculta). El número de capas, las neuronas y la activación se eligen en la búsqueda (sección 6).
- **Activación de salida: sigmoide.** La LGD vive en $[0, 1]$; la sigmoide garantiza $\hat{y} \in (0, 1)$ sin recortar.
- **Pérdida.** Se compararon dos opciones dentro de la búsqueda:
  - **Entropía cruzada binaria con objetivo continuo** (cuasi-verosimilitud Bernoulli, *fractional logit*; Papke y Wooldridge, 1996). Para $y_i \in [0, 1]$:
    $$\ell(\theta) = \sum_i \left[ y_i \log \mu_i + (1 - y_i)\log(1 - \mu_i) \right], \quad \mu_i = \sigma(f_\theta(x_i))$$
    Su valor esperado se maximiza en $\mu_i = E[y_i \mid x_i]$, así que estima consistentemente la media condicional aunque $y$ no sea binaria.
  - **Error cuadrático medio** sobre la salida sigmoide. Corresponde a máxima verosimilitud bajo un modelo gaussiano $y_i \sim \mathcal{N}(\mu_i, \sigma^2)$, y también se minimiza en $\mu_i = E[y_i \mid x_i]$.

  Ambas estiman la **media** condicional, que es lo que pide $E[L] = PD \cdot LGD \cdot EAD$.
- **Regularización:** *dropout*, penalización $L_2$ (`weight_decay` en Adam) y parada temprana con paciencia de 10 épocas, restaurando los pesos de la mejor época en validación.
- **Optimización:** Adam, `ReduceLROnPlateau` (factor 0.5, paciencia 3) y máximo 100 épocas.

> Con BCE o MSE la pérdida mínima no es 0: es la entropía o la varianza de $y$. Por eso la pérdida de validación puede quedar **por debajo** de la de entrenamiento, ya que la LGD de validación está más concentrada cerca de 1. No es un error.

## 6. Búsqueda de hiperparámetros

**Método: búsqueda aleatoria** (Bergstra y Bengio, 2012). Con el mismo presupuesto explora más valores distintos de cada hiperparámetro que una rejilla, lo que conviene cuando solo unos pocos importan. Se sortearon 30 configuraciones de este espacio (6,912 combinaciones posibles):

| Hiperparámetro | Valores |
|---|---|
| Capas ocultas | `[32]`, `[64]`, `[64, 32]`, `[128, 64]`, `[64, 32, 16]`, `[128, 64, 32]` |
| Activación | ReLU, tanh, Leaky ReLU |
| *Dropout* | 0, 0.1, 0.2, 0.3 |
| `weight_decay` ($L_2$) | 0, 1e-5, 1e-4, 1e-3 |
| Tasa de aprendizaje | 1e-4, 3e-4, 1e-3, 3e-3 |
| Tamaño de lote | 256, 512, 1024 |
| Pérdida | BCE, MSE |

- **Criterio de selección:** menor RMSE en validación, igual para todas las configuraciones. Como cada configuración usa su propia pérdida, no se pueden comparar por la pérdida.
- **Configuración ganadora:** `[64, 32, 16]`, tanh, *dropout* 0.2, `weight_decay` 1e-5, `lr` 3e-4, lote de 512 y pérdida MSE. Tiene **5,537 parámetros**.
- 8 de las 10 mejores configuraciones usan MSE.
- Resultados completos en `resultados/busqueda_lgd.csv`. La configuración final está en `resultados/config_final_lgd.json`.

### Presupuesto de cómputo (CPU)

| Modelo | Parámetros | Configuraciones | Tiempo de búsqueda | Tiempo de 30 semillas |
|---|---:|---:|---:|---:|
| Media de entrenamiento | 1 | 1 | 0 s | — |
| Regresión lineal | 42 | 1 | 0.05 s | — (determinista) |
| MLP | 5,537 | 30 | 5.7 min | 13.6 min |

## 7. Resultados

### 7.1 MLP final con 30 semillas (semillas 0 a 29)

| | MAE val | RMSE val | MAE prueba | RMSE prueba |
|---|---:|---:|---:|---:|
| Media | 0.0655 | 0.0775 | 0.0618 | 0.0796 |
| Desv. estándar | 0.0007 | 0.0002 | 0.0012 | 0.0005 |
| Mejor | 0.0643 | 0.0771 | 0.0598 | 0.0787 |
| Peor | 0.0675 | 0.0779 | 0.0653 | 0.0811 |

La variabilidad entre semillas es pequeña frente a las diferencias entre modelos. El detalle está en `resultados/semillas_lgd.csv`.

### 7.2 Comparación (MLP con semilla 56)

| Modelo | Conjunto | MAE | RMSE | RMSE sin sesgo | Sesgo | Spearman |
|---|---|---:|---:|---:|---:|---:|
| Media de entrenamiento | val | 0.0770 | 0.0834 | 0.0776 | −0.0306 | — |
| | prueba | 0.0814 | 0.0919 | 0.0765 | −0.0509 | — |
| Regresión lineal | entrenamiento | 0.0731 | 0.0910 | 0.0910 | 0.0000 | **0.199** |
| | val | 0.0716 | 0.0801 | 0.0771 | −0.0215 | **0.154** |
| | prueba | 0.0740 | 0.0878 | 0.0773 | −0.0418 | **0.140** |
| MLP | entrenamiento | 0.0744 | 0.0948 | 0.0913 | +0.0255 | 0.169 |
| | val | **0.0657** | **0.0772** | 0.0771 | −0.0017 | 0.152 |
| | prueba | **0.0630** | **0.0800** | 0.0766 | −0.0232 | 0.106 |

- **Sesgo** = media de la predicción − media observada.
- **RMSE sin sesgo** = RMSE después de quitar a ambas series su media, es decir, el error que no se debe al nivel.
- **Spearman** = correlación de rangos entre predicción y LGD observada, que mide qué tan bien el modelo ordena los créditos.

### 7.3 Diagnóstico: la ventaja del MLP es de nivel, no de información

En MAE y RMSE el MLP gana: en prueba su RMSE es 0.0800 frente a 0.0878 de la regresión lineal. Pero:

1. **Quitando el sesgo de nivel, los dos modelos empatan** (RMSE sin sesgo de 0.0766 frente a 0.0773 en prueba).
2. **La regresión lineal ordena mejor** los créditos en los tres conjuntos (Spearman de 0.140 frente a 0.106 en prueba).
3. **El MLP está sesgado hacia arriba en su propio conjunto de entrenamiento** (+0.026). La figura de LGD por trimestre lo muestra: su curva queda por encima de la observada durante todo el periodo de entrenamiento y es casi plana.
4. **La explicación está en la censura.** La LGD observada sube en validación y prueba porque el recobro de las cosechas recientes no ha terminado. La búsqueda y la parada temprana eligen según el RMSE de validación, así que **premian los modelos cuyo nivel se acerca al de validación**. No premian a los que entienden mejor al cliente. El MLP se beneficia porque, al ser más flexible, su nivel depende más de los hiperparámetros y de la época de parada.
5. **Ningún modelo reproduce la distribución observada.** En prueba, el 81 % de los créditos tiene LGD = 1 exacta. Las predicciones se concentran en un rango estrecho (0.90–0.96 la lineal, 0.93–0.96 el MLP), con desviación estándar de 0.018 y 0.010 frente a 0.077 observada. Es el comportamiento esperado de un estimador de la media condicional cuando las variables explican poco.

**Conclusión preliminar:** con variables de originación, la LGD es casi impredecible a nivel individual (Spearman de 0.10 a 0.20). Lo que un modelo gana en MAE o RMSE en prueba se debe sobre todo a acertar el **nivel** de una LGD inflada por la censura, no a discriminar entre clientes.

## 8. Exposición al incumplimiento (EAD) y pérdida esperada

**Definición de EAD al originar.** En el momento de la solicitud no se sabe cuándo incumplirá el cliente, así que se estima la fracción del monto que suele quedar pendiente al castigo, por plazo, con los castigados de entrenamiento:

$$\widehat{EAD}_i = \text{funded\_amnt}_i \cdot \bar{f}_{\text{plazo}(i)}, \qquad \bar{f}_{\text{plazo}} = \text{media}\left(\frac{\text{funded\_amnt} - \text{total\_rec\_prncp}}{\text{funded\_amnt}}\right)$$

| Plazo | $\bar{f}$ |
|---|---:|
| 36 meses | 0.670 |
| 60 meses | 0.846 |

Los créditos a 60 meses amortizan más despacio, así que al incumplir deben una fracción mayor. Para un crédito ya vigente, la exposición actual sería `out_prncp`, pero eso no es coherente con un modelo de originación.

**Prueba de coherencia** en el conjunto de prueba. Todos estos créditos incumplieron, así que se usa $PD = 1$:

| | Total (USD) | Promedio por crédito (USD) |
|---|---:|---:|
| Pérdida observada ($LGD \cdot EAD$ reales) | 71,196,347 | 13,621 |
| Pérdida estimada (MLP y $\widehat{EAD}$) | 57,736,333 | 11,046 |

La subestimación del 19 % se explica por la LGD de prueba inflada por la censura y por usar un factor de EAD promedio.

**Pérdida esperada:** $E[L]_i = PD_i \cdot \widehat{LGD}_i \cdot \widehat{EAD}_i$. La función `perdida_esperada` del notebook está lista. Falta conectarla con la **PD del modelo de clasificación**, que hoy usa un preprocesamiento distinto (escalado con todos los datos, sin `id`).

## 9. Pendientes

- [x] Búsqueda de hiperparámetros con presupuesto reportado.
- [x] 30 semillas de la configuración final.
- [x] Comparación de distribuciones.
- [x] Definición de EAD.
- [ ] Decidir cómo tratar la censura (sección 7.3). Opciones:
  - (a) mantener la partición actual y reportar el sesgo;
  - (b) limitar la muestra a cosechas con recobro maduro;
  - (c) seleccionar hiperparámetros con una métrica sin sesgo, como el RMSE sin sesgo o Spearman.
- [ ] Calcular $E[L]$ con la PD de clasificación (requiere preprocesamiento compartido y conservar `id`).
- [ ] Mover `CONFIG` a un archivo de configuración y lanzar el experimento con un solo comando.
- [ ] Verificar con fuentes la hipótesis de las ventas de cartera (2.3).

## Referencias

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html ⚠️ *Verificar antes de citar.*

Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. [https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1](<https://doi.org/10.1002/(SICI)1099-1255(199611)11:6%3C619::AID-JAE418%3E3.0.CO;2-1>)
