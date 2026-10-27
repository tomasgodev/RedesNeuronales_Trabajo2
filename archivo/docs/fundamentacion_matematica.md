# Fundamentación Matemática Formal
**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02  
**Trabajo 2:** Redes neuronales para datos tabulares (¿me prestas?)  
**Profesor:** Juan David Ospina Arango | **Monitor:** Andrés Mauricio Zapata Rincón  
**Universidad Nacional de Colombia — Facultad de Minas**

---

## 1. La Neurona Artificial y el Perceptrón Multicapa (MLP)

### 1.1. Neurona Artificial
Una neurona artificial procesa un vector de entrada $\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$ mediante dos fases consecutivas:
1. **Combinación afín ponderada:** Calcula la suma ponderada de las entradas más un término de sesgo o polarización (*bias*) $b \in \mathbb{R}$:
   $$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^d w_j x_j + b$$
   donde $\mathbf{w} \in \mathbb{R}^d$ es el vector de pesos sinápticos.
2. **Transformación no lineal:** Aplica una función de activación $\phi: \mathbb{R} \to \mathbb{R}$ para obtener la activación de salida $a$:
   $$a = \phi(z) = \phi(\mathbf{w}^T \mathbf{x} + b)$$

### 1.2. Perceptrón Multicapa (*Multilayer Perceptron*, MLP)
Un perceptrón multicapa con $L$ capas procesa la información de forma secuencial hacia adelante (*feedforward*). Sea $\mathbf{h}^{(l)} \in \mathbb{R}^{d_l}$ el vector de activaciones de la capa $l$, donde $\mathbf{h}^{(0)} = \mathbf{x} \in \mathbb{R}^{d_0}$ es la entrada:
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}, \quad l = 1, 2, \dots, L$$
$$\mathbf{h}^{(l)} = \phi^{(l)}\left(\mathbf{z}^{(l)}\right)$$
donde $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ es la matriz de pesos de la capa $l$, $\mathbf{b}^{(l)} \in \mathbb{R}^{d_l}$ es el vector de sesgos, y $\phi^{(l)}(\cdot)$ es la función de activación aplicada componente a componente.

---

## 2. Funciones de Activación Utilizadas

### 2.1. Unidad Lineal Rectificada (ReLU)
Definida como:
$$\phi(z) = \max(0, z)$$
Su derivada para $z \neq 0$ es:
$$\phi'(z) = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z < 0 \end{cases}$$
*Propiedades:* Resuelve el desvanecimiento del gradiente (*vanishing gradient*) para valores positivos y propicia representaciones dispersas (*sparse representations*).

### 2.2. Leaky ReLU
Modificación con pendiente no nula $\alpha \in (0, 1)$ (típicamente $\alpha = 0.01$):
$$\phi(z) = \begin{cases} z & \text{si } z \ge 0 \\ \alpha z & \text{si } z < 0 \end{cases}$$
Su derivada es:
$$\phi'(z) = \begin{cases} 1 & \text{si } z \ge 0 \\ \alpha & \text{si } z < 0 \end{cases}$$
*Propiedades:* Previene el fenómeno de "neuronas muertas" (*dying ReLU*) manteniendo flujo de gradiente cuando $z < 0$.

### 2.3. Tangente Hiperbólica (Tanh)
Definida como:
$$\phi(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
Su derivada se expresa en términos de la propia función:
$$\phi'(z) = 1 - \tanh^2(z) = 1 - \phi(z)^2$$
*Propiedades:* Salida acotada en $(-1, 1)$, con media centrada en cero, lo que facilita el acondicionamiento de los gradientes en las capas iniciales.

### 2.4. Función Sigmoide Logística
Definida como:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Su derivada analítica es:
$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \left(\frac{1}{1 + e^{-z}}\right)\left(1 - \frac{1}{1 + e^{-z}}\right) = \sigma(z)(1 - \sigma(z))$$
*Propiedades:* Mapea la recta real $\mathbb{R}$ en el intervalo abierto $(0, 1)$. En la capa de salida de clasificación modela la probabilidad condicional de impago ($PD$). En la capa de salida de regresión de LGD, garantiza que la predicción esté acotada naturalmente en $[0, 1]$ sin necesidad de truncamiento artificial.

---

## 3. Derivación de Funciones de Pérdida por Máxima Verosimilitud (MLE)

### 3.1. Entropía Cruzada Binaria (*Binary Cross-Entropy*, BCE) — Clasificación de Incumplimiento
Sea un conjunto de entrenamiento con $N$ observaciones independientes e idénticamente distribuidas (i.i.d.) $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, donde la etiqueta $y_i \in \{0, 1\}$ indica buen pagador ($y_i=0$) o mal pagador ($y_i=1$).

Modelamos la variable aleatoria $Y_i \mid \mathbf{X}_i = \mathbf{x}_i$ mediante una distribución de Bernoulli con parámetro $p_i = P(Y_i = 1 \mid \mathbf{x}_i; \mathbf{\theta})$:
$$P(Y_i = y_i \mid \mathbf{x}_i; \mathbf{\theta}) = p_i^{y_i} (1 - p_i)^{1 - y_i}, \quad p_i = \sigma(f(\mathbf{x}_i; \mathbf{\theta}))$$
donde $f(\mathbf{x}_i; \mathbf{\theta})$ es el logit producido por la red neuronal parametrizada por $\mathbf{\theta}$.

La función de verosimilitud conjunta para las $N$ observaciones es:
$$\mathcal{L}(\mathbf{\theta}) = \prod_{i=1}^N P(Y_i = y_i \mid \mathbf{x}_i; \mathbf{\theta}) = \prod_{i=1}^N p_i^{y_i} (1 - p_i)^{1 - y_i}$$
Aplicando el logaritmo natural para obtener la log-verosimilitud:
$$\ln \mathcal{L}(\mathbf{\theta}) = \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$
El estimador de máxima verosimilitud $\mathbf{\theta}_{\text{MLE}} = \arg\max_\mathbf{\theta} \ln \mathcal{L}(\mathbf{\theta})$ equivale a minimizar la log-verosimilitud negativa promediada, definiendo la función de pérdida de entropía cruzada binaria:
$$\mathcal{L}_{\text{BCE}}(\mathbf{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

#### Incorporación del Peso de Clase (`pos_weight`) para Desbalance
Dado que en crédito los incumplimientos son la clase minoritaria ($y=1$), se introduce un factor de ponderación $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$:
$$\mathcal{L}_{\text{wBCE}}(\mathbf{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ w_{\text{pos}} \cdot y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$
Para recuperar las probabilidades calibradas no sesgadas $P(Y=1 \mid \mathbf{x})$ a partir del logit predicho $\hat{z}$, se aplica la corrección inversa del sesgo inducido por el peso:
$$p_{\text{calibrada}} = \sigma\left(\hat{z} - \ln(w_{\text{pos}})\right)$$

---

### 3.2. Error Cuadrático Medio (*Mean Squared Error*, MSE) — Regresión de LGD
Sea la variable continua $y_i = LGD_i \in [0, 1]$. Asumimos que la observación observada $y_i$ se relaciona con la función de la red $\hat{y}_i = f(\mathbf{x}_i; \mathbf{\theta})$ mediante un término de error aditivo gaussiano:
$$y_i = f(\mathbf{x}_i; \mathbf{\theta}) + \epsilon_i, \quad \epsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$
Por lo tanto, la densidad condicional es:
$$p(y_i \mid \mathbf{x}_i; \mathbf{\theta}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - f(\mathbf{x}_i; \mathbf{\theta}))^2}{2\sigma^2} \right)$$
La log-verosimilitud conjunta para las $N$ muestras es:
$$\ln \mathcal{L}(\mathbf{\theta}, \sigma^2) = -\frac{N}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N \left( y_i - f(\mathbf{x}_i; \mathbf{\theta}) \right)^2$$
Al maximizar con respecto a $\mathbf{\theta}$ manteniendo $\sigma^2$ constante, el primer término desaparece y el segundo equivale exactamente a minimizar la suma del error cuadrático medio:
$$\mathcal{L}_{\text{MSE}}(\mathbf{\theta}) = \frac{1}{N} \sum_{i=1}^N \left( y_i - f(\mathbf{x}_i; \mathbf{\theta}) \right)^2$$
*Nota teórica (Papke & Wooldridge, 1996):* Al estimar con salida sigmoide, el MSE converge asintóticamente a la media condicional $E[y \mid \mathbf{x}]$, que es precisamente el valor requerido para calcular la pérdida esperada $E[L] = PD \cdot LGD \cdot EAD$.

---

## 4. Retropropagación (*Backpropagation*) y Optimizador Adam

### 4.1. Algoritmo de Retropropagación Matricial
Para una muestra con entrada $\mathbf{x}$ y objetivo $y$, sea $\mathcal{L}$ la función de costo escalar.

**1. Capa de salida ($L$):**  
Se define el vector de error de la capa de salida $\mathbf{\delta}^{(L)} \in \mathbb{R}^{d_L}$:
$$\mathbf{\delta}^{(L)} = \nabla_{\mathbf{z}^{(L)}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \odot \phi'^{(L)}(\mathbf{z}^{(L)})$$
donde $\odot$ denota el producto de Hadamard (multiplicación elemento a elemento). Para la entropía cruzada binaria con activación sigmoide ($\mathbf{a}^{(L)} = p$), la combinación algebraica se simplifica elegantemente:
$$\mathbf{\delta}^{(L)} = p - y$$

**2. Propagación hacia atrás del error (capas $l = L-1, L-2, \dots, 1$):**  
Aplicando la regla de la cadena multivariable:
$$\mathbf{\delta}^{(l)} = \nabla_{\mathbf{z}^{(l)}} \mathcal{L} = \left[ (\mathbf{W}^{(l+1)})^T \mathbf{\delta}^{(l+1)} \right] \odot \phi'^{(l)}(\mathbf{z}^{(l)})$$

**3. Gradientes con respecto a los parámetros:**  
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \mathbf{\delta}^{(l)} (\mathbf{h}^{(l-1)})^T \in \mathbb{R}^{d_l \times d_{l-1}}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \mathbf{\delta}^{(l)} \in \mathbb{R}^{d_l}$$

---

### 4.2. Ecuaciones de Actualización del Optimizador Adam
Adam (*Adaptive Moment Estimation*, Kingma & Ba, 2014) combina las ventajas del momento de primer orden y de tasas de aprendizaje adaptativas por coordenada basadas en momentos de segundo orden.

Sea $\mathbf{g}_t = \nabla_\mathbf{\theta} \mathcal{L}_t(\mathbf{\theta}_t)$ el gradiente evaluado en el minibatch en el paso $t$.
1. **Actualización del momento de primer orden (media exponencial móvil):**
   $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$
2. **Actualización del momento de segundo orden no centrado (varianza no centrada):**
   $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t \odot \mathbf{g}_t$$
3. **Corrección de sesgo para inicializaciones en cero:**
   $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
4. **Actualización de parámetros:**
   $$\mathbf{\theta}_{t+1} = \mathbf{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$
*Hiperparámetros estándar:* $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

---

## 5. Formulación de Técnicas de Regularización

### 5.1. Penalización $L_2$ (*Weight Decay*)
Modifica la función objetivo añadiendo la norma de Frobenius al cuadrado de las matrices de pesos:
$$\mathcal{J}(\mathbf{\theta}) = \mathcal{L}(\mathbf{\theta}) + \frac{\lambda}{2} \sum_{l=1}^L \|\mathbf{W}^{(l)}\|_F^2$$
El gradiente modificado es:
$$\nabla_{\mathbf{W}^{(l)}} \mathcal{J} = \nabla_{\mathbf{W}^{(l)}} \mathcal{L} + \lambda \mathbf{W}^{(l)}$$
En cada iteración, los pesos se multiplican por un factor de decaimiento $(1 - \eta \lambda) < 1$, contrayendo los pesos grandes y evitando que la red ajuste fluctuaciones espurias del ruido muestral.

### 5.2. Dropout Invertido (*Inverted Dropout*)
Durante la fase de entrenamiento, cada neurona $j$ en la capa $l$ se desactiva de forma aleatoria con probabilidad $p_{\text{drop}} \in [0, 1)$ generando una máscara de Bernoulli:
$$r_j^{(l)} \sim \text{Bernoulli}(1 - p_{\text{drop}})$$
$$\tilde{\mathbf{h}}^{(l)} = \frac{1}{1 - p_{\text{drop}}} \left( \mathbf{r}^{(l)} \odot \mathbf{h}^{(l)} \right)$$
El factor de escalamiento $\frac{1}{1 - p_{\text{drop}}}$ asegura que la esperanza matemática de la activación durante el entrenamiento sea idéntica a la activación determinista durante la evaluación:
$$\mathbb{E}[\tilde{h}_j^{(l)}] = \frac{1}{1 - p_{\text{drop}}} (1 - p_{\text{drop}}) h_j^{(l)} = h_j^{(l)}$$
Esto elimina la necesidad de reescalar los pesos durante la inferencia en producción.

### 5.3. Parada Temprana (*Early Stopping*)
Monitorea la métrica de generalización en el conjunto de validación independiente $\mathcal{M}_{\text{val}}^{(e)}$ al final de cada época $e$.
Se mantiene un contador de paciencia $k$. Si $\mathcal{M}_{\text{val}}^{(e)}$ no supera el mejor valor histórico registrado $\mathcal{M}^*$:
$$k \leftarrow k + 1$$
Si $k \ge \text{patience}$, el entrenamiento se interrumpe y se restauran los parámetros del punto óptimo:
$$\mathbf{\theta}^* = \arg\max_{\mathbf{\theta}^{(e)}} \mathcal{M}_{\text{val}}^{(e)}$$

---

## 6. Métricas de Evaluación

### 6.1. Área Bajo la Curva ROC (AUC-ROC) e Índice de Gini
La curva ROC grafica la Tasa de Verdaderos Positivos ($TPR$) contra la Tasa de Falsos Positivos ($FPR$) a través de todos los posibles umbrales de decisión $t \in [0, 1]$:
$$TPR(t) = P(\hat{p} > t \mid Y = 1), \quad FPR(t) = P(\hat{p} > t \mid Y = 0)$$
El área bajo la curva se expresa como una probabilidad de concordancia bivariada:
$$AUC = P(\hat{p}_{\text{malo}} > \hat{p}_{\text{bueno}})$$
El **Índice de Gini** (o coeficiente de precisión de Somers) mide la capacidad de ordenamiento relativo del modelo sobre la diagonal del clasificador aleatorio ($AUC = 0.5$):
$$\text{Gini} = 2 \cdot AUC - 1 \quad \in [-1, 1]$$

### 6.2. Estadístico Kolmogorov-Smirnov ($KS$)
En evaluación de riesgo crediticio, el estadístico $KS$ mide la máxima separación vertical entre las funciones de distribución acumulada empíricas de las probabilidades predichas para buenos ($F_0$) y malos ($F_1$):
$$F_0(s) = P(\hat{p} \le s \mid Y = 0), \quad F_1(s) = P(\hat{p} \le s \mid Y = 1)$$
$$KS = \sup_{s \in [0, 1]} \left| F_0(s) - F_1(s) \right| \quad \in [0, 1]$$
Un $KS > 0.30$ indica una capacidad discriminante satisfactoria para modelos de originación crediticia de consumo.

### 6.3. Puntaje de Brier (*Brier Score*, $BS$)
Evalúa conjuntamente la discriminación y la **calibración probabilística**:
$$BS = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 \quad \in [0, 1]$$
Se compara frente al **Brier Score de Referencia**, que corresponde a un modelo ingenuo que predice la tasa base poblacional constante $\bar{y} = \frac{1}{N} \sum y_i$:
$$BS_{\text{ref}} = \bar{y}(1 - \bar{y})$$
Un modelo calibrado y con valor informativo debe satisfacer estrictamente $BS < BS_{\text{ref}}$.

---

## 7. Derivación Matemática Formal de la Scorecard

### 7.1. Relación entre Probabilidad, Odds y Logit
Sea $p = P(Y = 1 \mid \mathbf{x})$ la probabilidad de incumplimiento estimada por el modelo. En la práctica crediticia internacional, las *odds* se definen como la razón entre la probabilidad de ser un cliente solvente ("bueno") y la probabilidad de impago ("malo"):
$$\text{Odds} = \frac{P(Y = 0 \mid \mathbf{x})}{P(Y = 1 \mid \mathbf{x})} = \frac{1 - p}{p}$$
Tomando el logaritmo natural:
$$\ln(\text{Odds}) = \ln\left(\frac{1 - p}{p}\right) = -\ln\left(\frac{p}{1 - p}\right) = -\text{logit}(p)$$

### 7.2. Transformación Lineal a Escala de Puntos
La scorecard modela el puntaje crediticio como una transformación afín del log-odds:
$$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln(\text{Odds})$$
Sustituyendo $\text{Odds} = \frac{1-p}{p}$:
$$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{1-p}{p}\right) = \text{Offset} - \text{Factor} \cdot \ln\left(\frac{p}{1-p}\right)$$

### 7.3. Calibración de Parámetros: Score Base, Odds Base y PDO
Para fijar de manera única los dos grados de libertad ($\text{Factor}$ y $\text{Offset}$), la institución bancaria especifica:
1. Un **Puntaje Base** ($\text{Score}_0$) asociado a unas **Odds de Referencia** ($\text{Odds}_0$):
   $$\text{Score}_0 = \text{Offset} + \text{Factor} \cdot \ln(\text{Odds}_0) \quad \text{--- (Ecuación 1)}$$
2. Los **Puntos para Duplicar las Odds** ($\text{PDO}$), que establece que cuando las probabilidades de pago se duplican ($2 \cdot \text{Odds}_0$), el puntaje debe incrementarse exactamente en $\text{PDO}$ puntos:
   $$\text{Score}_0 + \text{PDO} = \text{Offset} + \text{Factor} \cdot \ln(2 \cdot \text{Odds}_0) \quad \text{--- (Ecuación 2)}$$

Restando la Ecuación 1 de la Ecuación 2:
$$\text{PDO} = \text{Factor} \cdot \left[ \ln(2 \cdot \text{Odds}_0) - \ln(\text{Odds}_0) \right] = \text{Factor} \cdot \ln\left(\frac{2 \cdot \text{Odds}_0}{\text{Odds}_0}\right) = \text{Factor} \cdot \ln(2)$$
Despejando directamente el factor:
$$\mathbf{\text{Factor} = \frac{\text{PDO}}{\ln(2)}}$$
Sustituyendo el $\text{Factor}$ en la Ecuación 1 para despejar el $\text{Offset}$:
$$\mathbf{\text{Offset} = \text{Score}_0 - \text{Factor} \cdot \ln(\text{Odds}_0)}$$

### 7.4. Calibración Concreta del Trabajo
Utilizando los parámetros del caso de negocio fijados para el portafolio de consumo:
- $\text{Score}_0 = 600$ puntos
- $\text{Odds}_0 = 50:1$ (50 buenos por cada 1 malo, equivalente a una tasa base de $p_0 \approx 1.96\%$)
- $\text{PDO} = 20$ puntos

Calculamos:
$$\text{Factor} = \frac{20}{\ln(2)} \approx \frac{20}{0.693147} \approx \mathbf{28.8539}$$
$$\text{Offset} = 600 - 28.8539 \cdot \ln(50) = 600 - 28.8539 \cdot 3.912023 \approx 600 - 112.8773 \approx \mathbf{487.1227}$$

La fórmula explícita implementada en el motor de scoring y en la aplicación web es:
$$\mathbf{\text{Score}(p) = 487.1227 - 28.8539 \cdot \ln\left(\frac{p}{1-p}\right)}$$
Para mantener compatibilidad con las escalas estándar de la industria (escala FICO/VantageScore), el puntaje final se acota en el intervalo:
$$\text{Score}_{\text{final}} = \min(850, \max(300, \lfloor \text{Score}(p) + 0.5 \rfloor))$$
