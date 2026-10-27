# Cómo ejecutar el trabajo

Comandos para reproducir los resultados de `scripts/`, `app/` y `blog/`. Todos se corren desde la raíz del repositorio y con el entorno activado. Los tiempos son de un MacBook con Apple M1, en CPU.

## 1. Entorno

```bash
python3.14 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt # versiones exactas; requirements.txt es la lista original sin versiones
```

## 2. Datos

`datasets/creditos_procesados.csv.gz` ya está en el repositorio, así que nada de lo que sigue necesita internet. Para regenerarlo desde el archivo original de Kaggle (la primera vez lo descarga con `kagglehub`):

```bash
python -m src.preprocesamiento
```

Verificado el 27 de octubre de 2026: con la versión 3 del conjunto, este comando produce un archivo idéntico al del repositorio.

## 3. Experimentos

| Comando | Qué hace | Tiempo |
|---|---|---|
| `python scripts/demo_sustentacion.py` | **Demostración en vivo.** Entrena la red de la Parte 1, la compara con la logística y calcula la scorecard y la pérdida esperada. | 14 s |
| `python scripts/parte1_evaluacion.py` | Evalúa la logística y la red guardada, construye la scorecard e interpreta los modelos. Genera `blog/figuras/fig09` a `fig14`. | 10 s |
| `python scripts/parte1_busqueda.py --demo` | Búsqueda de hiperparámetros y semillas en versión corta. | 10 s |
| `python scripts/parte1_busqueda.py` | Búsqueda aleatoria (30 configuraciones más la base) y 30 semillas de la configuración final y de la base. | ≈ 35 min |
| `python scripts/perdida_esperada.py` | Pérdida esperada $E[L] = PD \cdot LGD \cdot EAD$ en prueba con los modelos guardados. | 5 s |
| `python scripts/parte1_descriptivo.py` | Hipótesis H1 a H4 y créditos excluidos por cosecha. Necesita el archivo original (sección 2). | 10 s |
| `python scripts/parte1_seleccion_entrenamiento.py` | Repite la selección de variables de `notebooks/procesamiento.ipynb` solo con entrenamiento, para medir la fuga. Necesita el archivo original (sección 2). | 8 s |
| `python scripts/parte2_lgd.py` | Parte 2 (LGD). Reproduce `notebooks/regresion.ipynb` sin internet: compara con los modelos guardados, calcula la EAD y la pérdida con PD = 1, y reentrena la configuración final (resultados idénticos al notebook). Genera `blog/figuras/fig06` a `fig08`. | 13 s |
| `python scripts/parte2_lgd.py --demo` | Lo anterior más una búsqueda y semillas cortas. | 17 s |
| `python scripts/parte2_lgd.py --busqueda` | Lo anterior más la búsqueda aleatoria (30) y las 30 semillas. Escribe en `resultados/parte2/` y no toca los resultados del notebook. | ≈ 8 min |
| `python scripts/parte2_lgd.py --descriptivo` | Agrega las estadísticas de la LGD sobre todos los castigados y `blog/figuras/fig05`. Necesita el archivo original (sección 2). | 20 s |

Los parámetros están en `configs/parte1.json` y `configs/parte2.json`, y también se pueden cambiar desde la línea de comandos, sin editar el código. Ejemplos para la modificación en vivo:

```bash
python scripts/demo_sustentacion.py --lr 0.01 --dropout 0.5
python scripts/demo_sustentacion.py --capas 128 64 --activacion tanh --epocas 10
python scripts/demo_sustentacion.py --sin_peso_clase
python scripts/demo_sustentacion.py --pdo 40 --puntaje_base 700 --odds_base 20
python scripts/parte1_evaluacion.py --pdo 25 --puntaje_base 650
python scripts/parte2_lgd.py --perdida bce --lr 0.003 --capas 32
python scripts/demo_sustentacion.py --help
```

Semillas: 56 para la búsqueda, la regresión logística usa `random_state=42`, y las corridas independientes usan las semillas 0 a 29.

Reproducibilidad entre computadores: `scripts/parte2_lgd.py --busqueda` corrido en macOS (Apple M1) reproduce los resultados que `notebooks/regresion.ipynb` obtuvo en Windows. La configuración ganadora y las 30 semillas de la red final son idénticas (diferencias menores a $10^{-7}$). En 5 de las 30 configuraciones de la búsqueda el RMSE de validación difiere hasta en 0.0006, por diferencias numéricas de PyTorch entre plataformas.

## 4. Aplicación web

```bash
streamlit run app/app_riesgo.py
```

Usa los modelos guardados en `weights/`; no reentrena. `app/opciones_app.json` (categorías, valores por defecto y rangos) se regenera con `python scripts/preparar_app.py`, que necesita el archivo original.

La app está publicada en https://redesneuronalestrabajo2-y8uxecxjqz6frn9ex3ypta.streamlit.app/ (desplegada desde la cuenta del dueño del repositorio, con la rama `main`, así que se actualiza sola con cada cambio).

### Publicarla en Streamlit Community Cloud

1. Entrar a https://share.streamlit.io con una cuenta de GitHub que tenga acceso al repositorio.
2. Crear una app con el repositorio `tomasgodev/RedesNeuronales_Trabajo2`, la rama `main` y el archivo principal `app/app_riesgo.py`.
3. En **Advanced settings**, escoger Python 3.12.
4. Poner la URL pública de la app en el encabezado del blog. Cuando existan, poner en `app/app_riesgo.py` los enlaces al reporte (`ENLACE_REPORTE`) y al video (`ENLACE_VIDEO`).

La plataforma instala las dependencias de `app/requirements.txt`, que tiene prioridad sobre el `requirements.txt` de la raíz porque está junto al archivo principal. Ese archivo instala torch solo para CPU, sin las librerías de CUDA. Se verificó que todas las versiones existen para Linux con Python 3.12 y 3.14.

## 5. Blog

```bash
Rscript -e 'rmarkdown::render("blog/blog_trabajo2.Rmd")'
```

Necesita R con `rmarkdown` y pandoc (viene con RStudio).

## Notas

- Los modelos de los notebooks están guardados en `weights/`, y los scripts los cargan sin reentrenar. `src/modelos.py` tiene copias exactas de las clases y funciones de los notebooks, porque un notebook no se puede importar desde un script.
- `scripts/run_demo.py`, `scripts/evaluar_30_semillas_clasif.py` y `resultados/semillas_clasificacion.csv` generan o contienen métricas simuladas con números aleatorios, no con el modelo. Los resultados de los modelos están en `resultados/parte1/`, `resultados/semillas_pd.csv` y `resultados/perdida_esperada/`.
- `codificar()` de `src/preprocesamiento.py` no sirve tal cual para codificar un solo solicitante: con una sola fila, `get_dummies(drop_first=True)` elimina la categoría presente. `app/motor_riesgo.py` explica cómo se evita.
