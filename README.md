# ¿Me prestas? Redes neuronales para riesgo de crédito

Trabajo 2 del curso *Redes neuronales y algoritmos bioinspirados* (Universidad Nacional de Colombia, Facultad de Minas, 2026-02). Con créditos de consumo de Lending Club estimamos la probabilidad de incumplimiento (Parte 1, clasificación) y la pérdida dado incumplimiento (Parte 2, regresión) con perceptrones multicapa, los comparamos con modelos de referencia, construimos una scorecard y calculamos la pérdida esperada.

**Equipo:** Tomás Gómez Cardona, Sebastián Ocampo Galvis, Juan Felipe Miranda Arciniegas, Ana Sofía Gómez Zapata y Juan José Baena García.

- **Aplicación web:** https://redesneuronalestrabajo2-y8uxecxjqz6frn9ex3ypta.streamlit.app/
- **Reporte técnico (blog):** https://rpubs.com/juan0911miranda/1472227 (la fuente es [`blog/blog_trabajo2.Rmd`](blog/blog_trabajo2.Rmd)).
- **Video promocional:** https://canva.link/xrpw4q9vvf19v24
- **Videos de contribución individual:** https://canva.link/9a2n3la84fpnyn0

## Inicio rápido

```bash
python3.14 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt   # versiones exactas (requirements.txt es la lista sin versiones)

python scripts/demo_sustentacion.py    # demostración completa en ~15 s
streamlit run app/app_riesgo.py        # aplicación web en local
```

Todos los comandos, sus parámetros y tiempos están en [`docs/como_ejecutar.md`](docs/como_ejecutar.md).

## Estructura

```
├── datasets/        Datos procesados (creditos_procesados.csv.gz) y diccionario de variables
├── src/             Código compartido: preprocesamiento, modelos y validación de parámetros
├── notebooks/       Exploración y desarrollo de los modelos (ver tabla abajo)
├── scripts/         Experimentos con un solo comando (Parte 1, Parte 2, pérdida esperada, demo)
├── configs/         Parámetros de los experimentos (parte1.json, parte2.json)
├── weights/         Modelos entrenados: la app y las figuras no necesitan reentrenar
├── resultados/      Tablas de resultados (ver abajo)
├── app/             Aplicación web (app_riesgo.py) y su motor (motor_riesgo.py)
├── blog/            Reporte técnico (blog_trabajo2.Rmd) y sus figuras
├── pruebas/         Casos de prueba de la app y de robustez de los scripts
├── docs/            Guía de ejecución, notas de la Parte 2, cacería de la alucinación y uso de IA
└── archivo/         Archivos de etapas anteriores que ya no se usan (ver su README)
```

**`resultados/`:** la raíz tiene las salidas de `notebooks/regresion.ipynb`, que el notebook lee de ahí. Las subcarpetas `parte1/`, `parte2/` y `perdida_esperada/` tienen las salidas de los scripts.

## Datos procesados

`datasets/creditos_procesados.csv.gz` contiene los datos que usan todos los modelos. Se regenera desde el archivo original de Kaggle con:

```bash
python -m src.preprocesamiento
```

El preprocesamiento está en [`src/preprocesamiento.py`](src/preprocesamiento.py) y es común a la Parte 1 (PD) y la Parte 2 (LGD):

- `incumple`: variable objetivo de clasificación (codificación del enunciado).
- `lgd` y `ead`: solo para créditos castigados (Parte 2).
- Variables de entrada seleccionadas en [`notebooks/procesamiento.ipynb`](notebooks/procesamiento.ipynb).
- Partición temporal común: validación desde 2014-05 y prueba desde 2014-11.

## Notebooks

| Notebook | Contenido |
|---|---|
| `procesamiento.ipynb` | Análisis de nulos y selección de variables (ANOVA, Pearson, V de Cramér). |
| `entrenamiento_validacion.ipynb` | Exploración de la partición temporal. |
| `regresion_logistica.ipynb` | Modelo de referencia de la Parte 1. |
| `clasificacion.ipynb` | Red neuronal de la Parte 1. |
| `regresion.ipynb` | Parte 2: LGD, regresión lineal, red neuronal y pérdida esperada ([notas](docs/parte2_lgd.md)). |

Los scripts de `scripts/` reproducen los resultados de los notebooks con un solo comando y sin internet, usando sus mismas funciones (copiadas sin cambios en `src/modelos.py`).

## Pruebas

```bash
python pruebas/probar_app.py        # la app usa los modelos entrenados; la interfaz no se cae con valores extremos
python pruebas/probar_robustez.py   # 100 casos con parámetros normales, extremos e inválidos
```
