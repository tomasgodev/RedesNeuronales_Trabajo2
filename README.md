# RedesNeuronales_Trabajo2

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

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
| `regresion_logistica.ipynb` | Modelo de referencia de la Parte 1. |
| `clasificacion.ipynb` | Red neuronal de la Parte 1. |
| `regresion.ipynb` | Parte 2: LGD, regresión lineal, red neuronal y pérdida esperada ([notas](docs/parte2_lgd.md)). |
