# Archivo

Archivos de etapas anteriores del trabajo que **ya no se usan**. Se conservan con su historial para consulta, pero no hacen parte de los resultados, del blog ni de la aplicación publicada, y no se mantienen: algunos pueden no correr desde esta carpeta.

| Archivo | Qué era | Qué lo reemplaza |
|---|---|---|
| `app_v1/app.py`, `app_v1/utils_app.py` | Primera versión de la aplicación web. Calculaba la probabilidad de incumplimiento con una fórmula de coeficientes escritos a mano; no usaba los modelos entrenados. | `app/app_riesgo.py`, que usa los modelos guardados en `weights/` (verificado con `pruebas/probar_app.py`). |
| `scripts/run_demo.py` | Demostración que simulaba las etiquetas y las predicciones con números aleatorios, así que sus métricas no correspondían a ningún modelo y sus parámetros no tenían efecto. | `scripts/demo_sustentacion.py`, que entrena y evalúa de verdad. |
| `scripts/evaluar_30_semillas_clasif.py` | Repetía esa misma simulación con 30 semillas. | `scripts/parte1_busqueda.py`, que entrena 30 veces de verdad. |
| `resultados/semillas_clasificacion.csv` | Salida simulada del script anterior (AUC alrededor de 0.80, que el modelo no tiene). | `resultados/parte1/semillas_pd.csv` (AUC de 0.703 a 0.704). |
| `docs/blog_post_tecnico_(Borrador).md` | Primer borrador del reporte, con números de antes de unificar el preprocesamiento. | `blog/blog_trabajo2.Rmd`. |
| `docs/fundamentacion_matematica.md` | Primera versión de la fundamentación matemática. | Sección 7 del blog, revisada contra el código (por ejemplo, el `weight_decay` de Adam y la corrección del peso de clase). |
| `docs/observaciones_y_sugerencias.md` | Notas de revisión de los notebooks. | Los problemas señalados ya se resolvieron en los scripts y el blog. |
