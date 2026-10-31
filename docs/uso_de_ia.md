# Uso de inteligencia artificial

## 1. Cómo usamos la IA

El enunciado permite y espera el uso de herramientas de IA, pero advierte que se equivocan con total seguridad. Por eso las usamos como apoyo y no como fuente final: todo lo que generaron (código, fórmulas, cifras y referencias) lo revisamos contra los datos, el código o la fuente original antes de dejarlo en el trabajo. Los errores más importantes que encontramos están en la cacería de la alucinación (`docs/caceria_alucinaciones.md`).

| Herramienta | Quién la usó | Para qué |
|---|---|---|
| Claude (Anthropic) | Juan Felipe Miranda | Revisar el repositorio contra el enunciado, revisar la cacería (rutas, referencias y DOI) y completar sus cifras, corregir y rediseñar la aplicación web, y unir la cacería y esta sección al reporte. Trabajó con acceso al repositorio local: ejecutó el código, probó la app y subió los cambios por instrucción de Juan Felipe |
| Gemini (Google) | Ana Sofía Gómez | Las cuatro conversaciones de la cacería (26 de octubre) y la ronda de verificación (27 de octubre, con prompts redactados con apoyo de Claude) |
| **[PENDIENTE]** | Tomás Gómez | **[PENDIENTE]** |
| **[PENDIENTE]** | Sebastián Ocampo | **[PENDIENTE]** |
| **[PENDIENTE]** | Juan José Baena | **[PENDIENTE]** |

## 2. Principales prompts y su impacto

### Tomás Gómez Cardona

| Prompt (literal) | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| **[PENDIENTE: prompt literal y herramienta]** | **[PENDIENTE]** | **[PENDIENTE]** |

### Sebastián Ocampo Galvis

| Prompt (literal) | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| **[PENDIENTE: prompt literal y herramienta]** | **[PENDIENTE]** | **[PENDIENTE]** |

### Juan Felipe Miranda Arciniegas

| Prompt (literal) | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| "hola, lee el trabajo 2 de RNA" y luego "ya lo tenemos hecho, por ahid eebe estar el repo clonado, trabajo 2" (a Claude) | Leyó el enunciado y el repositorio, y listó lo pendiente: las secciones 8 y 9 del blog, la publicación en RPubs, los videos y los enlaces de la app. También señaló problemas del primer borrador de la cacería: números que no coincidían con los resultados, una referencia con el año equivocado y afirmaciones sin evidencia. | Orientó lo que faltaba por hacer y la revisión de la cacería. |
| "esta es caceria que hizo mi compañera. podrias revisar lo de los links a ver si está bien y hay unos espacios que dice completar, si puede hacerlos también" (a Claude) | Revisó cada ruta y cada referencia. Encontró que se citaba un script que no existía y una ruta equivocada, verificó los DOI y agregó una cita adecuada para el comportamiento de Hosmer-Lemeshow con muestras grandes. Calculó con el archivo original de Kaggle, ejecutando código, el efecto del hallazgo 3. Corrigió la cita de Basilea, porque el párrafo citado exceptúa las exposiciones minoristas. No pudo confirmar en la fuente el número del párrafo 447. | Las cifras del hallazgo 3, las referencias y la corrección sobre Basilea de la cacería salen de esta revisión. Las menciones al script inexistente se quitaron. |
| "vamos a corregir un error de ui ux, mira: cuando el valor es muy grande, no se ve bien "Acá cuando hay un monto muy grande no se ven los número, pero es ya cuando son montos de un millón pa arriba"" (a Claude, con una captura de la app) | Encontró que las tarjetas cortaban la cifra con "…" porque la letra tenía tamaño fijo. Hizo que la cifra se ajuste al ancho de la tarjeta y que, desde un millón, se muestre en millones con los valores exactos en la nota. Lo probó con el test de Streamlit y con capturas en tres anchos de pantalla. | Commit `4b5edc3` de la app publicada. |
| "yo se que pedi colores pastel, pero no, esto se ve muy mal, no me gusta, cambiale los colores" (a Claude) | Cambió la paleta pastel por una de verde de billete sobre papel, con una escala de riesgo verde, ámbar y ladrillo. Actualizó las notas de la app que nombraban los colores viejos y el tema de Streamlit. | Pull request #8 de la app publicada. |
| "no, ya esta deployeada, revisa cual es la pagina" (a Claude) | Afirmó que el enlace de la aplicación "no está en ninguna parte del repo". Era falso: estaba en el README, en el blog y en la guía de ejecución. Lo detectó la misma IA al revisar el historial antes de fusionar un cambio, y lo corrigió. | Ninguno en los entregables, pero confirma que también hay que verificar lo que la IA dice sobre el propio repositorio. |

### Ana Sofía Gómez Zapata

| Prompt | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| Cuatro preguntas a Gemini sobre la codificación de `loan_status`, la fórmula de la LGD, el AUC frente a la calibración y la comparación de dos AUC. Los prompts literales están en la sección 1 de `docs/caceria_alucinaciones.md`. | Dio respuestas razonables en general, con seis errores: el objetivo codificado al revés, categorías descartadas en silencio, cargos por mora sumados a la recuperación, `out_prncp` presentado como el monto original, e imprecisiones sobre la calibración y sobre cómo comparar dos AUC. | Son los seis hallazgos de la cacería. El trabajo usa las versiones corregidas, por ejemplo la codificación de `src/preprocesamiento.py` y la LGD de `docs/parte2_lgd.md`. |
| Ronda de verificación del 27 de octubre: cuatro preguntas técnicas a Gemini sobre la corrección de logits, las odds de referencia de la scorecard, el KS y los gradientes integrados, con prompts redactados con apoyo de Claude (resumen en la sección 3 de la cacería). | Acertó en lo esencial, con imprecisiones menores. | Reprodujimos cada resultado con código antes de usarlo en el reporte. |

### Juan José Baena García

| Prompt (literal) | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| **[PENDIENTE: prompt literal y herramienta]** | **[PENDIENTE]** | **[PENDIENTE]** |

## 3. Discusión: qué aportó la IA y qué no

La IA nos ahorró tiempo sobre todo en revisar el trabajo contra el enunciado, escribir y depurar código y verificar referencias. Fue poco confiable justo en lo que el trabajo exige con precisión: la codificación del objetivo, el significado de los campos de Lending Club y algunos conceptos estadísticos, como muestra la cacería. También se equivocó sobre nuestro propio repositorio, al afirmar que el enlace de la aplicación no estaba en él.

De ahí salen tres prácticas que seguimos: verificar cada afirmación contra los datos, el código o la fuente original; obtener las cifras ejecutando código, nunca de la memoria de la IA; y revisar el historial de cambios antes de fusionar lo que la IA propone.
