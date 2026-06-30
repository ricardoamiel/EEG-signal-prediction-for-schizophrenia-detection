# Clasificación de pacientes con esquizofrenia mediante señales EEG usando modelos clásicos, CNN y Transformers

* **Autor:** Prof. Cristian López Del Alamo
* **Curso:** Series Temporales
* **Institución:** UTEC
* **Fecha:** 9 de junio de 2026

---

## Resumen

Este proyecto propone clasificar pacientes con esquizofrenia y controles sanos a partir de señales EEG de la base pública ASZED, disponible en Zenodo. Se compararán dos enfoques: primero, modelos clásicos de aprendizaje automático entrenados con características extraídas mediante `tsfresh`; segundo, modelos de aprendizaje profundo entrenados con imágenes tiempo-frecuencia generadas a partir de FFT, STFT/espectrogramas o transformada wavelet. 

Se evaluarán modelos como SVM, XGBoost, Random Forest, CNN, ResNet/EfficientNet y Transformers visuales o específicos para EEG. El desempeño se comparará usando métricas de clasificación binaria y una separación por sujeto para evitar fuga de información.

---

## 1. Base de datos

La base de datos seleccionada es **ASZED – African Schizophrenia EEG Dataset**, disponible en: [https://zenodo.org/records/14178398](https://zenodo.org/records/14178398)

ASZED contiene registros EEG de 76 pacientes con esquizofrenia y 77 controles sanos. Los datos incluyen diferentes paradigmas experimentales, como reposo, tareas cognitivas, *auditory oddball/mismatch negativity* (MMN) y *auditory steady-state response* (ASSR) con estimulación auditiva de 40 Hz. El archivo principal es `ASZED-153.zip`, con un tamaño aproximado de 207.7 MB.

### Tabla 1: Resumen del dataset ASZED

| Característica | Descripción |
| :--- | :--- |
| **Repositorio** | Zenodo |
| **Pacientes** | 76 sujetos con esquizofrenia |
| **Controles** | 77 sujetos sanos |
| **Paradigmas** | Reposo, tareas cognitivas, MMN, ASSR 40 Hz |
| **Archivo** | ASZED-153.zip |
| **Clases** | 0: control sano; 1: esquizofrenia |

---

## 2. Objetivo

Desarrollar y comparar modelos de *machine learning* y *deep learning* para clasificar sujetos con esquizofrenia y controles sanos a partir de señales EEG, usando tanto características temporales/frecuenciales como imágenes tiempo-frecuencia.

---

## 3. Metodología

El flujo general del proyecto se divide en cinco etapas: preprocesamiento, generación de imágenes, extracción de características, entrenamiento de modelos y comparación experimental.

### 3.1. Preprocesamiento EEG
Las señales EEG serán procesadas usando herramientas como MNE-Python, NumPy, SciPy y scikit-learn. La tubería propuesta incluye:

1. Carga y organización de los registros por sujeto, clase y paradigma experimental.
2. Selección inicial de una condición, preferentemente EEG en reposo, para reducir variabilidad.
3. Filtrado pasa banda, por ejemplo de 0.5 a 45 Hz.
4. Filtro notch en 50 o 60 Hz, según la frecuencia de línea correspondiente.
5. Segmentación de las señales en ventanas de duración fija, por ejemplo 2, 5 o 10 segundos.
6. Normalización por canal mediante z-score.
7. División de datos en entrenamiento, validación y prueba a nivel de sujeto.

> **Nota importante:** La separación por sujeto es esencial: todas las ventanas de un mismo sujeto deben pertenecer a un único conjunto. Esto evita fuga de información y una sobreestimación artificial del rendimiento.

### 3.2. Transformación de señales en imágenes
Para los modelos de *deep learning*, las señales temporales se transformarán en imágenes tiempo-frecuencia. Se evaluarán tres representaciones principales:

* **FFT:** permite obtener potencia espectral por canal y banda de frecuencia. Se analizaráa frecuencia de línea correspondiente.
5. Segmentación de las señales en ventanas de duración fija, por ejemplo 2, 5 o 10 segundos.
6. Normalización por canal mediante z-score.
7. División de datos en entrenamiento, validación y prueba a nivel de sujeto.n bandas delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) y gamma baja (30–45 Hz).
* **STFT/espectrogramas:** permite representar la evolución temporal del contenido frecuencial de la señal.
* **Wavelets:** permite generar escalogramas, útiles para señales no estacionarias como EEG.

Estas imágenes serán redimensionadas y normalizadas para ser usadas como entrada de CNN y Transformers.

### 3.3. Modelos de deep learning
Se entrenarán modelos de clasificación usando las imágenes generadas. Los modelos propuestos son:

* **CNN propia:** arquitectura base entrenada desde cero.
* **ResNet o EfficientNet:** modelos convolucionales con *transfer learning*.
* **Vision Transformer (ViT) o Swin Transformer:** modelos basados en atención aplicados a espectrogramas o escalogramas.
* **EEG-Conformer:** alternativa específica para EEG que combina convoluciones y mecanismos de atención.

El objetivo de esta etapa es evaluar si las representaciones tiempo-frecuencia combinadas con arquitecturas profundas mejoran la clasificación frente a métodos clásicos.

### 3.4. Extracción de características y modelos clásicos
En paralelo, se extraerán características directamente de las señales EEG usando `tsfresh`. Esta herramienta permite calcular automáticamente características estadísticas, temporales y frecuenciales, como media, varianza, energía, entropía, autocorrelación, curtosis, asimetría y descriptores espectrales.

Después de la extracción se aplicará selección de características para reducir dimensionalidad. Con las características seleccionadas se entrenarán los siguientes modelos:

* **SVM:** modelo robusto para espacios de características de alta dimensión.
* **XGBoost:** modelo basado en *boosting*, útil para datos tabulares y selección implícita de variables.
* **Random Forest:** modelo de referencia basado en árboles.
* **Regresión logística:** línea base interpretable para clasificación binaria.

---

## 4. Comparación experimental

Se compararán dos enfoques principales:

### Tabla 2: Comparación de enfoques

| Machine learning clásico | Deep learning |
| :--- | :--- |
| EEG temporal $\rightarrow$ preprocesamiento $\rightarrow$ características con `tsfresh` $\rightarrow$ selección de características $\rightarrow$ SVM/XGBoost/Random Forest | EEG temporal $\rightarrow$ preprocesamiento $\rightarrow$ FFT/STFT/Wavelet $\rightarrow$ imágenes tiempo-frecuencia $\rightarrow$ CNN/ResNet/EfficientNet/Transformer |

Las métricas de evaluación serán *accuracy*, *precision*, *recall* o sensibilidad, *specificity*, *F1-score*, *ROC-AUC*, *balanced accuracy* y matriz de confusión. La métrica principal recomendada será **F1-score** o **ROC-AUC**, ya que ofrecen una evaluación más completa que el *accuracy*, especialmente si existe desbalance o diferencias en la dificultad de detectar cada clase.

---

## 5. Resultado esperado

Se espera obtener una comparación cuantitativa entre modelos clásicos y modelos de *deep learning* para la clasificación de esquizofrenia usando EEG. El proyecto permitirá determinar si las imágenes tiempo-frecuencia analizadas con CNN o Transformers superan a los modelos clásicos basados en características extraídas con `tsfresh`. Además, permitirá definir una tubería reproducible para el procesamiento de señales EEG y la evaluación de modelos de clasificación clínica.