# Clasificación de textos ODS 🌍

Clasificación automática de textos según los **Objetivos de Desarrollo Sostenible (ODS)** usando machine learning — un aporte desde el ML a la agenda 2030.

Microproyecto del curso **ML No Supervisado** · Maestría en Inteligencia Artificial, Universidad de los Andes.

## 🎯 Objetivo

Asignar a cada texto la categoría ODS correspondiente, probando distintos modelos y eligiendo el de mejor rendimiento.

## ⚙️ Enfoque

1. Carga y exploración de datos
2. Pruebas para elegir el modelo
3. Tuning del mejor modelo
4. Modelo final y explicación de su funcionamiento

## 📁 Contenido

| Archivo | Descripción |
|---|---|
| `microproyecto_2_MNS.ipynb` | Notebook del microproyecto |
| `streamlit-app/` | App web (Streamlit) para clasificar textos en tiempo real |
| `Train_textosODS.xlsx` | Datos de entrenamiento (textos etiquetados) |
| `Microproyecto2.pdf` | Enunciado del microproyecto |

> El modelo entrenado final (~380 MB) no se incluye por su tamaño; la app lo carga desde `pipeline_calibrado.pkl`.

## 🛠️ Stack

Python · scikit-learn · NLP · Streamlit · NLTK

---
*Maestría en Inteligencia Artificial — Universidad de los Andes*
