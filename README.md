# SDG Text Classification 🌍

Automatic classification of texts into **Sustainable Development Goals (SDGs)** using machine learning, with a **Streamlit web app** for real-time predictions.

Microproject for the **Unsupervised ML** course · MSc in Artificial Intelligence, Universidad de los Andes.

## 🎯 Objective

Assign each text its corresponding SDG category, testing several models and tuning the best one.

## ⚙️ Approach

1. Data loading and exploration
2. Model selection experiments
3. Hyperparameter tuning
4. Final model + explanation of how it works

## 📁 Contents

| File | Description |
|---|---|
| `microproyecto_2_MNS.ipynb` | Project notebook |
| `streamlit-app/` | Web app (Streamlit) for real-time classification |
| `Train_textosODS.xlsx` | Labeled training texts |

## 📊 Results

Best model (tuned pipeline): **88.0% accuracy** and **0.85 macro F1** on the test set (1,932 samples).

> The trained model (~380 MB) is not included; the app loads it from `pipeline_calibrado.pkl`.

## 🛠️ Stack

Python · scikit-learn · NLP · Streamlit · NLTK

---
*MSc in Artificial Intelligence — Universidad de los Andes*
