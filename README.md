# Sleep Stress Detection System 😴🧠

## 📌 Project Overview
Stress is a silent killer, and sleep quality is one of its strongest indicators. This project builds a Machine Learning system designed to analyze physiological data collected from **Smart Pillows** (SaYoPillow) to classify human stress levels.

The goal is to provide early intervention and personalized health insights by predicting stress on a scale of 5 levels (0-4).

## 📂 The Dataset
* **Dataset Name:** Human Stress Detection in and through Sleep (SaYoPillow).
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep)
* **Features:** The dataset includes physiological parameters such as:
    * Snoring Range
    * Respiration Rate
    * Body Temperature
    * Limb Movement
    * Blood Oxygen Levels
    * Sleeping Hours
    * Heart Rate

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
* Analyzed correlations between physiological metrics and stress levels.
* Visualized sleep patterns using Seaborn & Matplotlib.

### 2. Machine Learning Modeling
* We trained and evaluated multiple supervised learning algorithms.
* **Champion Model:** Support Vector Machine (SVM).
* The SVM model showed the best performance in separating the distinct stress classes.

### 3. Model Serialization (Ready for Deployment)
* The trained model (`best_model`) and the data scaler (`scaler`) were saved using **Joblib**.
* **Output Files:** * `SaYoPillow_Best_Model.pkl`
    * `SaYoPillow_Scaler.pkl`

## 🚀 Deployment (Streamlit)
This project is designed to be integrated with **Streamlit** for a user-friendly web interface. The saved `.pkl` files are used to make real-time predictions based on user input.

## 🧰 Tech Stack
* **Python**
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (SVM)
* **Deployment:** Streamlit, Joblib

---
*Created by: Mohamed AbdelAziz*
