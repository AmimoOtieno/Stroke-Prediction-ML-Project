# Stroke-Prediction-ML-Project
Machine learning classification, regression, and clustering analysis on a stroke prediction dataset.

This project applies end-to-end data analysis and machine learning techniques on the Stroke Prediction Dataset, including descriptive analysis, data preprocessing, classification, regression, and clustering.

It was completed as part of the Programming for Data Analysis & AI coursework.

🔍 Project Overview

The project includes:

1️⃣ Descriptive Analysis

Summary statistics

Histograms, boxplots, correlation matrix

Categorical distribution visualisations

2️⃣ Data Preparation

Dropping noisy attributes

Outlier detection using IQR

Handling missing values

Label Encoding

SMOTE Oversampling

Train–test split

Standardization

3️⃣ Classification Models
Model	Best Accuracy	AUC
k-Nearest Neighbour	0.98	0.977
Naïve Bayes	0.79	0.787
Decision Tree	0.96	0.955
XGBoost	0.97	0.991

Best Model: XGBoost (highest AUC and balanced performance)

4️⃣ Regression (Predicting BMI)

Models used:

OLS Linear Regression

Decision Tree Regressor

Best Model: Decision Tree Regressor (Higher R², lower RMSE & MAE)

5️⃣ Clustering

Algorithms used:

K-Means (k=2 using Calinski-Harabasz)

Hierarchical Clustering

Silhouette scores:

K-Means: 0.18

Hierarchical: 0.17

Two clusters emerged:

Cluster 1: Older, high-risk (hypertension, heart disease, stroke)

## 📁 Repository Structure

stroke-prediction-ml-project/
│
├── notebooks/
│   └── stroke_analysis.ipynb
│
├── report/
│   └── Stroke Analysis Report.docx
│
├── src/
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   └── data_preprocessing.py
│
└── README.md


## 🚀 How to Run

### Install dependencies:
pip install numpy pandas scikit-learn xgboost matplotlib seaborn

### Open the notebook:
jupyter notebook notebooks/stroke_analysis.ipynb

## 📌 Key Skills Demonstrated
- Exploratory Data Analysis (EDA)
- Data cleaning & preprocessing
- Classification, regression & clustering models
- Feature engineering
- SMOTE balancing technique
- Evaluation metrics (Accuracy, AUC, RMSE, R², Silhouette score)
- Visualisation using Matplotlib & Seaborn



Cluster 2: Younger, healthier
