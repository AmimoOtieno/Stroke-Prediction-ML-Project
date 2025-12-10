Stroke Prediction Machine Learning Project

This project applies end-to-end data analysis and machine learning techniques to the Stroke Prediction Dataset.
It includes exploratory analysis, data preprocessing, classification, regression, and clustering to extract insights and develop predictive models.

🔍 Project Overview

The project covers the full data science workflow:

1️⃣ Descriptive Analysis

Summary statistics

Distribution plots (histograms, boxplots, pie charts)

Correlation matrix

Insights on demographic & clinical variables

2️⃣ Data Preparation

Dropping noisy / irrelevant attributes

Outlier detection using IQR

Handling missing values (incl. BMI prediction using Decision Tree Regressor)

Label Encoding

Class balancing using SMOTE

Train–test split

Standardization

3️⃣ Classification Models

The following machine learning algorithms were used to classify whether a patient is likely to have a stroke:

Model	Best Accuracy	AUC
k-Nearest Neighbour	0.98	0.977
Naïve Bayes	0.79	0.787
Decision Tree	0.96	0.955
XGBoost	0.97	0.991

Best Model:
✔️ XGBoost — highest AUC and most robust clinical performance
Alternative:
✔️ kNN — performs strongly with simple interpretability

4️⃣ Regression (Predicting BMI)

Models used:

Ordinary Least Squares (OLS)

Decision Tree Regressor

Best Model:
✔️ Decision Tree Regressor — higher R², lower RMSE & MAE

Also used to predict missing BMI values.

5️⃣ Clustering

Implemented two clustering approaches:

Method	Silhouette Score	Optimal Clusters
K-Means	0.18	2
Hierarchical Clustering	0.17	2

Cluster insights:

Cluster 1: Older, higher risk (hypertension, heart disease, stroke)

Cluster 2: Younger, healthier population

📁 Repository Structure
```
stroke-prediction-ml-project/
│
├── notebooks/
│   └── stroke_analysis.ipynb
│
├── report/
│   └── Stroke Prediction Machine Learning Project.pdf
│
├── src/
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   └── data_preprocessing.py
│
└── README.md
```
🚀 How to Run
Install dependencies
```
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```
Open the notebook
```
jupyter notebook notebooks/stroke_analysis.ipynb
```
📌 Key Skills Demonstrated

Exploratory Data Analysis (EDA)

Data cleaning & preprocessing

Feature engineering

Class balancing using SMOTE

Classification, regression & clustering

Train/test pipeline development

Model evaluation (Accuracy, AUC, RMSE, MAE, R², Silhouette score)

Visualisation using Matplotlib & Seaborn

Interpretation of ML outputs in a real-world health context

📈 Results Summary

XGBoost provided the best overall classification performance

Decision Tree Regressor outperformed linear regression for BMI prediction

Clustering revealed two distinct risk groups in the population

The dataset shows non-linear patterns, favouring tree-based models

🛠️ Technologies Used

Python

NumPy

Pandas

Scikit-learn

XGBoost

Matplotlib

Seaborn

Jupyter Notebook
