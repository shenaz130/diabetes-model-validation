# Diabetes Hospital Readmission Prediction | Model Validation

**Data Science & Machine Learning | Master's in Artificial Intelligence**

## Project Overview

This project investigates hospital readmission prediction for patients with diabetes using machine learning.

The objective is to classify hospital encounters into three categories:

- **NO:** No readmission
- **<30:** Readmission within 30 days
- **>30:** Readmission after 30 days

The project focuses on model comparison, feature selection, class imbalance, and reliable model validation.

## Dataset

The analysis uses a diabetes hospital encounter dataset containing 101,766 records.

The project applies patient-grouped splitting to prevent encounters belonging to the same patient from appearing in both training and test sets.

## Methodology

### 1. Data Preparation
- Categorical feature encoding
- Patient-grouped train/test splitting
- Training-only preprocessing to reduce data leakage

### 2. Feature Selection

Three feature tiers were evaluated:

- **Tier 1:** Top 20 correlated features
- **Tier 2:** Top features plus selected clinical and admission-related categories
- **Tier 3:** All available encoded features

### 3. Machine Learning Models

Two algorithms were compared:

- Multinomial Logistic Regression
- LightGBM

Each algorithm was evaluated using class weighting and SMOTENC to address class imbalance.

### 4. Model Validation

Five-fold Stratified Group Cross-Validation was used, keeping patient groups separated.

Performance was assessed using:

- Macro F1-score
- Balanced Accuracy
- Macro AUC
- Recall for readmission within 30 days

## Final Model Results

**Selected model:** LightGBM with class weighting, using Tier 3 features.

| Evaluation Metric | Hold-out Test Result |
|---|---:|
| Macro F1-score | 0.444 |
| Balanced Accuracy | 0.465 |
| Macro AUC | 0.664 |
| Overall Accuracy | 0.502 |
| Recall — Within 30 Days | 0.393 |

## Key Findings

LightGBM with class weighting achieved the strongest overall cross-validation performance among the evaluated configurations.

Using all encoded features produced the best overall validation results.

SMOTENC did not consistently improve performance compared with class weighting.

The final model demonstrated moderate predictive ability, with early readmission remaining the most difficult category to identify.

## Limitations and Future Improvements

The model identified approximately 39.3% of readmissions occurring within 30 days in the held-out test set.

Distinguishing early readmissions from later readmissions remains challenging.

Future work could explore binary classification, further feature engineering, hyperparameter optimization, and additional validation.

The model is an academic experiment and is not intended for clinical decision-making.

## Technologies Used

Python, Google Colab, Pandas, NumPy, Scikit-learn, LightGBM, Imbalanced-learn, SciPy, and Matplotlib.

## Project Notebook

[View the Model Validation Notebook](Diabetes_ModelValidation.ipynb)

## Reproducibility

This notebook currently relies on a separate exploratory data analysis notebook and files stored in Google Drive. The preprocessing notebook and permitted dataset access instructions must be provided before the full workflow can be reproduced independently.
