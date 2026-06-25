# Telco Customer Churn Prediction

Predicting customer churn for a telecom company using **XGBoost** with SMOTE for handling class imbalance.

## Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (Yes / No)
- **Class imbalance:** ~84% No Churn vs ~16% Churn

## Project Pipeline

| Step | Description |
|------|-------------|
| **Preprocessing** | Encoding categoricals, handling missing values in `TotalCharges` |
| **SMOTE** | Oversampling minority class to fix class imbalance |
| **XGBoost** | Training gradient boosting classifier |
| **Evaluation** | F1-Score, AUC-ROC, Confusion Matrix, ROC Curve |

## Results

| Metric | Score |
|--------|-------|
| **AUC-ROC** | 0.843 |
| **Churn Recall** | 0.71 |
| **Churn F1-Score** | 0.64 |
| **Overall Accuracy** | 79% |

> **Why SMOTE?** In churn prediction, missing an actual churner (False Negative) is costlier than a false alarm. SMOTE improved Churn Recall from 0.54 → 0.71, making the model more business-relevant.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost

