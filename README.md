# House Price Predictor

End-to-end machine learning pipeline for predicting housing prices using structured tabular data.

Built with Python, pandas, NumPy, and scikit-learn.

---

## Overview

This project implements a full supervised regression workflow for predicting housing prices.  
It includes:

- Data preprocessing (missing value handling + one-hot encoding)
- Train/test split
- 5-fold cross-validation
- Hyperparameter tuning using GridSearchCV
- Model evaluation using RMSE and R²
- Feature importance visualization
- Model persistence for reproducible inference

---
## Dataset

This project uses the Ames Housing dataset, a structured tabular dataset containing residential home sales data in Ames, Iowa.

- 2,930 observations
- 82 features (numeric and categorical)
- Target variable: `SalePrice`

The dataset includes detailed property characteristics such as lot size, neighborhood, overall quality, living area, year built, garage features, and sale condition.

Identifier columns (`Order`, `PID`) are excluded from modeling.

---

## 🛠 Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib

---

## Model Performance

### Baseline Comparison

| Model              | Hold-out RMSE | Hold-out R² |
|-------------------|--------------|------------|
| Linear Regression | ~45.9k       | 0.74       |
| Random Forest     | ~26.7k       | 0.91       |

---

### Cross-Validation (5-Fold)

Random Forest:

- CV RMSE: **26.5k ± 5.7k**
- CV R²: **0.876 ± 0.054**

---

### Tuned Model (GridSearchCV)

Best Parameters:
```text
max_features = 0.7
min_samples_leaf = 2
min_samples_split = 5
n_estimators = 300
max_depth = None
```

Performance:

- CV RMSE (best): **25.9k**
- Hold-out RMSE: **26.5k**
- Hold-out R²: **0.912**

---

## Approach

### 1. Preprocessing

- Numeric features → median imputation
- Categorical features → most frequent imputation + one-hot encoding
- Implemented via `ColumnTransformer` inside a `Pipeline` to prevent data leakage.

---

### 2. Model Selection

Compared:
- Linear Regression (baseline)
- Random Forest Regressor

Random Forest significantly outperformed the linear baseline, capturing nonlinear relationships in the housing data.

---

### 3. Cross-Validation

Used 5-fold K-Fold cross-validation on the training set to evaluate generalization performance and reduce variance from a single train/test split.

---

### 4. Hyperparameter Tuning

Used GridSearchCV to tune:

- max_depth
- n_estimators
- min_samples_split
- min_samples_leaf
- max_features

Selected model based on lowest cross-validated RMSE.

---

### 5. Feature Importance

Top predictive factors included:

- Overall quality
- Living area
- Neighborhood indicators
- Year built

Feature importance plot saved to:
```text
reports/feature_importance.png
```

---

## ▶ How to Run

### 1. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Train Model
```bash
python -m src.train
```
### 4. Hyperparameter Tuning
```bash
python -m src.tune
```
### 5. Generate Feature Importance
```bash
python -m src.feature_importance
```
---

## Project Structure
```text
house-price-predictor/
├── data/
│   └── raw/
|       └── housing.csv
├── models/
|   └── model.joblib
├── reports/
|   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── predict.py
│   ├── train.py
│   ├── tune.py
│   └── utils.py
├── requirements.txt
└── README.md
```
---

## Key Takeaways

- Demonstrates proper ML workflow design using sklearn Pipelines
- Prevents data leakage through structured preprocessing
- Applies cross-validation for robust performance estimation
- Uses hyperparameter tuning to improve generalization
- Produces reproducible saved models for inference

---

## Contact

**Derek Sun**  
University of Toronto — Computer Science  
📧 sunderek3602@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/derek-sun)  
🔗 [GitHub](https://github.com/dereksun00)
