# Pima Indians Diabetes Prediction
### Predicting diabetes from clinical measurements using KNN classification

**Tools:** Python · Pandas · Seaborn · Scikit-learn  
**Dataset:** [Pima Indians Diabetes — UCI / Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Author:** Hafsah Shamsi · Microbiology + Data Science · Mithibai College, Mumbai

---

## Overview

This project builds a K-Nearest Neighbours (KNN) classifier to predict whether a patient has diabetes based on 8 clinical measurements — including glucose concentration, BMI, age, insulin levels, and blood pressure.

The dataset's central challenge is **data quality**: five features contain zeros that are biologically impossible in a living patient. These are missing values recorded as zeros, not real measurements. Handling them correctly before modelling is the most critical step in this project.

A secondary focus is **model reliability** — rather than trusting a single train/test split, 5-fold cross-validation is used to evaluate performance across multiple data subsets, giving a more honest estimate of real-world accuracy.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~77% |
| ROC-AUC | ~0.82 |
| Diabetic Recall | ~70% |
| Best K | found via cross-validation |

---

## Key challenge: zeros as missing values

These features cannot biologically be zero:

| Feature | Zeros in dataset | % missing |
|---|---|---|
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | 48.7% |
| BMI | 11 | 1.4% |

All replaced with column medians before modelling. Insulin and SkinThickness were nearly half missing — a real-world data quality problem, not a toy exercise.

---

## Charts produced

| File | Description |
|---|---|
| `eda_diabetes.png` | Outcome distribution, glucose histogram, BMI vs age scatter |
| `correlation_heatmap_diabetes.png` | Feature correlations with diabetes outcome |
| `knn_k_selection.png` | CV accuracy across K=1 to 20 — how the best K was chosen |
| `model_evaluation_diabetes.png` | Confusion matrix + ROC curve |

---

## How to run

```bash
git clone https://github.com/HafsahShamsi/pima-diabetes-prediction.git
cd pima-diabetes-prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the project folder, then:

```bash
jupyter notebook pima_diabetes_complete.ipynb
```

---

## What I learned

- Identifying and handling impossible zeros as missing values in clinical data
- Median imputation — why median is more robust than mean for skewed clinical features
- KNN classification — distance-based prediction, sensitivity to feature scale
- Hyperparameter tuning — finding optimal K via cross-validation
- StratifiedKFold — preserving class balance across folds
- Why cross-validation gives more reliable performance estimates than a single split

---

## Next steps

- [ ] Compare KNN with logistic regression and Random Forest on this dataset
- [ ] Tune classification threshold to improve diabetic recall
- [ ] Test whether removing heavily-missing Insulin improves consistency
- [ ] Move to Project 3: Mushroom Edibility Classification (decision trees)
