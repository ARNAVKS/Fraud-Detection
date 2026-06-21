# Fraudulent Transaction Detection

A machine learning project to detect fraudulent credit card transactions using supervised and unsupervised learning techniques on a highly imbalanced dataset.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group).

- 284,807 transactions with 492 fraud cases (~0.17% of all transactions)
- Target: `Class` (0 = legitimate, 1 = fraud)

## Pipeline Overview

```
Raw Data → EDA → Preprocessing → PCA → Resampling → Supervised Models → Unsupervised Clustering
```

### 1. Preprocessing
- Dropped duplicates, checked for nulls
- Scaled `Time` and `Amount` using `StandardScaler` (fit on train only)
- Applied PCA (95% variance retention) to reduce dimensionality
- 80/20 stratified train-test split

### 2. Handling Class Imbalance
Two strategies compared on the training set (test set kept original):

| Strategy | Method |
|---|---|
| Oversampling | SMOTE (`imblearn`) |
| Undersampling | `RandomUnderSampler` |

### 3. Supervised Models

**Random Forest** and **XGBoost** trained on both resampled variants and evaluated on the original imbalanced test set.

Metrics reported: Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.

Key finding: SMOTE-trained models outperformed undersampled models on the unchanged test set. Custom probability thresholds (e.g. 0.75, 0.95) were also explored to tune the precision-recall tradeoff.

### 4. Unsupervised Clustering (K-Means)

- Elbow method used to select `k=2`
- Silhouette score comparison:

| Training Data | Train Silhouette | Test Silhouette |
|---|---|---|
| SMOTE (oversampled) | ~0.64 | ~0.83 |
| Undersampled | ~0.64 | ~0.83 |

SMOTE-trained clustering showed significant improvement on the test set (0.83), confirming that synthetic oversampling produces better-separated cluster boundaries.

## Results Summary

- Random Forest + SMOTE achieves strong recall on the minority (fraud) class
- ROC-AUC plotted for the best model
- K-Means clustering on SMOTE data yields a silhouette score of **0.83** on original test data

## Tech Stack

- **Data:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **ML:** `scikit-learn` (Random Forest, PCA, KMeans, metrics), `xgboost`
- **Resampling:** `imbalanced-learn` (SMOTE, RandomUnderSampler)

## Setup

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn kaggle
```

Place your `kaggle.json` in the root, then run all cells in `Fraudulent_Transaction_Detection.ipynb`.

## Project Structure

```
├── Fraudulent_Transaction_Detection.ipynb
├── kaggle.json            # your Kaggle API key (not committed)
└── README.md
```

> **Note:** `kaggle.json` should be added to `.gitignore`.
