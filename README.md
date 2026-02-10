# Credit Card Fraud Detection (Imbalanced Classification)

Detect fraudulent credit card transactions using supervised learning under severe class imbalance.  
This project compares baseline and imbalance-aware approaches (SMOTE / undersampling), and evaluates trade-offs between fraud recall and false positives for practical risk operations.

## Highlights
- Built models on **280K+** transactions with heavy class imbalance
- Applied **SMOTE**, undersampling, and feature engineering to improve minority-class learning
- Trained and tuned **Logistic Regression**, **Decision Tree**, and **XGBoost** with cross-validation + grid search
- Achieved **0.92 ROC-AUC** and improved **fraud recall by 21%**
- Produced clear diagnostics (class distribution, ROC/PR, confusion matrix) to support threshold selection

## Tech Stack
- **Python**: pandas, numpy, scikit-learn
- **Modeling**: XGBoost, Logistic Regression, Decision Tree
- **Imbalance handling**: imbalanced-learn (SMOTE), undersampling
- **Visualization**: matplotlib / seaborn

## Project Structure (suggested)
├── data/ # (optional) raw/processed data (not tracked if sensitive)
├── notebooks/
│ └── fraud_detection.ipynb # exploration + experiments
├── src/
│ ├── preprocess.py # cleaning, split, scaling
│ ├── features.py # feature engineering
│ ├── train.py # training loop
│ ├── evaluate.py # metrics + plots
│ └── utils.py
├── outputs/
│ ├── figures/ # ROC/PR curves, confusion matrix, etc.
│ └── reports/ # experiment summaries
├── requirements.txt
└── README.md

## Data
This project uses a credit card transactions dataset with anonymized features and an extremely imbalanced label distribution (fraud is rare).  
> If you are using a public dataset (e.g., Kaggle “Credit Card Fraud Detection”), place the CSV under `data/` (and keep it out of Git history if required).

## Methodology
1. **EDA**
   - Inspect missing values, feature distributions, and fraud rate patterns across categories
   - Quantify imbalance ratio and establish baseline metrics

2. **Preprocessing**
   - Train/validation/test split with stratification
   - Optional scaling for linear models
   - Baseline model training on original distribution

3. **Imbalance Handling**
   - **SMOTE** on training set only (to avoid leakage)
   - Compare with **undersampling**
   - Track how sampling impacts recall vs false positives

4. **Modeling & Tuning**
   - Logistic Regression (baseline + interpretable)
   - Decision Tree (nonlinear baseline)
   - XGBoost (strong nonlinear + interactions)
   - Grid search + cross-validation using metrics suitable for imbalance

5. **Evaluation**
   - ROC-AUC (overall ranking)
   - Precision / Recall / F1 for fraud class
   - PR curve (recommended under heavy imbalance)
   - Confusion matrix at chosen threshold
   - Threshold tuning based on business cost trade-offs

## Results (from my experiments)
- Best model: **XGBoost**
- Performance: **0.92 ROC-AUC**
- Impact: **+21% fraud recall** (vs baseline)
- Practical insight: improved detection came with increased false positives, requiring threshold tuning based on operational tolerance.

## How to Run
### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
