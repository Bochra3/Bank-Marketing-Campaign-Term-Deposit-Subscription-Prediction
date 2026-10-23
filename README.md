# 🏦 Bank Marketing: Term Deposit Subscription Prediction

A machine learning project that predicts whether a bank client will subscribe to a term deposit, based on their demographic profile and how they were contacted during a marketing campaign. Created as the final project for our Data Mining course (Spring 2025–2026) at Tunis Business School.

## 🎯 Goal

Direct marketing is expensive, and most calls end in a "no". This project builds a model that helps a bank decide **which clients to call first, and why they are likely to say yes**.

## 📊 Results

Best model: **XGBoost**, evaluated on a held-out test set (~1,117 clients).

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.857 |
| Recall    | 0.875 |
| Precision | 0.831 |
| F1-Score  | 0.853 |
| ROC AUC   | 0.925 |

Six models were compared using 5-fold cross-validation (Recall as the primary metric):

| Model               | Recall | F1-Score | AUC   |
|---------------------|--------|----------|-------|
| **XGBoost**         | 0.873  | 0.848    | 0.925 |
| Random Forest       | 0.869  | 0.838    | 0.924 |
| SVM                 | 0.817  | 0.802    | 0.896 |
| Logistic Regression | 0.785  | 0.781    | 0.878 |
| Decision Tree       | 0.765  | 0.768    | 0.795 |
| KNN                 | 0.668  | 0.705    | 0.806 |

## 🔍 Key Insights

- **Previous campaign success** is the strongest signal: past subscribers are far more likely to subscribe again.
- **Contact channel, housing loan status and call duration** were the top predictors in XGBoost.
- **Cellular contact** and certain months (March, September, October, December) show higher subscription rates.
- **Students and retired clients** respond best.
- **Adjustable threshold:** lowering the decision threshold from 0.5 to 0.3 raises recall from 0.875 to 0.936, a useful lever depending on call-centre capacity.

## 🛠️ Approach

1. Exploratory data analysis (class balance, correlations, subscription rates by segment)
2. Preprocessing: ordinal encoding, stratified 90/10 train/test split, Min-Max scaling (fitted on train only to avoid leakage)
3. SMOTE applied inside each cross-validation fold via an `imbalanced-learn` pipeline
4. Training and comparison of 6 classifiers
5. Evaluation: confusion matrix, classification report, ROC curves, threshold analysis
6. Feature importance and business interpretation

## 🧰 Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · matplotlib · seaborn

## 📁 Dataset

[Bank Marketing Dataset (Balanced Version)](https://www.kaggle.com/datasets/rouseguy/bankbalanced) — originally from the UCI Machine Learning Repository (Moro, Cortez & Rita, 2014). ~11,000 client contacts from a Portuguese bank's telemarketing campaigns.

## ⚠️ Limitations

- `duration` (call length) is only known *after* a call, so it would be excluded in a real pre-campaign deployment.
- The dataset is balanced (~50% positives), whereas the real-world rate is ~11%. Precision would likely be lower on real campaign data.
- Random split used; a temporal split would give a more realistic estimate.

## ▶️ How to Run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
jupyter notebook <your-notebook>.ipynb
```

## 👥 Authors

- Bochra Messaoud
- Chadha Werghemi

Supervised by Prof. Lassad El Moubarki
