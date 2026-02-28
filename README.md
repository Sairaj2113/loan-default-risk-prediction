# 🏦 Home Credit Default Risk Prediction — Stacking Ensemble

Advanced machine learning solution for predicting loan default risk using the Home Credit dataset.  
Built for the SYNTHESIZE 2026 AI Hackathon.

This project implements a high-performance stacked ensemble combining LightGBM, XGBoost, and CatBoost with extensive feature engineering and target encoding.

---

# 📌 Problem Statement

Financial institutions must assess the risk of loan default before approving credit.  
The objective is to predict the probability that a loan applicant will default.

- TARGET = 1 → Default
- TARGET = 0 → Repay

Dataset:
- ~300,000 loan applications  
- 120+ features  
- Numerical + categorical + missing data  
- Imbalanced (~8% defaults)  

Evaluation Metric: **ROC-AUC**

---

# 🚀 Solution Overview

This solution uses a **multi-model stacking ensemble** with advanced feature engineering to achieve strong predictive performance (>0.80 ROC-AUC).

## 🔹 Key Techniques

- Extensive feature engineering (financial ratios, EXT_SOURCE interactions)
- Target encoding (OOF)
- Frequency & count encoding
- Polynomial feature interactions
- Risk score engineered features
- Missing value indicators
- Feature binning
- Class imbalance handling
- GPU-accelerated gradient boosting
- 3-model stacking ensemble

---

# 🧠 Model Architecture

Level-1 Models:
- LightGBM (DART)
- LightGBM (GBDT)
- XGBoost
- CatBoost

Level-2 Meta-Learner:
- Logistic Regression / LightGBM

Ensemble Strategy:
- Stratified K-Fold
- Multi-seed averaging
- Out-of-fold predictions

---

# 📊 Feature Engineering Highlights

Examples of engineered predictors:

- CREDIT_INCOME_RATIO  
- PAYMENT_RATE  
- LOAN_BURDEN_SCORE  
- EXT_SOURCE interactions  
- AGE_EMPLOYED_RATIO  
- INCOME_PER_PERSON  
- EXT_MEAN / EXT_STD / EXT_PROD  
- CREDIT_ANNUITY_RATIO  
- RISK_SCORE / NET_RISK_SCORE  

Total features after engineering: **500+**

---

# ⚙️ Installation

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost
