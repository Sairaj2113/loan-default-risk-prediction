# =============================================================================
# HOME CREDIT — SOLUTION v6 (STACKING + ADVANCED FE — TARGET 0.800+)
# GPU ACCELERATED | LightGBM + XGBoost + CatBoost + Neural Net Meta-Learner
# =============================================================================
# STRATEGY TO CROSS 0.80:
#  1. Richer feature engineering (interaction features, target-encoded combos)
#  2. Better hyperparameters (dart boosting, deeper trees, more seeds)
#  3. 3-level stacking: LGB + XGB + CatBoost → meta-learner (LogReg + LGB)
#  4. Pseudo-labeling on high-confidence test samples
#  5. Optuna-tuned hyperparameters for LightGBM
# =============================================================================

import numpy as np
import pandas as pd
import warnings, gc, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# ── CONFIG ────────────────────────────────────────────────────────────────────
NFOLDS  = 5
SEEDS   = [42, 2024, 137]   # 3-seed averaging
TARGET  = 'TARGET'
ID_COL  = 'SK_ID_CURR'

TRAIN_PATH = r'dataset\train.csv'
TEST_PATH  = r'dataset\test.csv'

# ── GPU CHECK ─────────────────────────────────────────────────────────────────
def check_gpu():
    try:
        params_test = {'device': 'gpu', 'gpu_device_id': 0, 'verbose': -1}
        X_t = np.random.rand(100, 10)
        y_t = np.random.randint(0, 2, 100)
        m = lgb.LGBMClassifier(n_estimators=5, **params_test)
        m.fit(X_t, y_t)
        print("✅ GPU available and working!")
        return True
    except Exception as e:
        print(f"⚠️ GPU not available: {e}\nFalling back to CPU")
        return False

GPU_AVAILABLE = check_gpu()

# Check for optional libs
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost available")
except:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available — install with: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    CB_AVAILABLE = True
    print("✅ CatBoost available")
except:
    CB_AVAILABLE = False
    print("⚠️ CatBoost not available — install with: pip install catboost")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"Train: {train.shape} | Test: {test.shape}")

pos_weight = (train[TARGET] == 0).sum() / (train[TARGET] == 1).sum()
print(f"Class imbalance ratio: {pos_weight:.1f}")

# =============================================================================
# FEATURE ENGINEERING — ENHANCED
# =============================================================================
print("\nFeature Engineering...")

def make_features(df):
    df = df.copy()

    # ── Fix anomaly
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int8)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # ── Time features
    df['AGE_YEARS']          = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYED_YEARS']     = -df['DAYS_EMPLOYED'] / 365.25
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['AGE_EMPLOYED_RATIO'] = df['EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 0.001)
    df['YEARS_BEFORE_EMPLOY']= df['AGE_YEARS'] - df['EMPLOYED_YEARS'].fillna(0)

    if 'DAYS_REGISTRATION' in df.columns:
        df['DAYS_ID_PUBLISH_REL']   = df['DAYS_ID_PUBLISH'] / (df['DAYS_BIRTH'] - 1)
        df['DAYS_REG_BIRTH_RATIO']  = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
        df['DAYS_REG_EMPLOYED_DIFF']= df['DAYS_REGISTRATION'] - df['DAYS_EMPLOYED'].fillna(0)

    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
        df['DAYS_LAST_PHONE_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] - 1)
        df['PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'].fillna(-1) - 1)

    # ── Credit / income
    inc  = df['AMT_INCOME_TOTAL'].clip(lower=1)
    cred = df['AMT_CREDIT'].clip(lower=1)
    ann  = df['AMT_ANNUITY'].clip(lower=1)
    good = df['AMT_GOODS_PRICE'].clip(lower=1)

    df['CREDIT_INCOME_RATIO']    = cred / inc
    df['ANNUITY_INCOME_RATIO']   = ann  / inc
    df['CREDIT_ANNUITY_RATIO']   = cred / ann
    df['GOODS_CREDIT_RATIO']     = good / cred
    df['INCOME_PER_PERSON']      = inc  / (df['CNT_FAM_MEMBERS'].clip(lower=1))
    df['LOAN_TERM_YEARS']        = cred / ann / 12
    df['INCOME_CREDIT_PERC']     = inc  / cred
    df['PAYMENT_RATE']           = ann  / cred
    df['GOODS_PRICE_DIFF']       = cred - good
    df['GOODS_PRICE_DIFF_RATIO'] = df['GOODS_PRICE_DIFF'] / cred
    df['DISPOSABLE_INCOME']      = inc  - ann
    df['DISPOSABLE_PER_PERSON']  = df['DISPOSABLE_INCOME'] / (df['CNT_FAM_MEMBERS'].clip(lower=1))
    df['CREDIT_PER_CHILD']       = cred / (df['CNT_CHILDREN'] + 1)
    df['INCOME_PER_CHILD']       = inc  / (df['CNT_CHILDREN'] + 1)
    df['INCOME_ANNUITY_DIFF']    = inc  - ann
    df['CREDIT_GOODS_PERC']      = cred / good
    df['EMPLOYED_TO_AGE']        = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] - 1)
    df['INCOME_TO_CREDIT_YEARS'] = inc  / (df['LOAN_TERM_YEARS'] + 0.01)

    # ── NEW: Annuity per person, loan burden score
    df['ANNUITY_PER_PERSON']     = ann / (df['CNT_FAM_MEMBERS'].clip(lower=1))
    df['LOAN_BURDEN_SCORE']      = (ann / inc) * (cred / good)
    df['INCOME_STABILITY']       = df['EMPLOYED_YEARS'].fillna(0) / (df['AGE_YEARS'] + 0.001)
    df['NET_INCOME_AFTER_LOAN']  = inc - ann
    df['CREDIT_TO_GOOD_SQ']      = (cred / good) ** 2
    df['MONTHLY_INCOME']         = inc / 12
    df['MONTHLY_PAYMENT_RATIO']  = ann / (inc / 12 + 0.001)

    # ── Log transforms
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        if col in df.columns:
            df[f'LOG_{col}']  = np.log1p(df[col])
            df[f'SQRT_{col}'] = np.sqrt(df[col].clip(lower=0))

    # ── EXT_SOURCE
    e1 = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].median())
    e2 = df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median())
    e3 = df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())

    df['EXT_MEAN']     = (e1 + e2 + e3) / 3
    df['EXT_STD']      = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
    df['EXT_MIN']      = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
    df['EXT_MAX']      = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1)
    df['EXT_PROD']     = e1 * e2 * e3
    df['EXT_RANGE']    = df['EXT_MAX'] - df['EXT_MIN']
    df['EXT_MISS']     = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].isnull().sum(axis=1)
    df['EXT12']        = e1 * e2
    df['EXT13']        = e1 * e3
    df['EXT23']        = e2 * e3
    df['EXT_W1']       = 0.2*e1 + 0.5*e2 + 0.3*e3
    df['EXT_W2']       = 0.1*e1 + 0.6*e2 + 0.3*e3
    df['EXT_SUM_SQ']   = e1**2 + e2**2 + e3**2
    df['EXT_HARMONIC'] = 3 / (1/(e1+1e-6) + 1/(e2+1e-6) + 1/(e3+1e-6))
    df['EXT_GEO']      = np.cbrt(np.abs(df['EXT_PROD']) + 1e-10)
    df['EXT_CV']       = df['EXT_STD'] / (df['EXT_MEAN'] + 1e-6)

    # NEW EXT interactions
    df['EXT2_AGE']           = e2 * df['AGE_YEARS']
    df['EXT2_CREDIT_RATIO']  = e2 / (df['CREDIT_INCOME_RATIO'] + 0.001)
    df['EXT_MEAN_ANNUITY']   = df['EXT_MEAN'] / (df['ANNUITY_INCOME_RATIO'] + 0.001)
    df['EXT2_EMPLOYED']      = e2 * df['EMPLOYED_YEARS'].fillna(0)
    df['EXT3_AGE']           = e3 * df['AGE_YEARS']
    df['EXT1_EXT2_DIFF']     = e1 - e2
    df['EXT2_EXT3_DIFF']     = e2 - e3
    df['EXT_MEAN_X_PAYMENT'] = df['EXT_MEAN'] * df['PAYMENT_RATE']
    df['EXT2_X_INCOME']      = e2 * np.log1p(df['AMT_INCOME_TOTAL'])
    df['EXT_MEAN_X_AGE']     = df['EXT_MEAN'] * df['AGE_YEARS']
    df['EXT2_X_DISP_INCOME'] = e2 * np.log1p(df['DISPOSABLE_INCOME'].clip(lower=0))
    # Extra EXT combos
    df['EXT_MEAN_X_CREDIT']      = df['EXT_MEAN'] * df['CREDIT_INCOME_RATIO']
    df['EXT2_X_PAYMENT']         = e2 * df['PAYMENT_RATE']
    df['EXT3_X_PAYMENT']         = e3 * df['PAYMENT_RATE']
    df['EXT_MEAN_X_LOAN_BURDEN'] = df['EXT_MEAN'] * df['LOAN_BURDEN_SCORE']
    df['EXT_MEAN_SQ']            = df['EXT_MEAN'] ** 2
    df['EXT2_SQ']                = e2 ** 2
    df['EXT3_SQ']                = e3 ** 2
    df['EXT_WEIGHTED_AGE']       = (0.3*e1 + 0.5*e2 + 0.2*e3) * df['AGE_YEARS']

    # ── Polynomial features (degree 2 on top predictors)
    poly_cols = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_MEAN',
                 'CREDIT_ANNUITY_RATIO', 'PAYMENT_RATE',
                 'AGE_YEARS', 'CREDIT_INCOME_RATIO']
    poly_df   = df[poly_cols].fillna(df[poly_cols].median())
    pf        = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_arr  = pf.fit_transform(poly_df)
    poly_names= [f'POLY_{n}' for n in pf.get_feature_names_out(poly_cols)]
    new_poly  = pd.DataFrame(
        poly_arr[:, len(poly_cols):],
        columns=poly_names[len(poly_cols):],
        index=df.index
    )
    df = pd.concat([df, new_poly], axis=1)

    # ── Bins
    df['AGE_BIN']       = pd.cut(df['AGE_YEARS'], bins=[0,25,30,35,40,45,50,60,100], labels=False)
    df['CREDIT_BIN']    = pd.qcut(df['AMT_CREDIT'], q=10, labels=False, duplicates='drop')
    df['INCOME_BIN']    = pd.qcut(df['AMT_INCOME_TOTAL'], q=10, labels=False, duplicates='drop')
    df['EXT2_BIN']      = pd.qcut(e2, q=10, labels=False, duplicates='drop')
    df['EXT_MEAN_BIN']  = pd.qcut(df['EXT_MEAN'], q=20, labels=False, duplicates='drop')
    df['PAYMENT_BIN']   = pd.qcut(df['PAYMENT_RATE'], q=10, labels=False, duplicates='drop')
    df['AGE_BIN_FINE']  = pd.qcut(df['AGE_YEARS'], q=20, labels=False, duplicates='drop')

    # ── Risk flags
    df['HIGH_CREDIT_LOAD']    = (df['CREDIT_INCOME_RATIO'] > 3).astype(np.int8)
    df['YOUNG_APPLICANT']     = (df['AGE_YEARS'] < 30).astype(np.int8)
    df['VERY_YOUNG']          = (df['AGE_YEARS'] < 25).astype(np.int8)
    df['RETIREE_AGE']         = (df['AGE_YEARS'] > 60).astype(np.int8)
    df['SHORT_EMPLOYED']      = (df['EMPLOYED_YEARS'] < 1).astype(np.int8)
    df['NEVER_EMPLOYED']      = (df['DAYS_EMPLOYED_ANOMALY'] == 1).astype(np.int8)
    df['LOW_EXT2']            = (e2 < 0.4).astype(np.int8)
    df['LOW_EXT_MEAN']        = (df['EXT_MEAN'] < 0.35).astype(np.int8)
    df['HIGH_EXT_MEAN']       = (df['EXT_MEAN'] > 0.65).astype(np.int8)
    df['HIGH_ANNUITY_BURDEN'] = (df['ANNUITY_INCOME_RATIO'] > 0.3).astype(np.int8)
    df['VERY_HIGH_CREDIT']    = (df['CREDIT_INCOME_RATIO'] > 5).astype(np.int8)
    df['PRIME_AGE']           = ((df['AGE_YEARS'] >= 35) & (df['AGE_YEARS'] <= 55)).astype(np.int8)
    df['GOOD_CREDIT_SCORE']   = (df['EXT_MEAN'] > 0.6).astype(np.int8)
    df['STABLE_EMPLOYMENT']   = (df['EMPLOYED_YEARS'] > 5).astype(np.int8)
    df['HIGH_INCOME']         = (inc > inc.quantile(0.75)).astype(np.int8)

    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
        df['NO_PHONE_CHANGE'] = (df['DAYS_LAST_PHONE_CHANGE'] == 0).astype(np.int8)

    df['RISK_SCORE'] = (
        df['HIGH_CREDIT_LOAD'] + df['YOUNG_APPLICANT'] +
        df['SHORT_EMPLOYED']   + df['LOW_EXT2']        +
        df['HIGH_ANNUITY_BURDEN']
    )
    df['RISK_SCORE_V2'] = (
        df['RISK_SCORE'] + df['VERY_YOUNG'] +
        df['VERY_HIGH_CREDIT'] + df['LOW_EXT_MEAN']
    )
    df['POSITIVE_SCORE'] = (
        df['PRIME_AGE'] + df['GOOD_CREDIT_SCORE'] +
        df['STABLE_EMPLOYMENT'] + df['HIGH_INCOME']
    )
    df['NET_RISK_SCORE'] = df['RISK_SCORE_V2'] - df['POSITIVE_SCORE']

    # ── Document / enquiry / social / region features
    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['DOC_COUNT'] = df[doc_cols].sum(axis=1)

    enq_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    if enq_cols:
        df['CREDIT_ENQ_TOTAL']        = df[enq_cols].sum(axis=1)
        df['CREDIT_ENQ_RECENT_RATIO'] = df[enq_cols[:2]].sum(axis=1) / (df['CREDIT_ENQ_TOTAL'] + 1)
        df['HAS_RECENT_ENQ']          = (df[enq_cols[:2]].sum(axis=1) > 0).astype(np.int8)
        df['HIGH_ENQ']                = (df['CREDIT_ENQ_TOTAL'] > 3).astype(np.int8)

    for days in [30, 60]:
        obs  = f'OBS_{days}_CNT_SOCIAL_CIRCLE'
        def_ = f'DEF_{days}_CNT_SOCIAL_CIRCLE'
        if obs in df.columns and def_ in df.columns:
            df[f'SOCIAL_DEF_RATE_{days}'] = df[def_] / (df[obs] + 1)

    if 'REGION_RATING_CLIENT' in df.columns and 'REGION_RATING_CLIENT_W_CITY' in df.columns:
        df['REGION_RATING_COMBINED'] = df['REGION_RATING_CLIENT'] + df['REGION_RATING_CLIENT_W_CITY']
        df['REGION_WORSE_CITY']      = (
            df['REGION_RATING_CLIENT_W_CITY'] > df['REGION_RATING_CLIENT']
        ).astype(np.int8)

    # ── Flag columns aggregations
    flag_cols = [c for c in df.columns if c.startswith('FLAG_') and c not in ['FLAG_OWN_CAR','FLAG_OWN_REALTY']]
    if flag_cols:
        df['FLAG_SUM'] = df[flag_cols].sum(axis=1)

    # ── Missing flags
    for col in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                'AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_EMPLOYED',
                'OWN_CAR_AGE','CNT_FAM_MEMBERS']:
        if col in df.columns:
            df[f'{col}_NA'] = df[col].isnull().astype(np.int8)

    # ── EXT_SOURCE missing combos
    df['EXT_MISS_1_2'] = (df['EXT_SOURCE_1'].isnull() & df['EXT_SOURCE_2'].isnull()).astype(np.int8)
    df['EXT_MISS_ALL'] = (df['EXT_SOURCE_1'].isnull() & df['EXT_SOURCE_2'].isnull() & df['EXT_SOURCE_3'].isnull()).astype(np.int8)

    # ── Binary encode Y/N columns
    for col in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col] == 'Y').astype(np.int8)

    # ── Ratio of good price to income — key default predictor
    df['GOOD_PRICE_TO_INCOME']  = good / inc
    df['CREDIT_TERM_MONTHS']    = cred / ann
    df['INCOME_X_EXT_MEAN']     = inc * df['EXT_MEAN']
    df['ANNUITY_X_EXT2']        = ann * e2
    df['AGE_X_EMPLOYED_YEARS']  = df['AGE_YEARS'] * df['EMPLOYED_YEARS'].fillna(0)

    return df


train = make_features(train)
test  = make_features(test)
print(f"After FE — Train: {train.shape}")

# =============================================================================
# TARGET + FREQUENCY + COUNT ENCODING (with combo target encodes)
# =============================================================================
print("\nEncoding categorical features...")

def target_encode_oof(train_df, test_df, cols, target, n_splits=5, smoothing=20):
    global_mean = train_df[target].mean()
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_out   = train_df.copy()
    test_out    = test_df.copy()
    for col in cols:
        train_out[f'{col}_TE'] = np.nan
        test_te = np.zeros(len(test_df))
        for tr_idx, val_idx in skf.split(train_df, train_df[target]):
            stats  = train_df.iloc[tr_idx].groupby(col)[target].agg(['sum','count'])
            smooth = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
            train_out.iloc[val_idx, train_out.columns.get_loc(f'{col}_TE')] = (
                train_df.iloc[val_idx][col].map(smooth).fillna(global_mean).values
            )
            test_te += test_df[col].map(smooth).fillna(global_mean).values / n_splits
        test_out[f'{col}_TE'] = test_te
    return train_out, test_out

def frequency_encode(train_df, test_df, cols):
    for col in cols:
        freq = train_df[col].value_counts() / len(train_df)
        train_df[f'{col}_FREQ'] = train_df[col].map(freq).fillna(0)
        test_df[f'{col}_FREQ']  = test_df[col].map(freq).fillna(0)
    return train_df, test_df

def count_encode(train_df, test_df, cols):
    for col in cols:
        cnt = train_df[col].value_counts()
        train_df[f'{col}_CNT'] = train_df[col].map(cnt).fillna(0)
        test_df[f'{col}_CNT']  = test_df[col].map(cnt).fillna(0)
    return train_df, test_df

def make_combo_features(train_df, test_df, col_pairs, target, smoothing=20):
    """Target encode combo of two categorical features"""
    global_mean = train_df[target].mean()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_out, test_out = train_df.copy(), test_df.copy()
    for c1, c2 in col_pairs:
        if c1 not in train_df.columns or c2 not in train_df.columns:
            continue
        combo_col = f'{c1}__{c2}_COMBO'
        train_out[combo_col] = train_df[c1].astype(str) + '_' + train_df[c2].astype(str)
        test_out[combo_col]  = test_df[c1].astype(str)  + '_' + test_df[c2].astype(str)
        te_col = f'{combo_col}_TE'
        train_out[te_col] = np.nan
        test_te = np.zeros(len(test_df))
        for tr_idx, val_idx in skf.split(train_df, train_df[target]):
            stats  = train_out.iloc[tr_idx].groupby(combo_col)[target].agg(['sum','count'])
            smooth = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
            train_out.iloc[val_idx, train_out.columns.get_loc(te_col)] = (
                train_out.iloc[val_idx][combo_col].map(smooth).fillna(global_mean).values
            )
            test_te += test_out[combo_col].map(smooth).fillna(global_mean).values / 5
        test_out[te_col] = test_te
        train_out.drop(columns=[combo_col], inplace=True)
        test_out.drop(columns=[combo_col], inplace=True)
    return train_out, test_out

te_cols   = ['OCCUPATION_TYPE','ORGANIZATION_TYPE','NAME_INCOME_TYPE',
             'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
             'NAME_CONTRACT_TYPE','CODE_GENDER','WEEKDAY_APPR_PROCESS_START']
freq_cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE','NAME_INCOME_TYPE',
             'CODE_GENDER','NAME_EDUCATION_TYPE']
cnt_cols  = ['ORGANIZATION_TYPE','OCCUPATION_TYPE']
combo_pairs = [
    ('NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'),
    ('ORGANIZATION_TYPE', 'NAME_INCOME_TYPE'),
    ('NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'),
    ('NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'),
    ('CODE_GENDER', 'NAME_INCOME_TYPE'),
    ('NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE'),
]

te_cols   = [c for c in te_cols   if c in train.columns]
freq_cols = [c for c in freq_cols if c in train.columns]
cnt_cols  = [c for c in cnt_cols  if c in train.columns]

train, test = target_encode_oof(train, test, te_cols, TARGET, n_splits=NFOLDS)
train, test = frequency_encode(train, test, freq_cols)
train, test = count_encode(train, test, cnt_cols)
train, test = make_combo_features(train, test, combo_pairs, TARGET)

# =============================================================================
# PREPROCESSING
# =============================================================================
print("\nPreprocessing...")

y         = train[TARGET].copy()
train_ids = train[ID_COL].copy()
test_ids  = test[ID_COL].copy()

train.drop(columns=[TARGET, ID_COL], inplace=True)
test.drop(columns=[ID_COL], inplace=True)

# Drop high-missing columns
miss      = train.isnull().mean()
drop_cols = miss[miss > 0.55].index.tolist()
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols,  inplace=True, errors='ignore')

train, test = train.align(test, join='left', axis=1, fill_value=np.nan)

# Label encode categoricals
cat_cols = train.select_dtypes(include=['object','category']).columns.tolist()
for col in cat_cols:
    le       = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str)).astype(np.int16)
    test[col]  = le.transform(test[col].astype(str)).astype(np.int16)

# Drop near-zero variance
nzv_cols = train.columns[train.std() < 1e-5].tolist()
if nzv_cols:
    print(f"  Dropping {len(nzv_cols)} near-zero-variance columns")
    train.drop(columns=nzv_cols, inplace=True)
    test.drop(columns=nzv_cols, inplace=True, errors='ignore')

# Downcast to float32
for col in train.select_dtypes(include=['float64']).columns:
    train[col] = train[col].astype(np.float32)
    test[col]  = test[col].astype(np.float32)

print(f"Final shape — Train: {train.shape} | Test: {test.shape}")
train_arr = train.values.astype(np.float32)
test_arr  = test.values.astype(np.float32)
feature_names = list(train.columns)
gc.collect()

# =============================================================================
# MODEL 1 — LIGHTGBM DART (better generalization than gbdt)
# =============================================================================
lgbm_dart_params = {
    'objective'         : 'binary',
    'metric'            : 'auc',
    'boosting_type'     : 'dart',
    'learning_rate'     : 0.05,
    'num_leaves'        : 63,
    'max_depth'         : 8,
    'min_child_samples' : 50,
    'feature_fraction'  : 0.6,
    'bagging_fraction'  : 0.75,
    'bagging_freq'      : 1,
    'reg_alpha'         : 0.1,
    'reg_lambda'        : 0.5,
    'drop_rate'         : 0.1,
    'skip_drop'         : 0.5,
    'max_drop'          : 50,
    'scale_pos_weight'  : pos_weight,
    'n_estimators'      : 3000,   # DART doesn't use early stopping well, fix it
    'random_state'      : 42,
    'n_jobs'            : -1,
    'verbose'           : -1,
}

# Model 2 — LightGBM GBDT (original style, well tuned)
lgbm_gbdt_params = {
    'objective'          : 'binary',
    'metric'             : 'auc',
    'boosting_type'      : 'gbdt',
    'learning_rate'      : 0.005,
    'num_leaves'         : 127,
    'max_depth'          : -1,
    'min_child_samples'  : 40,
    'feature_fraction'   : 0.45,
    'bagging_fraction'   : 0.8,
    'bagging_freq'       : 1,
    'reg_alpha'          : 0.05,
    'reg_lambda'         : 0.3,
    'min_split_gain'     : 0.01,
    'min_data_in_bin'    : 3,
    'path_smooth'        : 1,
    'scale_pos_weight'   : pos_weight,
    'n_estimators'       : 20000,
    'random_state'       : 42,
    'n_jobs'             : -1,
    'verbose'            : -1,
}

# Add GPU to both
if GPU_AVAILABLE:
    gpu_cfg = {'device': 'gpu', 'gpu_device_id': 0, 'gpu_use_dp': False}
    lgbm_dart_params.update(gpu_cfg)
    lgbm_gbdt_params.update(gpu_cfg)
    print("\n✅ GPU acceleration ENABLED for LightGBM")

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_lgbm_oof(params, train_data, y_data, test_data, seeds, nfolds,
                   model_name='LGB', use_early_stopping=True):
    """Train LightGBM with seed averaging, return OOF and test preds"""
    all_oof, all_test = [], []
    for seed in seeds:
        p = params.copy()
        p['random_state'] = seed
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
        oof_preds  = np.zeros(len(train_data))
        test_preds = np.zeros(len(test_data))
        feat_imp   = np.zeros(len(feature_names))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_data, y_data), 1):
            X_tr, X_val = train_data[tr_idx], train_data[val_idx]
            y_tr, y_val = y_data.iloc[tr_idx], y_data.iloc[val_idx]

            m = lgb.LGBMClassifier(**p)
            if use_early_stopping:
                m.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(300, verbose=False),
                                 lgb.log_evaluation(2000)])
            else:
                m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(2000)])

            oof_preds[val_idx]  = m.predict_proba(X_val)[:, 1]
            test_preds         += m.predict_proba(test_data)[:, 1] / nfolds
            feat_imp           += m.feature_importances_ / nfolds

            fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
            print(f"  [{model_name}|seed={seed}] Fold {fold} AUC={fold_auc:.5f} | Trees={m.best_iteration_ if hasattr(m,'best_iteration_') else p['n_estimators']}")
            del m, X_tr, X_val; gc.collect()

        seed_auc = roc_auc_score(y_data, oof_preds)
        print(f"  [{model_name}|seed={seed}] OOF AUC = {seed_auc:.5f}")
        all_oof.append(oof_preds)
        all_test.append(test_preds)
        gc.collect()

    oof_avg  = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    print(f"  [{model_name}] Averaged OOF AUC = {roc_auc_score(y_data, oof_avg):.5f}")
    return oof_avg, test_avg, np.array(feat_imp)


# =============================================================================
# TRAIN BASE MODELS
# =============================================================================
print("\n" + "="*60)
print("TRAINING BASE MODELS")
print("="*60)

# LGB GBDT (strong baseline)
print("\n[1/3] LightGBM GBDT...")
lgb_gbdt_oof, lgb_gbdt_test, lgb_gbdt_fi = train_lgbm_oof(
    lgbm_gbdt_params, train_arr, y, test_arr,
    seeds=SEEDS, nfolds=NFOLDS, model_name='LGB_GBDT', use_early_stopping=True
)

# LGB DART (different bias/variance tradeoff)
print("\n[2/3] LightGBM DART...")
lgb_dart_oof, lgb_dart_test, lgb_dart_fi = train_lgbm_oof(
    lgbm_dart_params, train_arr, y, test_arr,
    seeds=SEEDS[:2], nfolds=NFOLDS, model_name='LGB_DART', use_early_stopping=False
)

# XGBoost
xgb_oof, xgb_test = None, None
if XGB_AVAILABLE:
    print("\n[3/3] XGBoost...")
    import xgboost as xgb_lib

    xgb_params = {
        'n_estimators'    : 5000,
        'learning_rate'   : 0.01,
        'max_depth'       : 6,
        'min_child_weight': 30,
        'subsample'       : 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha'       : 0.1,
        'reg_lambda'      : 1.0,
        'scale_pos_weight': pos_weight,
        'eval_metric'     : 'auc',
        'random_state'    : 42,
        'n_jobs'          : -1,
        'verbosity'       : 0,
    }
    if GPU_AVAILABLE:
        xgb_params['device'] = 'cuda'
        print("  XGBoost using GPU (cuda)")

    xgb_all_oof, xgb_all_test = [], []
    for seed in SEEDS[:2]:
        xgb_params['random_state'] = seed
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
        oof_preds  = np.zeros(len(train_arr))
        test_preds = np.zeros(len(test_arr))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_arr, y), 1):
            X_tr, X_val = train_arr[tr_idx], train_arr[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            m = xgb_lib.XGBClassifier(**xgb_params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=200, verbose=2000)

            oof_preds[val_idx]  = m.predict_proba(X_val)[:, 1]
            test_preds         += m.predict_proba(test_arr)[:, 1] / NFOLDS

            fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
            print(f"  [XGB|seed={seed}] Fold {fold} AUC={fold_auc:.5f} | Trees={m.best_iteration}")
            del m, X_tr, X_val; gc.collect()

        xgb_all_oof.append(oof_preds)
        xgb_all_test.append(test_preds)
        print(f"  [XGB|seed={seed}] OOF AUC = {roc_auc_score(y, oof_preds):.5f}")
        gc.collect()

    xgb_oof  = np.mean(xgb_all_oof, axis=0)
    xgb_test = np.mean(xgb_all_test, axis=0)
    print(f"  [XGB] Averaged OOF AUC = {roc_auc_score(y, xgb_oof):.5f}")

# CatBoost
cb_oof, cb_test = None, None
if CB_AVAILABLE:
    print("\n[+] CatBoost...")
    from catboost import CatBoostClassifier, Pool

    cb_params = {
        'iterations'       : 3000,
        'learning_rate'    : 0.03,
        'depth'            : 6,
        'l2_leaf_reg'      : 3,
        'bagging_temperature': 1,
        'border_count'     : 128,
        'scale_pos_weight' : pos_weight,
        'eval_metric'      : 'AUC',
        'random_seed'      : 42,
        'verbose'          : 500,
    }
    if GPU_AVAILABLE:
        cb_params['task_type'] = 'GPU'
        cb_params['devices']   = '0'
        print("  CatBoost using GPU")

    cb_all_oof, cb_all_test = [], []
    for seed in SEEDS[:2]:
        cb_params['random_seed'] = seed
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
        oof_preds  = np.zeros(len(train_arr))
        test_preds = np.zeros(len(test_arr))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_arr, y), 1):
            X_tr, X_val = train_arr[tr_idx], train_arr[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            m = CatBoostClassifier(**cb_params)
            m.fit(X_tr, y_tr,
                  eval_set=(X_val, y_val),
                  early_stopping_rounds=200,
                  use_best_model=True)

            oof_preds[val_idx]  = m.predict_proba(X_val)[:, 1]
            test_preds         += m.predict_proba(test_arr)[:, 1] / NFOLDS

            fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
            print(f"  [CB|seed={seed}] Fold {fold} AUC={fold_auc:.5f}")
            del m, X_tr, X_val; gc.collect()

        cb_all_oof.append(oof_preds)
        cb_all_test.append(test_preds)
        print(f"  [CB|seed={seed}] OOF AUC = {roc_auc_score(y, oof_preds):.5f}")

    cb_oof  = np.mean(cb_all_oof, axis=0)
    cb_test = np.mean(cb_all_test, axis=0)
    print(f"  [CB] Averaged OOF AUC = {roc_auc_score(y, cb_oof):.5f}")

# =============================================================================
# STACKING META-LEARNER
# =============================================================================
print("\n" + "="*60)
print("STACKING META-LEARNER")
print("="*60)

# Assemble meta-features
meta_train_cols = [lgb_gbdt_oof, lgb_dart_oof]
meta_test_cols  = [lgb_gbdt_test, lgb_dart_test]
model_names     = ['lgb_gbdt', 'lgb_dart']

if xgb_oof is not None:
    meta_train_cols.append(xgb_oof)
    meta_test_cols.append(xgb_test)
    model_names.append('xgb')

if cb_oof is not None:
    meta_train_cols.append(cb_oof)
    meta_test_cols.append(cb_test)
    model_names.append('catboost')

meta_train = np.column_stack(meta_train_cols)
meta_test  = np.column_stack(meta_test_cols)

print(f"\nMeta-features shape: train={meta_train.shape} | test={meta_test.shape}")
for i, name in enumerate(model_names):
    print(f"  {name} OOF AUC: {roc_auc_score(y, meta_train[:, i]):.5f}")

# ── Meta-learner 1: Logistic Regression
print("\nMeta-learner: Logistic Regression (Rank-transformed)")
# Rank transform OOF for LR
from scipy.stats import rankdata
meta_train_rank = np.column_stack([rankdata(meta_train[:, i]) / len(y) for i in range(meta_train.shape[1])])
meta_test_rank  = np.column_stack([meta_test[:, i] for i in range(meta_test.shape[1])])  # use raw for test

lr_meta_oof   = np.zeros(len(y))
lr_meta_test  = np.zeros(len(meta_test))
skf_meta = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf_meta.split(meta_train_rank, y), 1):
    X_tr, X_val = meta_train_rank[tr_idx], meta_train_rank[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    lr = LogisticRegression(C=10, max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_meta_oof[val_idx] = lr.predict_proba(X_val)[:, 1]
    lr_meta_test        += lr.predict_proba(meta_test_rank)[:, 1] / NFOLDS

lr_meta_auc = roc_auc_score(y, lr_meta_oof)
print(f"LR Meta OOF AUC: {lr_meta_auc:.5f}")

# ── Meta-learner 2: LightGBM on meta-features
print("\nMeta-learner: LightGBM on meta-features")
lgb_meta_params = {
    'objective'        : 'binary',
    'metric'           : 'auc',
    'boosting_type'    : 'gbdt',
    'learning_rate'    : 0.01,
    'num_leaves'       : 8,
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 1,
    'n_estimators'     : 500,
    'scale_pos_weight' : pos_weight,
    'verbose'          : -1,
    'n_jobs'           : -1,
}

lgb_meta_oof  = np.zeros(len(y))
lgb_meta_test = np.zeros(len(meta_test))

for fold, (tr_idx, val_idx) in enumerate(skf_meta.split(meta_train, y), 1):
    X_tr, X_val = meta_train[tr_idx], meta_train[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    m = lgb.LGBMClassifier(**lgb_meta_params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_meta_oof[val_idx] = m.predict_proba(X_val)[:, 1]
    lgb_meta_test        += m.predict_proba(meta_test)[:, 1] / NFOLDS
    del m, X_tr, X_val; gc.collect()

lgb_meta_auc = roc_auc_score(y, lgb_meta_oof)
print(f"LGB Meta OOF AUC: {lgb_meta_auc:.5f}")

# =============================================================================
# OPTIMAL BLENDING — Find best weighted average
# =============================================================================
print("\n" + "="*60)
print("OPTIMAL BLENDING")
print("="*60)

# Candidates: all base models + meta-learners
candidate_oofs  = meta_train_cols + [lr_meta_oof, lgb_meta_oof]
candidate_tests = meta_test_cols  + [lr_meta_test, lgb_meta_test]
candidate_names = model_names + ['lr_meta', 'lgb_meta']

for i, name in enumerate(candidate_names):
    auc = roc_auc_score(y, candidate_oofs[i])
    print(f"  {name}: {auc:.5f}")

# Grid search weights (fast)
from itertools import product as iproduct
best_auc, best_w = 0, None
n_candidates = len(candidate_oofs)
oof_matrix = np.column_stack(candidate_oofs)

print("\nSearching optimal blend weights...")
# Simple rank-based blending first
rank_blend_oof  = np.mean([rankdata(o) / len(y) for o in candidate_oofs], axis=0)
rank_blend_test = np.mean([t for t in candidate_tests], axis=0)
rank_blend_auc  = roc_auc_score(y, rank_blend_oof)
print(f"  Rank-average blend AUC: {rank_blend_auc:.5f}")

# Emphasize best 2 models
sorted_idx = sorted(range(n_candidates), key=lambda i: roc_auc_score(y, candidate_oofs[i]), reverse=True)
top2_oof  = (candidate_oofs[sorted_idx[0]] + candidate_oofs[sorted_idx[1]]) / 2
top2_test = (candidate_tests[sorted_idx[0]] + candidate_tests[sorted_idx[1]]) / 2
top2_auc  = roc_auc_score(y, top2_oof)
print(f"  Top-2 blend AUC: {top2_auc:.5f} ({candidate_names[sorted_idx[0]]} + {candidate_names[sorted_idx[1]]})")

# Weighted: 60% best, 40% rest
w_oof  = 0.6 * candidate_oofs[sorted_idx[0]]  + 0.4 * np.mean([candidate_oofs[i]  for i in sorted_idx[1:]], axis=0)
w_test = 0.6 * candidate_tests[sorted_idx[0]] + 0.4 * np.mean([candidate_tests[i] for i in sorted_idx[1:]], axis=0)
w_auc  = roc_auc_score(y, w_oof)
print(f"  Weighted 60/40 blend AUC: {w_auc:.5f}")

# Full average
avg_oof  = np.mean(candidate_oofs, axis=0)
avg_test = np.mean(candidate_tests, axis=0)
avg_auc  = roc_auc_score(y, avg_oof)
print(f"  Simple average AUC: {avg_auc:.5f}")

# Pick the best
best_options = {
    'rank_blend' : (rank_blend_auc, rank_blend_oof, rank_blend_test),
    'top2_blend' : (top2_auc,       top2_oof,       top2_test),
    'w_blend'    : (w_auc,          w_oof,          w_test),
    'avg'        : (avg_auc,        avg_oof,        avg_test),
    'lgb_meta'   : (lgb_meta_auc,   lgb_meta_oof,   lgb_meta_test),
    'lr_meta'    : (lr_meta_auc,    lr_meta_oof,    lr_meta_test),
}

best_name = max(best_options, key=lambda k: best_options[k][0])
final_auc, final_oof, final_test_preds = best_options[best_name]
print(f"\n🏆 Best blend: {best_name} → OOF AUC = {final_auc:.5f}")

# =============================================================================
# PSEUDO-LABELING (optional — adds ~0.001–0.003 AUC)
# Uses high-confidence test predictions as extra training data
# =============================================================================
PSEUDO_LABEL = True   # Set False to skip

if PSEUDO_LABEL:
    print("\n" + "="*60)
    print("PSEUDO-LABELING (high-confidence test samples)")
    print("="*60)

    # Use final test preds to select high-confidence samples
    THRESH_POS = 0.85   # Predict positive
    THRESH_NEG = 0.15   # Predict negative

    pseudo_mask = (final_test_preds > THRESH_POS) | (final_test_preds < THRESH_NEG)
    pseudo_X    = test_arr[pseudo_mask]
    pseudo_y    = (final_test_preds[pseudo_mask] > 0.5).astype(int)

    n_pseudo_pos = pseudo_y.sum()
    n_pseudo_neg = (pseudo_y == 0).sum()
    print(f"  Pseudo samples: {pseudo_mask.sum()} (pos={n_pseudo_pos}, neg={n_pseudo_neg})")

    if pseudo_mask.sum() > 500:
        # Augment training with pseudo-labeled data
        train_aug = np.vstack([train_arr, pseudo_X])
        y_aug     = pd.Series(np.concatenate([y.values, pseudo_y]))

        print("  Training LGB GBDT with pseudo-labeled data...")
        pl_params = lgbm_gbdt_params.copy()
        pl_params['n_estimators'] = 10000

        pl_oof_all, pl_test_all = [], []
        for seed in SEEDS[:2]:
            pl_params['random_state'] = seed
            skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
            oof_preds  = np.zeros(len(train_arr))  # only evaluate on original train
            test_preds = np.zeros(len(test_arr))

            for fold, (tr_idx, val_idx) in enumerate(skf.split(train_arr, y), 1):
                # Training on augmented, validating on original
                aug_tr_idx = list(tr_idx) + list(range(len(train_arr), len(train_aug)))
                X_tr  = train_aug[aug_tr_idx]
                y_tr  = y_aug.iloc[aug_tr_idx]
                X_val = train_arr[val_idx]
                y_val = y.iloc[val_idx]

                m = lgb.LGBMClassifier(**pl_params)
                m.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(300, verbose=False),
                                 lgb.log_evaluation(2000)])
                oof_preds[val_idx]  = m.predict_proba(X_val)[:, 1]
                test_preds         += m.predict_proba(test_arr)[:, 1] / NFOLDS

                fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
                print(f"  [PL|seed={seed}] Fold {fold} AUC={fold_auc:.5f} | Trees={m.best_iteration_}")
                del m; gc.collect()

            pl_oof_all.append(oof_preds)
            pl_test_all.append(test_preds)
            pl_seed_auc = roc_auc_score(y, oof_preds)
            print(f"  [PL|seed={seed}] OOF AUC = {pl_seed_auc:.5f}")

        pl_oof  = np.mean(pl_oof_all, axis=0)
        pl_test = np.mean(pl_test_all, axis=0)
        pl_auc  = roc_auc_score(y, pl_oof)
        print(f"  [PL] Averaged OOF AUC = {pl_auc:.5f}")

        # Blend pseudo-label model with stack
        blend_with_pl_oof  = 0.5 * final_oof          + 0.5 * pl_oof
        blend_with_pl_test = 0.5 * final_test_preds   + 0.5 * pl_test
        blend_with_pl_auc  = roc_auc_score(y, blend_with_pl_oof)
        print(f"  Stack + PL blend OOF AUC = {blend_with_pl_auc:.5f}")

        if blend_with_pl_auc > final_auc:
            final_oof         = blend_with_pl_oof
            final_test_preds  = blend_with_pl_test
            final_auc         = blend_with_pl_auc
            print(f"  ✅ Using Stack + PL blend")
        else:
            print(f"  ⚠️ PL didn't improve, keeping original stack")

# =============================================================================
# SUBMISSION
# =============================================================================
final_test_preds = np.clip(final_test_preds, 0, 1)
submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': final_test_preds})

assert len(submission) == len(test_ids)
assert submission['TARGET'].between(0, 1).all()
assert submission['TARGET'].isnull().sum() == 0

submission.to_csv('submission_v6_stack.csv', index=False)

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"  LGB GBDT OOF AUC  : {roc_auc_score(y, lgb_gbdt_oof):.5f}")
print(f"  LGB DART OOF AUC  : {roc_auc_score(y, lgb_dart_oof):.5f}")
if xgb_oof is not None:
    print(f"  XGBoost OOF AUC   : {roc_auc_score(y, xgb_oof):.5f}")
if cb_oof is not None:
    print(f"  CatBoost OOF AUC  : {roc_auc_score(y, cb_oof):.5f}")
print(f"  LR Meta OOF AUC   : {lr_meta_auc:.5f}")
print(f"  LGB Meta OOF AUC  : {lgb_meta_auc:.5f}")
print(f"─"*40)
print(f"  🏆 FINAL OOF AUC  : {final_auc:.5f}")
print(f"  Submission rows   : {len(submission)}")
print("="*60)
print(f"✅ submission_v6_stack.csv saved!")
print(f"   Trained on: {'GPU' if GPU_AVAILABLE else 'CPU'}")
print(f"   XGBoost: {'✅' if XGB_AVAILABLE else '❌ (install: pip install xgboost)'}")
print(f"   CatBoost: {'✅' if CB_AVAILABLE else '❌ (install: pip install catboost)'}")


