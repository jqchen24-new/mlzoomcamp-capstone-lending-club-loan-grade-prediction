
import numpy as np
import pandas as pd
import os
import joblib
import kagglehub
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL_OUTPUT    = 'xgboost_model.pkl'
DV_OUTPUT       = 'dv.pkl'
LE_OUTPUT       = 'le.pkl'


def load_data():
    # Download latest version
    print("Loading data...")
    path = kagglehub.dataset_download("adarshsng/lending-club-loan-data-csv")
    df = pd.read_csv(os.path.join(path, 'loan.csv'), nrows = 75000)
    drop_missing_columns = list(df.columns[df.isnull().sum() > 50000])
    df.drop(columns = drop_missing_columns, inplace=True)

    numerical = list(df.dtypes[df.dtypes != 'object'].index)
    categorical = list(df.dtypes[df.dtypes == 'object'].index)

    selector = VarianceThreshold(threshold=0.01)  

    numeric_cols = df.select_dtypes(include='number').columns

    selector.fit(df[numeric_cols])

    kept = numeric_cols[selector.get_support()]
    dropped = numeric_cols[~selector.get_support()]

    for var in categorical:
        if df[var].nunique() > 10:
            del df[var]

    leakage_cols = [
        # Post-origination payment data
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
        'total_rec_int', 'total_rec_late_fee', 'recoveries',
        'collection_recovery_fee', 'last_pymnt_amnt',
        'out_prncp', 'out_prncp_inv', 'disbursement_method',
        
        # Loan status after origination
        'loan_status',
        
        # Payment plan info
        'pymnt_plan',
        
        # Next payment date (post-origination)
        'pymnt_plan', 'last_pymnt_d',
        
        # Hardship/settlement (post-origination)
        'hardship_flag', 'debt_settlement_flag',
        
        # Already removing these
        'int_rate', 'sub_grade'
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    print("Splitting...")
    df_full_train, df_test = train_test_split(df, test_size = 0.2, stratify=df['grade'])
    df_train, df_val = train_test_split(df_full_train, test_size = 0.25, stratify=df_full_train['grade'])

    y_train = df_train['grade']
    y_val = df_val['grade']
    y_test = df_test['grade']

    del df_train['grade']
    del df_val['grade']
    del df_test['grade']


    dv = DictVectorizer(sparse = False)

    train_dict = df_train.to_dict(orient = 'records')
    val_dict = df_val.to_dict(orient = 'records')
    test_dict = df_test.to_dict(orient = 'records')

    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    X_test = dv.transform(test_dict)

    return X_train, X_val, X_test, y_train, y_val, y_test, dv


X_train, X_val, X_test, y_train, y_val, y_test, dv = load_data()


# model training

# Encode target to integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  # 'A'->0, 'B'->1, etc.
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

print("Training XGBoost...")
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    device='cuda',              # GPU
    tree_method='hist',         # required for GPU
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    early_stopping_rounds=20,   # stops if val loss doesn't improve
    random_state=42
)

model.fit(
    X_train, y_train_enc,
    eval_set=[(X_val, y_val_enc)],
    verbose=50                  # print every 50 rounds
)


def model_perf(X = X_val, y = y_val_enc):
    y_pred_xgb = model.predict(X)

    # Decode back to grade labels
    y_pred_labels = le.inverse_transform(y_pred_xgb)
    y_labels = le.inverse_transform(y)

    print(classification_report(y_labels, y_pred_labels, target_names=list('ABCDEFG')))
    print(f"Weighted F1: {f1_score(y_labels, y_pred_labels, average='weighted'):.4f}")
    print(f"Macro F1:    {f1_score(y_labels, y_pred_labels, average='macro'):.4f}")

print("\n── Validation ──")
model_perf(X = X_train, y = y_train_enc)

model_perf(X = X_val, y = y_val_enc)

model_perf(X = X_test, y = y_test_enc)

# ── Save artifacts ────────────────────────────────────────────────────────────
print("\nSaving artifacts...")
joblib.dump(model, MODEL_OUTPUT)
joblib.dump(dv,    DV_OUTPUT)
joblib.dump(le,    LE_OUTPUT)

print(f"Saved: {MODEL_OUTPUT}, {DV_OUTPUT}, {LE_OUTPUT}")
print("Done.")