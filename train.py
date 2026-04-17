import torch
import numpy as np
import pandas as pd
import kagglehub
import os
import argparse
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from model import LoanGradePredictor

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cpu',   action='store_true', help='Force CPU training')
parser.add_argument('--nrows', type=int, default=500000)
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(nrows=500000):
    path = kagglehub.dataset_download("adarshsng/lending-club-loan-data-csv")
    df   = pd.read_csv(os.path.join(path, 'loan.csv'), nrows=nrows, low_memory=False)

    df.drop(columns=list(df.columns[df.isnull().mean() > 0.5]), inplace=True)

    numeric_cols = df.select_dtypes(include='number').columns
    from sklearn.feature_selection import VarianceThreshold
    selector     = VarianceThreshold(threshold=0.01)
    selector.fit(df[numeric_cols])
    kept = numeric_cols[selector.get_support()]
    df   = df[kept.tolist() + list(df.select_dtypes(include='object').columns)]

    for col in list(df.select_dtypes(include='object').columns):
        if df[col].nunique() > 10:
            df = df.drop(columns=col)

    leakage_cols = [
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_amnt', 'out_prncp', 'out_prncp_inv', 'disbursement_method',
        'loan_status', 'pymnt_plan', 'last_pymnt_d', 'next_pymnt_d',
        'hardship_flag', 'debt_settlement_flag', 'int_rate', 'sub_grade'
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    df = df[df['grade'].notna()]

    print(f"Dataset shape: {df.shape}")
    print(f"Grade distribution:\n{df['grade'].value_counts().sort_index()}")
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading data...')
    df = load_data(nrows=args.nrows)

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['grade']
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=42, stratify=df_full_train['grade']
    )
    print(f"Train: {df_train.shape} | Val: {df_val.shape} | Test: {df_test.shape}")

    num_epochs = 25 if device.type == 'cpu' else 50

    predictor = LoanGradePredictor(device=device)
    predictor.fit(df_train, df_val, num_epochs=num_epochs)

    predictor.evaluate(df_train, label='Train')
    predictor.evaluate(df_val,   label='Validation')
    predictor.evaluate(df_test,  label='Test')

    low_risk = pd.DataFrame([{
        "loan_amnt": 10000, "funded_amnt": 10000, "installment": 320.0,
        "annual_inc": 120000, "dti": 8.5, "revol_util": 12.5,
        "delinq_2yrs": 0, "inq_last_6mths": 0, "pub_rec": 0,
        "term": " 36 months", "home_ownership": "MORTGAGE",
        "verification_status": "Verified"
    }])
    high_risk = pd.DataFrame([{
        "loan_amnt": 35000, "funded_amnt": 35000, "installment": 950.0,
        "annual_inc": 32000, "dti": 42.0, "revol_util": 95.0,
        "delinq_2yrs": 5, "pub_rec": 2, "num_tl_90g_dpd_24m": 4,
        "pct_tl_nvr_dlq": 40.0, "pub_rec_bankruptcies": 1,
        "term": " 60 months", "home_ownership": "RENT",
        "verification_status": "Not Verified"
    }])
    print(f"\nLow risk:  {predictor.predict(low_risk)[0]}")
    print(f"High risk: {predictor.predict(high_risk)[0]}")

    predictor.save('predictor.pkl')
    print('\nFeature names:', predictor.dv.feature_names_)