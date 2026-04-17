import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from model import LoanGradePredictor

# ── Load model ────────────────────────────────────────────────────────────────
predictor = LoanGradePredictor.load('predictor.pkl')
print("Model loaded successfully")
print("Classes:", predictor.le.classes_)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title='Loan Grade Predictor', version='1.0')

# ── Request schema ────────────────────────────────────────────────────────────
class LoanInput(BaseModel):
    acc_open_past_24mths:       Optional[float] = None
    all_util:                   Optional[float] = None
    annual_inc:                 Optional[float] = None
    avg_cur_bal:                Optional[float] = None
    bc_open_to_buy:             Optional[float] = None
    bc_util:                    Optional[float] = None
    collections_12_mths_ex_med: Optional[float] = None
    delinq_2yrs:                Optional[float] = None
    delinq_amnt:                Optional[float] = None
    dti:                        Optional[float] = None
    funded_amnt:                Optional[float] = None
    funded_amnt_inv:            Optional[float] = None
    il_util:                    Optional[float] = None
    inq_fi:                     Optional[float] = None
    inq_last_12m:               Optional[float] = None
    inq_last_6mths:             Optional[float] = None
    installment:                Optional[float] = None
    loan_amnt:                  Optional[float] = None
    max_bal_bc:                 Optional[float] = None
    mo_sin_old_il_acct:         Optional[float] = None
    mo_sin_old_rev_tl_op:       Optional[float] = None
    mo_sin_rcnt_rev_tl_op:      Optional[float] = None
    mo_sin_rcnt_tl:             Optional[float] = None
    mort_acc:                   Optional[float] = None
    mths_since_rcnt_il:         Optional[float] = None
    mths_since_recent_bc:       Optional[float] = None
    mths_since_recent_inq:      Optional[float] = None
    num_accts_ever_120_pd:      Optional[float] = None
    num_actv_bc_tl:             Optional[float] = None
    num_actv_rev_tl:            Optional[float] = None
    num_bc_sats:                Optional[float] = None
    num_bc_tl:                  Optional[float] = None
    num_il_tl:                  Optional[float] = None
    num_op_rev_tl:              Optional[float] = None
    num_rev_accts:              Optional[float] = None
    num_rev_tl_bal_gt_0:        Optional[float] = None
    num_sats:                   Optional[float] = None
    num_tl_90g_dpd_24m:         Optional[float] = None
    num_tl_op_past_12m:         Optional[float] = None
    open_acc:                   Optional[float] = None
    open_acc_6m:                Optional[float] = None
    open_act_il:                Optional[float] = None
    open_il_12m:                Optional[float] = None
    open_il_24m:                Optional[float] = None
    open_rv_12m:                Optional[float] = None
    open_rv_24m:                Optional[float] = None
    pct_tl_nvr_dlq:             Optional[float] = None
    percent_bc_gt_75:           Optional[float] = None
    pub_rec:                    Optional[float] = None
    pub_rec_bankruptcies:       Optional[float] = None
    revol_bal:                  Optional[float] = None
    revol_util:                 Optional[float] = None
    tax_liens:                  Optional[float] = None
    tot_coll_amt:               Optional[float] = None
    tot_cur_bal:                Optional[float] = None
    tot_hi_cred_lim:            Optional[float] = None
    total_acc:                  Optional[float] = None
    total_bal_ex_mort:          Optional[float] = None
    total_bal_il:               Optional[float] = None
    total_bc_limit:             Optional[float] = None
    total_cu_tl:                Optional[float] = None
    total_il_high_credit_limit: Optional[float] = None
    total_rev_hi_lim:           Optional[float] = None
    application_type:           Optional[str]   = None
    home_ownership:             Optional[str]   = None
    initial_list_status:        Optional[str]   = None
    term:                       Optional[str]   = None
    verification_status:        Optional[str]   = None

    model_config = {'extra': 'allow'}

# ── Risk level mapping ────────────────────────────────────────────────────────
RISK_LEVELS = {
    'A': 'Very Low Risk',
    'B': 'Low Risk',
    'C': 'Moderate Risk',
    'D': 'Medium-High Risk',
    'E': 'High Risk',
    'F': 'Very High Risk',
    'G': 'Highest Risk'
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(loan: LoanInput):
    input_data = {k: v for k, v in loan.model_dump().items() if v is not None}
    input_df   = pd.DataFrame([input_data])
    grade      = predictor.predict(input_df)[0]
    return {
        'grade':      grade,
        'risk_level': RISK_LEVELS.get(grade, 'Unknown')
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9696)