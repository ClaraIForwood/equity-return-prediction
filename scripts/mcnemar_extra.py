"""
Additional McNemar's tests using saved test predictions.

Task 1: Each trained model vs Majority Class baseline.
Task 2: Each deep learning model (MLP, LSTM, Transformer) vs XGBoost.

Uses identical implementation to notebook Cell 83:
  - continuity correction, two-sided, alpha = 0.05
  - b = reference correct & model wrong
  - c = model correct & reference wrong
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# ── Project root ──────────────────────────────────────────────────────────────
def find_project_root(marker: str = "config.py") -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Cannot locate {marker} above {Path(__file__)}")

PROJECT_ROOT = find_project_root()
TABLES_DIR   = PROJECT_ROOT / "results" / "tables"

# ── Load predictions (identical to notebook Cell 83 reload guard) ─────────────
cache = np.load(TABLES_DIR / "predictions_cache.npz", allow_pickle=False)

y_true       = cache["y_test_eval"]
lr_pred      = cache["lr_pred"]
mlp_pred     = cache["mlp_pred"]
xgb_pred     = cache["xgb_pred"]
lstm_pred    = cache["lstm_pred"]
tf_pred      = cache["tf_pred"]
maj_pred     = cache["maj_pred"]

print(f"Loaded {len(y_true)} test predictions from predictions_cache.npz\n")

# ── McNemar helper (identical to notebook) ────────────────────────────────────
def run_mcnemar(ref_pred, ref_name, model_pred, model_name, y_true):
    correct_ref   = (ref_pred   == y_true)
    correct_model = (model_pred == y_true)
    b = int(np.sum( correct_ref & ~correct_model))  # ref right, model wrong
    c = int(np.sum(~correct_ref &  correct_model))  # model right, ref wrong
    table  = np.array([[0, b], [c, 0]])
    result = mcnemar(table, exact=False, correction=True)
    return {
        "model":           model_name,
        "reference_model": ref_name,
        "b_ref_right_model_wrong": b,
        "c_model_right_ref_wrong": c,
        "chi2":            round(result.statistic, 4),
        "p_value":         round(result.pvalue,    4),
        "significant":     "Yes *" if result.pvalue < 0.05 else "No",
    }

# ── Task 1: trained models vs Majority Class ──────────────────────────────────
trained_models = [
    ("Logistic Regression", lr_pred),
    ("MLP",                 mlp_pred),
    ("XGBoost",             xgb_pred),
    ("LSTM",                lstm_pred),
    ("Transformer",         tf_pred),
]

rows_task1 = [
    run_mcnemar(maj_pred, "Majority Class", pred, name, y_true)
    for name, pred in trained_models
]
df1 = pd.DataFrame(rows_task1)

print("=" * 68)
print("TASK 1 — Each trained model vs Majority Class baseline (alpha=0.05)")
print("=" * 68)
print(df1.to_string(index=False))
print()

df1.to_csv(TABLES_DIR / "mcnemar_vs_majority.csv", index=False)
print(f"Saved -> {TABLES_DIR / 'mcnemar_vs_majority.csv'}\n")

# ── Task 2: deep learning models vs XGBoost ───────────────────────────────────
dl_models = [
    ("MLP",         mlp_pred),
    ("LSTM",        lstm_pred),
    ("Transformer", tf_pred),
]

rows_task2 = [
    run_mcnemar(xgb_pred, "XGBoost", pred, name, y_true)
    for name, pred in dl_models
]
df2 = pd.DataFrame(rows_task2)

print("=" * 68)
print("TASK 2 — Deep learning models vs XGBoost (alpha=0.05)")
print("=" * 68)
print(df2.to_string(index=False))
print()

df2.to_csv(TABLES_DIR / "mcnemar_dl_vs_xgboost.csv", index=False)
print(f"Saved -> {TABLES_DIR / 'mcnemar_dl_vs_xgboost.csv'}")
