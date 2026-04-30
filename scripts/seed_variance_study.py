"""
Seed variance study for deep learning models.

Retrains MLP, LSTM, and Transformer (classifier + regressor) across five
random seeds and records test balanced accuracy and RMSE.  All hyperparameters,
data splits, and preprocessing are identical to the main training notebook.

Output: results/tables/seed_variance_results.csv
"""

import random
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ── Project root ──────────────────────────────────────────────────────────────
def find_project_root(marker: str = "config.py") -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Cannot locate {marker} above {Path(__file__)}")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURE_DATA_DIR,
    TARGET_SYMBOL, AUXILIARY_SYMBOLS, AUXILIARY_FEATURES,
    TRAIN_END_DATE, VAL_END_DATE,
)

# ── Constants (must match notebook exactly) ───────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATE_COLUMN  = "date"
SYMBOL_COLUMN = "symbol"
TARGET_COLUMN  = "target_direction"
REGRESSION_TARGET = "target_next_return"
ALL_TARGET_COLUMNS = ["target_next_close", "target_next_return", "target_direction"]
METADATA_COLUMNS   = [DATE_COLUMN, SYMBOL_COLUMN]
EXCLUDE_FEATURES   = []

TRAIN_END_DATE = pd.Timestamp(TRAIN_END_DATE)
VAL_END_DATE   = pd.Timestamp(VAL_END_DATE)

SEQ_LEN      = 20
LSTM_HIDDEN_DIM  = 32
LSTM_NUM_LAYERS  = 1
LSTM_DROPOUT     = 0.5
TF_D_MODEL   = 32
TF_NHEAD     = 2
TF_NUM_LAYERS = 1
TF_DIM_FF    = 64
TF_DROPOUT   = 0.5
BATCH_SIZE   = 64
MAX_EPOCHS   = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
LR_PATIENCE  = 7
LR_FACTOR    = 0.5
GRAD_CLIP    = 1.0

SEEDS = [42, 123, 7, 2024, 99]
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "seed_variance_results.csv"


# ── Data helpers ──────────────────────────────────────────────────────────────
def build_spy_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    target_df = (
        raw_df.loc[raw_df[SYMBOL_COLUMN] == TARGET_SYMBOL]
        .copy().sort_values(DATE_COLUMN).reset_index(drop=True)
    )
    base_features = [
        c for c in target_df.columns
        if c not in METADATA_COLUMNS + ALL_TARGET_COLUMNS + EXCLUDE_FEATURES
    ]
    aux_cols: List[str] = []
    available = set(raw_df[SYMBOL_COLUMN].unique())
    selected  = [s for s in AUXILIARY_SYMBOLS if s in available and s != TARGET_SYMBOL]
    if selected and AUXILIARY_FEATURES:
        aux_df   = raw_df.loc[raw_df[SYMBOL_COLUMN].isin(selected),
                               [DATE_COLUMN, SYMBOL_COLUMN, *AUXILIARY_FEATURES]].copy()
        aux_wide = aux_df.pivot_table(
            index=DATE_COLUMN, columns=SYMBOL_COLUMN,
            values=AUXILIARY_FEATURES, aggfunc="last",
        )
        aux_wide.columns = [f"{s}_{f}" for f, s in aux_wide.columns.to_flat_index()]
        target_df = target_df.merge(aux_wide.reset_index(), on=DATE_COLUMN, how="left")
        aux_cols  = [c for c in target_df.columns
                     if c not in METADATA_COLUMNS + ALL_TARGET_COLUMNS + base_features]
    target_df = target_df.loc[target_df[TARGET_COLUMN].notna()].copy()
    target_df[TARGET_COLUMN] = target_df[TARGET_COLUMN].astype(int)
    return target_df, base_features + aux_cols


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len: int, valid_indices):
        self.X, self.y = X, y
        self.seq_len   = seq_len
        self.indices   = valid_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        end = self.indices[i]
        return (
            torch.from_numpy(self.X[end - self.seq_len + 1 : end + 1]),
            torch.tensor(self.y[end]),
        )


def make_split_indices(dates, start_cond, end_cond):
    mask  = (start_cond & end_cond).values
    cands = np.where(mask)[0]
    return cands[cands >= SEQ_LEN - 1]


def make_loaders(X_scaled, y, train_idx, val_idx, test_idx):
    train_ld = DataLoader(
        SequenceDataset(X_scaled, y, SEQ_LEN, train_idx),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_ld = DataLoader(
        SequenceDataset(X_scaled, y, SEQ_LEN, val_idx),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_ld = DataLoader(
        SequenceDataset(X_scaled, y, SEQ_LEN, test_idx),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_ld, val_ld, test_ld


# ── Model architectures (identical to notebook) ───────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(self.dropout(h_n[-1])).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_seq_len=100):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x   = self.input_proj(x) + self.pos_embed(pos)
        x   = self.transformer(x)
        return self.classifier(self.dropout(x[:, -1, :])).squeeze(-1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,        64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(self.dropout(h_n[-1])).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_seq_len=100):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x   = self.input_proj(x) + self.pos_embed(pos)
        x   = self.transformer(x)
        return self.head(self.dropout(x[:, -1, :])).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,        64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x).squeeze(-1)


# ── Training loop (identical to notebook) ─────────────────────────────────────
def train_one_epoch(model, loader, opt, criterion):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loader(model, loader, criterion, apply_sigmoid: bool = True):
    model.eval()
    total, out_list, label_list = 0.0, [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        raw  = model(X)
        total += criterion(raw, y).item() * len(y)
        out = torch.sigmoid(raw).cpu().numpy() if apply_sigmoid else raw.cpu().numpy()
        out_list.append(out)
        label_list.append(y.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(out_list), np.concatenate(label_list)


def run_training(model, name, train_loader, val_loader,
                 train_criterion, val_criterion=None, task="classification"):
    if val_criterion is None:
        val_criterion = train_criterion
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
    )
    best_loss, best_state, patience_ctr = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, train_criterion)
        vl_loss, vp, vl = eval_loader(
            model, val_loader, val_criterion,
            apply_sigmoid=(task == "classification"),
        )
        scheduler.step(vl_loss)

        if vl_loss < best_loss:
            best_loss    = vl_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0 or epoch == 1:
            metric = (balanced_accuracy_score(vl, (vp >= 0.5).astype(int))
                      if task == "classification"
                      else float(np.sqrt(mean_squared_error(vl, vp))))
            metric_name = "val_ba" if task == "classification" else "val_rmse"
            print(f"  [{name}] epoch {epoch:3d} | tr={tr_loss:.4f}  vl={vl_loss:.4f}"
                  f"  {metric_name}={metric:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"  [{name}] Early stop at epoch {epoch}  ({time.time()-t0:.0f}s)")
            break

    if best_state:
        model.load_state_dict(best_state)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # Load feature dataset
    for candidate in [
        FEATURE_DATA_DIR / "equity_daily_features.parquet",
        FEATURE_DATA_DIR / "equity_daily_features.csv",
    ]:
        if candidate.exists():
            raw_df = (pd.read_parquet(candidate) if candidate.suffix == ".parquet"
                      else pd.read_csv(candidate, parse_dates=[DATE_COLUMN]))
            print(f"Loaded: {candidate}  shape={raw_df.shape}")
            break
    else:
        raise FileNotFoundError("Feature dataset not found. Run notebook 02 first.")

    # Build SPY dataset and splits (deterministic — done once)
    model_df, feature_columns = build_spy_dataset(raw_df)
    dates_all = model_df[DATE_COLUMN].reset_index(drop=True)

    train_mask = (dates_all <= TRAIN_END_DATE).values
    test_end   = dates_all.max()

    train_idx = make_split_indices(dates_all, dates_all >= dates_all.min(), dates_all <= TRAIN_END_DATE)
    val_idx   = make_split_indices(dates_all, dates_all > TRAIN_END_DATE,   dates_all <= VAL_END_DATE)
    test_idx  = make_split_indices(dates_all, dates_all > VAL_END_DATE,     dates_all <= test_end)
    print(f"Sequence indices — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    # Preprocessing (deterministic — fitted on train only)
    X_all_raw = model_df[feature_columns].values.astype(np.float64)
    imputer   = SimpleImputer(strategy="median").fit(X_all_raw[train_mask])
    X_imp     = imputer.transform(X_all_raw)
    scaler    = StandardScaler().fit(X_imp[train_mask])
    X_scaled  = scaler.transform(X_imp).astype(np.float32)

    # Targets
    y_cls = model_df[TARGET_COLUMN].values.astype(np.float32)
    y_reg = model_df[REGRESSION_TARGET].values.astype(np.float32)

    # pos_weight (deterministic)
    n_pos      = int((y_cls[train_mask] == 1).sum())
    n_neg      = int((y_cls[train_mask] == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"Pos weight: {pos_weight.item():.4f}  (pos={n_pos}, neg={n_neg})")

    # Ground-truth test labels (for direct numpy evaluation)
    y_test_cls = y_cls[test_idx]
    y_test_reg = y_reg[test_idx]

    n_features = X_scaled.shape[1]
    print(f"Features: {n_features}  |  Seeds: {SEEDS}\n")

    records = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        def set_seed(s: int):
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        # ── Classification loaders (recreated per seed for reproducible shuffle) ──
        set_seed(seed)
        cls_train_ld, cls_val_ld, cls_test_ld = make_loaders(
            X_scaled, y_cls, train_idx, val_idx, test_idx,
        )
        cls_train_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
        cls_val_crit   = nn.BCEWithLogitsLoss()

        # ── MLP Classifier ───────────────────────────────────────────────────────
        print("\n-- MLP Classifier --")
        set_seed(seed)
        mlp_clf = MLPClassifier(n_features, dropout=0.5).to(DEVICE)
        # recreate loaders so shuffle is seeded by the set_seed call above
        cls_train_ld, cls_val_ld, cls_test_ld = make_loaders(
            X_scaled, y_cls, train_idx, val_idx, test_idx,
        )
        run_training(mlp_clf, "MLP Clf", cls_train_ld, cls_val_ld,
                     nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE)),
                     nn.BCEWithLogitsLoss(), task="classification")
        with torch.no_grad():
            mlp_clf.eval()
            logits = mlp_clf(torch.tensor(X_scaled[test_idx], dtype=torch.float32).to(DEVICE))
            tp = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
        ba = balanced_accuracy_score(y_test_cls, tp)
        print(f"  test balanced_accuracy = {ba:.4f}")
        records.append({"model": "MLP Classifier", "seed": seed,
                        "test_balanced_accuracy": ba, "test_rmse": float("nan")})

        # ── LSTM Classifier ──────────────────────────────────────────────────────
        print("\n-- LSTM Classifier --")
        set_seed(seed)
        lstm_clf = LSTMClassifier(n_features, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT).to(DEVICE)
        cls_train_ld, cls_val_ld, cls_test_ld = make_loaders(
            X_scaled, y_cls, train_idx, val_idx, test_idx,
        )
        run_training(lstm_clf, "LSTM Clf", cls_train_ld, cls_val_ld,
                     cls_train_crit, cls_val_crit, task="classification")
        _, tp, tl = eval_loader(lstm_clf, cls_test_ld, cls_val_crit, apply_sigmoid=True)
        ba = balanced_accuracy_score(tl.astype(int), (tp >= 0.5).astype(int))
        print(f"  test balanced_accuracy = {ba:.4f}")
        records.append({"model": "LSTM Classifier", "seed": seed,
                        "test_balanced_accuracy": ba, "test_rmse": float("nan")})

        # ── Transformer Classifier ───────────────────────────────────────────────
        print("\n-- Transformer Classifier --")
        set_seed(seed)
        tf_clf = TransformerClassifier(n_features, TF_D_MODEL, TF_NHEAD, TF_NUM_LAYERS,
                                       TF_DIM_FF, TF_DROPOUT, max_seq_len=SEQ_LEN + 10).to(DEVICE)
        cls_train_ld, cls_val_ld, cls_test_ld = make_loaders(
            X_scaled, y_cls, train_idx, val_idx, test_idx,
        )
        run_training(tf_clf, "TF Clf", cls_train_ld, cls_val_ld,
                     cls_train_crit, cls_val_crit, task="classification")
        _, tp, tl = eval_loader(tf_clf, cls_test_ld, cls_val_crit, apply_sigmoid=True)
        ba = balanced_accuracy_score(tl.astype(int), (tp >= 0.5).astype(int))
        print(f"  test balanced_accuracy = {ba:.4f}")
        records.append({"model": "Transformer Classifier", "seed": seed,
                        "test_balanced_accuracy": ba, "test_rmse": float("nan")})

        # ── Regression loaders ────────────────────────────────────────────────────
        reg_crit = nn.L1Loss()

        # ── MLP Regressor ─────────────────────────────────────────────────────────
        print("\n-- MLP Regressor --")
        set_seed(seed)
        mlp_reg = MLPRegressor(n_features, dropout=0.5).to(DEVICE)
        reg_train_ld, reg_val_ld, _ = make_loaders(
            X_scaled, y_reg, train_idx, val_idx, test_idx,
        )
        run_training(mlp_reg, "MLP Reg", reg_train_ld, reg_val_ld,
                     reg_crit, task="regression")
        with torch.no_grad():
            mlp_reg.eval()
            tp_reg = mlp_reg(torch.tensor(X_scaled[test_idx], dtype=torch.float32).to(DEVICE)).cpu().numpy()
        rmse = float(np.sqrt(mean_squared_error(y_test_reg, tp_reg)))
        print(f"  test RMSE = {rmse:.6f}")
        records.append({"model": "MLP Regressor", "seed": seed,
                        "test_balanced_accuracy": float("nan"), "test_rmse": rmse})

        # ── LSTM Regressor ────────────────────────────────────────────────────────
        print("\n-- LSTM Regressor --")
        set_seed(seed)
        lstm_reg = LSTMRegressor(n_features, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT).to(DEVICE)
        reg_train_ld, reg_val_ld, reg_test_ld = make_loaders(
            X_scaled, y_reg, train_idx, val_idx, test_idx,
        )
        run_training(lstm_reg, "LSTM Reg", reg_train_ld, reg_val_ld,
                     reg_crit, task="regression")
        _, tp_reg, tl_reg = eval_loader(lstm_reg, reg_test_ld, reg_crit, apply_sigmoid=False)
        rmse = float(np.sqrt(mean_squared_error(tl_reg, tp_reg)))
        print(f"  test RMSE = {rmse:.6f}")
        records.append({"model": "LSTM Regressor", "seed": seed,
                        "test_balanced_accuracy": float("nan"), "test_rmse": rmse})

        # ── Transformer Regressor ─────────────────────────────────────────────────
        print("\n-- Transformer Regressor --")
        set_seed(seed)
        tf_reg = TransformerRegressor(n_features, TF_D_MODEL, TF_NHEAD, TF_NUM_LAYERS,
                                      TF_DIM_FF, TF_DROPOUT, max_seq_len=SEQ_LEN + 10).to(DEVICE)
        reg_train_ld, reg_val_ld, reg_test_ld = make_loaders(
            X_scaled, y_reg, train_idx, val_idx, test_idx,
        )
        run_training(tf_reg, "TF Reg", reg_train_ld, reg_val_ld,
                     reg_crit, task="regression")
        _, tp_reg, tl_reg = eval_loader(tf_reg, reg_test_ld, reg_crit, apply_sigmoid=False)
        rmse = float(np.sqrt(mean_squared_error(tl_reg, tp_reg)))
        print(f"  test RMSE = {rmse:.6f}")
        records.append({"model": "Transformer Regressor", "seed": seed,
                        "test_balanced_accuracy": float("nan"), "test_rmse": rmse})

    # ── Save and summarise ────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_PATH}")

    print("\n" + "="*60)
    print("SUMMARY  (mean ± std across seeds)")
    print("="*60)

    clf_models = ["MLP Classifier", "LSTM Classifier", "Transformer Classifier"]
    reg_models = ["MLP Regressor",  "LSTM Regressor",  "Transformer Regressor"]

    print("\nClassification — test balanced accuracy:")
    for m in clf_models:
        vals = df.loc[df["model"] == m, "test_balanced_accuracy"].values
        print(f"  {m:<28s}  {vals.mean():.4f} ± {vals.std():.4f}  "
              f"  (min={vals.min():.4f}  max={vals.max():.4f})")

    print("\nRegression — test RMSE:")
    for m in reg_models:
        vals = df.loc[df["model"] == m, "test_rmse"].values
        print(f"  {m:<28s}  {vals.mean():.6f} ± {vals.std():.6f}  "
              f"  (min={vals.min():.6f}  max={vals.max():.6f})")


if __name__ == "__main__":
    main()
