# ECM3175 Dissertation — Stock Market Direction Prediction

Undergraduate dissertation project (University of Exeter, 2025–26) investigating whether machine learning models trained on daily equity features can predict the next-day directional movement of the S&P 500 (SPY ETF).

## Research Question

Can LSTM, Transformer, MLP, XGBoost, and Logistic Regression models trained on daily equity market features reliably predict SPY next-day direction, and how do they compare against a buy-and-hold benchmark?

## Project Structure

```
DissWork/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and data cleaning
│   ├── 02_feature_engineering.ipynb    # Feature construction and target labelling
│   └── 03_model_training.ipynb         # Model training, evaluation, and comparison
├── data/
│   ├── processed/                      # Cleaned price and volume data
│   └── features/                       # Engineered feature dataset
├── models/
│   └── trained/                        # Saved model weights and training histories
├── results/
│   ├── figures/model_comparison/       # SVG evaluation figures
│   └── tables/                         # CSV result tables
├── scripts/                            # LEAN algorithmic trading strategy
├── config.py                           # Shared constants (paths, symbols, split dates)
└── requirements.txt
```

## Models

| Model | Architecture | Task |
|---|---|---|
| Logistic Regression | sklearn linear baseline | Classification |
| MLP | PyTorch 128→64→1, dropout 0.5 | Classification + Regression |
| XGBoost | Gradient-boosted trees | Classification + Regression |
| LSTM | PyTorch, hidden=32, 1 layer | Classification + Regression |
| Transformer | PyTorch, d_model=32, 1 encoder layer | Classification + Regression |

All deep learning models share the same training loop: Adam optimiser, `ReduceLROnPlateau`, early stopping (patience=15), gradient clipping (max norm=1).

## Data Splits (chronological)

| Split | Date range | Rows |
|---|---|---|
| Train | 2010-01-01 – 2016-12-30 | ~1 760 |
| Validation | 2017-01-01 – 2018-12-31 | 502 |
| Test | 2019-01-01 – 2021-12-31 | 565 |

Target asset: **SPY** (S&P 500 ETF). Auxiliary context: QQQ, IWM, AAPL, EEM, USO.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order: `01` → `02` → `03`.

> **Note:** GPU is optional. The training loop falls back to CPU automatically. CUDA 12.6 build of PyTorch is listed in requirements; install the CPU-only build if no NVIDIA GPU is available (`pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu`).

## Key Results

Classification (balanced accuracy on held-out test set):

| Model | Balanced Accuracy | AUC-ROC |
|---|---|---|
| XGBoost | 0.536 | — |
| MLP (PyTorch) | 0.509 | — |
| LSTM | — | — |
| Transformer | — | — |
| Logistic Regression | — | — |

*(Full results in `results/tables/evaluation_summary.csv`)*

All regression models converge near the historical mean (R² ≈ 0); the PyTorch MLP achieves the highest R² (+0.010).
