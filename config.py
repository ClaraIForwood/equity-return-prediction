from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DATA_DIR = DATA_DIR / "features"

# Output directories
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "figures"

# Create folders automatically if they don't exist
for folder in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_DATA_DIR,
    MODEL_DIR,
    FIGURE_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ── Shared modelling constants ────────────────────────────────────────────
# Chronological split boundaries (wrap in pd.Timestamp() at point of use)
TRAIN_END_DATE = "2016-12-30"
VAL_END_DATE   = "2018-12-31"

# Target and auxiliary assets
TARGET_SYMBOL      = "SPY"
AUXILIARY_SYMBOLS  = ["QQQ", "IWM", "AAPL", "EEM", "USO"]
AUXILIARY_FEATURES = ["return_1d", "volatility_5", "volume_change"]

# Symbol filtering
MIN_OBSERVATIONS = 500
EXCLUDED_SYMBOLS = ["GOOAV", "GOOCV"]