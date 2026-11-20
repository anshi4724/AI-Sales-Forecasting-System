
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dataset directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODEL_DIR = ROOT_DIR / "models"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

# Dataset configuration
TARGET_COLUMN = "sales"
DATE_COLUMN = "date"

STORE_ID = 1
PRODUCT_FAMILY = "GROCERY I"

# Forecasting configuration
LOOKBACK_WINDOW = 90
FORECAST_HORIZON = 16

# Model configuration
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Create directories automatically
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    ARTIFACT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
