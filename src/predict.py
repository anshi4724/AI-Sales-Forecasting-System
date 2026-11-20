
from pathlib import Path
import json
import joblib
import numpy as np
import torch

from src.model import ImprovedSalesLSTM


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "improved_lstm_best.pth"
ARTIFACT_DIR = ROOT_DIR / "artifacts"


class SalesPredictor:

    def __init__(self):

        # Use CPU for inference
        self.device = torch.device("cpu")

        # Load model metadata
        with open(ARTIFACT_DIR / "model_metadata.json", "r") as file:
            self.metadata = json.load(file)

        # Load saved scalers
        self.sales_scaler = joblib.load(
            ARTIFACT_DIR / "sales_scaler.pkl"
        )

        self.promotion_scaler = joblib.load(
            ARTIFACT_DIR / "promotion_scaler.pkl"
        )

        # Initialize model with saved configuration
        self.model = ImprovedSalesLSTM(
            historical_input_size=self.metadata["input_size"],
            future_input_size=self.metadata["future_input_size"]
        ).to(self.device)

        # Load trained model weights
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint)

        # Set model to evaluation mode
        self.model.eval()

    def predict(self, historical_features, future_features):

        historical_features = np.asarray(
            historical_features, dtype=np.float32
        )

        future_features = np.asarray(
            future_features, dtype=np.float32
        )

        # Validate input dimensions
        if historical_features.shape != (90, 6):
            raise ValueError(
                "Historical input must have shape (90, 6)"
            )

        if future_features.shape != (16, 4):
            raise ValueError(
                "Future input must have shape (16, 4)"
            )

        # Convert NumPy arrays into PyTorch tensors
        history_tensor = torch.tensor(
            historical_features
        ).unsqueeze(0).to(self.device)

        future_tensor = torch.tensor(
            future_features
        ).unsqueeze(0).to(self.device)

        # Generate predictions
        with torch.no_grad():

            scaled_predictions = self.model(
                history_tensor,
                future_tensor
            ).cpu().numpy()

        # Convert predictions back to original sales units
        sales_predictions = self.sales_scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        ).flatten()

        # Sales cannot be negative
        sales_predictions = np.maximum(sales_predictions, 0)

        return sales_predictions
