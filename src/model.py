
import torch
import torch.nn as nn

from src.config import (
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    FORECAST_HORIZON
)


class ImprovedSalesLSTM(nn.Module):

    def __init__(
        self,
        historical_input_size=6,
        future_input_size=4
    ):
        super().__init__()

        # 1. LSTM: Process historical sales and features
        self.lstm = nn.LSTM(
            input_size=historical_input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        # 2. Future feature layer: Process calendar information
        self.future_fc = nn.Linear(
            future_input_size,
            HIDDEN_SIZE
        )

        # 3. Output network: Generate sales predictions
        self.output = nn.Sequential(
            nn.Linear(
                HIDDEN_SIZE * 2,
                HIDDEN_SIZE
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                HIDDEN_SIZE,
                1
            )
        )

    def forward(self, x_history, x_future):

        # Process historical data through LSTM
        _, (hidden_state, _) = self.lstm(x_history)

        # Extract the final hidden state
        context = hidden_state[-1]

        # Repeat historical context for each forecast day
        context = context.unsqueeze(1).expand(
            -1,
            x_future.size(1),
            -1
        )

        # Process future calendar features
        future_embedding = torch.relu(
            self.future_fc(x_future)
        )

        # Combine historical context and future features
        combined = torch.cat(
            [context, future_embedding],
            dim=-1
        )

        # Generate predictions
        predictions = self.output(combined).squeeze(-1)

        return predictions
