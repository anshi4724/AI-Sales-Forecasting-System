
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from src.exceptions.custom_exceptions import (
    InvalidHistoricalDataError,
    InvalidFutureDatesError
)


class PreprocessingService:

    # Exact feature order used during LSTM training
    HISTORICAL_FEATURES = [
        "sales_scaled",
        "promotion_scaled",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "is_holiday"
    ]

    FUTURE_FEATURES = [
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "is_holiday"
    ]

    def __init__(self):

        ROOT_DIR = Path(__file__).resolve().parents[2]

        ARTIFACT_DIR = ROOT_DIR / "artifacts"

        # Load original fitted training scalers
        self.sales_scaler = joblib.load(
            ARTIFACT_DIR / "sales_scaler.pkl"
        )

        self.promotion_scaler = joblib.load(
            ARTIFACT_DIR / "promotion_scaler.pkl"
        )

    # 1. Validate raw historical data
    def prepare_historical_data(self, raw_data):

        if not raw_data:
            raise InvalidHistoricalDataError(
                "Historical data cannot be empty"
            )

        df = pd.DataFrame(raw_data)

        required_columns = [
            "date",
            "sales",
            "onpromotion"
        ]

        for column in required_columns:

            if column not in df.columns:
                raise InvalidHistoricalDataError(
                    f"Missing required column: {column}"
                )

        # Convert dates
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        if df["date"].isna().any():
            raise InvalidHistoricalDataError(
                "Invalid date values found"
            )

        # Convert numerical columns
        for column in ["sales", "onpromotion"]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        if df[["sales", "onpromotion"]].isna().any().any():

            raise InvalidHistoricalDataError(
                "Sales and promotion values must be numeric"
            )

        # Check finite values
        if not np.isfinite(
            df[["sales", "onpromotion"]].to_numpy()
        ).all():

            raise InvalidHistoricalDataError(
                "Sales and promotion values must be finite"
            )

        # Check negative values
        if (df[["sales", "onpromotion"]] < 0).any().any():

            raise InvalidHistoricalDataError(
                "Sales and promotion values cannot be negative"
            )

        # Check duplicate dates
        if df["date"].duplicated().any():

            raise InvalidHistoricalDataError(
                "Duplicate dates are not allowed"
            )

        # Sort chronologically
        df = df.sort_values(
            "date"
        ).reset_index(drop=True)

        # Require exactly 90 records
        if len(df) != 90:

            raise InvalidHistoricalDataError(
                "Historical data must contain exactly 90 records"
            )

        # Check consecutive daily dates
        expected_dates = pd.date_range(
            start=df["date"].iloc[0],
            periods=90,
            freq="D"
        )

        if not df["date"].reset_index(drop=True).equals(
            pd.Series(expected_dates, name="date")
        ):

            raise InvalidHistoricalDataError(
                "Historical dates must be consecutive daily records"
            )

        return df

    # 2. Generate historical calendar features
    def add_historical_features(self, df):

        df = df.copy()

        df["day_of_week"] = df["date"].dt.dayofweek

        # Cyclical weekday encoding
        df["day_of_week_sin"] = np.sin(
            2 * np.pi * df["day_of_week"] / 7
        )

        df["day_of_week_cos"] = np.cos(
            2 * np.pi * df["day_of_week"] / 7
        )

        # Weekend feature
        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(float)

        # TEMPORARY PLACEHOLDER:
        # Replace with the Ecuador holiday calendar
        # used during model training.
        df["is_holiday"] = 0.0

        return df

    # 3. Generate future calendar features
    def add_future_features(self, future_dates):

        future_df = pd.DataFrame({
            "date": pd.to_datetime(future_dates)
        })

        future_df["day_of_week"] = (
            future_df["date"].dt.dayofweek
        )

        future_df["day_of_week_sin"] = np.sin(
            2 * np.pi * future_df["day_of_week"] / 7
        )

        future_df["day_of_week_cos"] = np.cos(
            2 * np.pi * future_df["day_of_week"] / 7
        )

        future_df["is_weekend"] = (
            future_df["day_of_week"] >= 5
        ).astype(float)

        # TEMPORARY PLACEHOLDER:
        # Replace with the same Ecuador holiday calendar.
        future_df["is_holiday"] = 0.0

        return future_df[
            self.FUTURE_FEATURES
        ]

    # 4. Scale historical features
    def scale_historical_features(self, df):

        df = df.copy()

        # Scale sales
        df["sales_scaled"] = self.sales_scaler.transform(
            df[["sales"]]
        ).flatten()

        # Scale promotions
        df["promotion_scaled"] = self.promotion_scaler.transform(
            df[["onpromotion"]]
        ).flatten()

        # Select exact training feature order
        historical_data = df[
            self.HISTORICAL_FEATURES
        ]

        return historical_data.to_numpy(
            dtype=np.float32
        )

    # 5. Prepare historical model input
    def prepare_model_input(self, raw_data):

        df = self.prepare_historical_data(raw_data)

        df = self.add_historical_features(df)

        historical_input = self.scale_historical_features(df)

        return historical_input

    # 6. Prepare future model input
    def prepare_future_input(self, future_dates):

        if len(future_dates) != 16:

            raise InvalidFutureDatesError(
                "Future dates must contain exactly 16 days"
            )

        future_dates = pd.to_datetime(
            future_dates,
            errors="coerce"
        )

        if pd.isna(future_dates).any():

            raise InvalidFutureDatesError(
                "Invalid future dates found"
            )

        # Check consecutive future dates
        expected_dates = pd.date_range(
            start=future_dates[0],
            periods=16,
            freq="D"
        )

        if not pd.DatetimeIndex(future_dates).equals(
            expected_dates
        ):

            raise InvalidFutureDatesError(
                "Future dates must be consecutive daily dates"
            )

        future_df = self.add_future_features(
            future_dates
        )

        future_input = future_df[
            self.FUTURE_FEATURES
        ].to_numpy(dtype=np.float32)

        return future_input

    # 7. Prepare both model inputs
    def prepare_prediction_inputs(
        self,
        raw_data,
        future_dates
    ):

        if len(raw_data) != 90:
            raise InvalidHistoricalDataError(
                "Historical data must contain exactly 90 records"
            )

        # Validate and prepare historical features
        historical_input = self.prepare_model_input(
            raw_data
        )

        # Prepare future features
        future_input = self.prepare_future_input(
            future_dates
        )

        # Check that future starts after historical data
        historical_last_date = pd.to_datetime(
            raw_data[-1]["date"]
        )

        future_dates = pd.to_datetime(
            future_dates
        )

        expected_start = (
            historical_last_date + pd.Timedelta(days=1)
        )

        if future_dates[0] != expected_start:

            raise InvalidFutureDatesError(
                "Future dates must start the day after historical data"
            )

        return historical_input, future_input
