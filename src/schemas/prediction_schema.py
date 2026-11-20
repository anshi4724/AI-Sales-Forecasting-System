
import math
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):

    historical_features: list[list[float]] = Field(
        ...,
        description="90 historical days with 6 features per day"
    )

    future_features: list[list[float]] = Field(
        ...,
        description="16 future days with 4 features per day"
    )

    @field_validator("historical_features")
    @classmethod
    def validate_historical_features(cls, value):

        if len(value) != 90:
            raise ValueError(
                "Historical data must contain exactly 90 days"
            )

        for row in value:

            if len(row) != 6:
                raise ValueError(
                    "Each historical day must contain 6 features"
                )

            if not all(math.isfinite(x) for x in row):
                raise ValueError(
                    "Historical features must contain finite numbers"
                )

        return value

    @field_validator("future_features")
    @classmethod
    def validate_future_features(cls, value):

        if len(value) != 16:
            raise ValueError(
                "Future data must contain exactly 16 days"
            )

        for row in value:

            if len(row) != 4:
                raise ValueError(
                    "Each future day must contain 4 features"
                )

            if not all(math.isfinite(x) for x in row):
                raise ValueError(
                    "Future features must contain finite numbers"
                )

        return value


class PredictionResponse(BaseModel):

    forecast: list[float]
    forecast_horizon: int
    message: str


class RawPredictionRequest(BaseModel):

    historical_data: list[dict] = Field(
        ...,
        description="90 historical records containing date, sales and onpromotion"
    )

    future_dates: list[str] = Field(
        ...,
        description="16 consecutive future dates"
    )

    @field_validator("historical_data")
    @classmethod
    def validate_historical_data(cls, value):

        if len(value) != 90:
            raise ValueError(
                "Exactly 90 historical records are required"
            )

        return value

    @field_validator("future_dates")
    @classmethod
    def validate_future_dates(cls, value):

        if len(value) != 16:
            raise ValueError(
                "Exactly 16 future dates are required"
            )

        return value
