
from fastapi import Request
from fastapi.responses import JSONResponse

from src.exceptions.custom_exceptions import (
    InvalidHistoricalDataError,
    InvalidFutureDatesError,
    ModelPredictionError
)


async def invalid_historical_data_handler(
    request: Request,
    exc: InvalidHistoricalDataError
):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Invalid Historical Data",
            "detail": str(exc)
        }
    )


async def invalid_future_dates_handler(
    request: Request,
    exc: InvalidFutureDatesError
):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Invalid Future Dates",
            "detail": str(exc)
        }
    )


async def model_prediction_error_handler(
    request: Request,
    exc: ModelPredictionError
):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Model Prediction Failed",
            "detail": str(exc)
        }
    )

