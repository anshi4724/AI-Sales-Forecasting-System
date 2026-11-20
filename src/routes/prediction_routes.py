
from fastapi import APIRouter, HTTPException

from src.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
    RawPredictionRequest
)

from src.services.prediction_service import PredictionService


router = APIRouter(
    prefix="/predict",
    tags=["Sales Prediction"]
)

# Create the prediction service
prediction_service = PredictionService()


@router.post("/", response_model=PredictionResponse)
def predict_sales(request: PredictionRequest):

    try:
        forecast = prediction_service.generate_forecast(
            request.historical_features,
            request.future_features
        )

        return PredictionResponse(
            forecast=forecast.tolist(),
            forecast_horizon=len(forecast),
            message="Sales forecast generated successfully"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@router.post("/raw", response_model=PredictionResponse)
def predict_from_raw_data(request: RawPredictionRequest):

    try:
        forecast = prediction_service.generate_forecast_from_raw_data(
            request.historical_data,
            request.future_dates
        )

        return PredictionResponse(
            forecast=forecast.tolist(),
            forecast_horizon=len(forecast),
            message="Sales forecast generated successfully from raw data"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
    
