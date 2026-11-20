
from fastapi import FastAPI

from src.routes.prediction_routes import router as prediction_router
from fastapi.middleware.cors import CORSMiddleware

from src.exceptions.custom_exceptions import (
    InvalidHistoricalDataError,
    InvalidFutureDatesError,
    ModelPredictionError
)

from src.exceptions.exception_handlers import (
    invalid_historical_data_handler,
    invalid_future_dates_handler,
    model_prediction_error_handler
)


app = FastAPI(
    title="AI Sales Forecasting API",
    description="LSTM-based sales forecasting system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register centralized exception handlers
app.add_exception_handler(
    InvalidHistoricalDataError,
    invalid_historical_data_handler
)

app.add_exception_handler(
    InvalidFutureDatesError,
    invalid_future_dates_handler
)

app.add_exception_handler(
    ModelPredictionError,
    model_prediction_error_handler
)


# Register prediction routes
app.include_router(prediction_router)


@app.get("/")
def home():
    return {
        "message": "AI Sales Forecasting API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }
