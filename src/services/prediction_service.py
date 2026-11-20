
from src.predict import SalesPredictor
from src.services.preprocessing_service import PreprocessingService


class PredictionService:

    def __init__(self):

        self.predictor = SalesPredictor()
        self.preprocessor = PreprocessingService()

    # Existing method: accepts already-prepared features
    def generate_forecast(
        self,
        historical_features,
        future_features
    ):

        forecast = self.predictor.predict(
            historical_features,
            future_features
        )

        return forecast

    # New method: accepts raw historical data
    def generate_forecast_from_raw_data(
        self,
        raw_data,
        future_dates
    ):

        # Step 1: Prepare model inputs automatically
        historical_input, future_input = (
            self.preprocessor.prepare_prediction_inputs(
                raw_data,
                future_dates
            )
        )

        # Step 2: Generate forecast
        forecast = self.predictor.predict(
            historical_input,
            future_input
        )

        return forecast
