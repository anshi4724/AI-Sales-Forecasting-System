
class SalesForecastException(Exception):
    """Base exception for the sales forecasting application."""

    pass


class InvalidHistoricalDataError(SalesForecastException):
    """Raised when historical sales data is invalid."""

    pass


class InvalidFutureDatesError(SalesForecastException):
    """Raised when future forecast dates are invalid."""

    pass


class ModelPredictionError(SalesForecastException):
    """Raised when the forecasting model fails."""

    pass
