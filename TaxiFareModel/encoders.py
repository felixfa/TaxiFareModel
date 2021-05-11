from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized, calculate_direction, minkowski_distance_gps
import pandas as pd


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        X_copy = X.copy()
        X_copy.index = pd.to_datetime(X_copy[self.time_column])
        X_copy.index = X_copy.index.tz_convert(self.time_zone_name)
        X_copy['dow'] = X_copy.index.weekday
        X_copy['hour'] = X_copy.index.hour
        X_copy['month'] = X_copy.index.month
        X_copy['year'] = X_copy.index.year
        return X_copy[['dow','hour','month','year']].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        X_copy = X.copy()
        X_copy['distance'] = haversine_vectorized(X,self.start_lat,self.start_lon,self.end_lat,self.end_lon)
        return X_copy[['distance']].reset_index(drop=True)

class DistanceFromCenter(BaseEstimator, TransformerMixin):
    """Compute the haversine distance from the Center."""

    def __init__(self,
        start_lat="nyc_lat",
        start_lon="nyc_lng",
        end_lat="pickup_latitude",
        end_lon="pickup_longitude"):
        self.start_lat = start_lat
        self.start_lon = end_lat
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance_to_center'"""
        X_copy = X.copy()
        nyc_center = (40.7141667, -74.0063889)
        X_copy["nyc_lat"], X_copy["nyc_lng"] = nyc_center[0], nyc_center[1]
        X_copy['distance_to_center'] = haversine_vectorized(X_copy,self.start_lat,self.start_lon,self.end_lat,self.end_lon)
        return X_copy[['distance_to_center']].reset_index(drop=True)

class CalculationDirection(BaseEstimator, TransformerMixin):
    """Compute the Direction in which the target is going."""

    def __init__(self,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = end_lat
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'Direction'"""
        X_copy = X.copy()
        X_copy['delta_lon'] = X_copy[self.start_lon] - X_copy[self.end_lon]
        X_copy['delta_lat'] = X_copy[self.start_lat] - X_copy[self.end_lat]
        X_copy['direction'] = calculate_direction(X_copy.delta_lon, X_copy.delta_lat)
        return X_copy[['direction']].reset_index(drop=True)


class MinkowskiDistance(BaseEstimator, TransformerMixin):
    """Compute the Minkowski Distance."""

    def __init__(self,
        p=1,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude"
        ):
        self.p = p
        self.start_lat = start_lat
        self.start_lon = end_lat
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'Minkowski_Distance'"""

        X_copy = X.copy()
        X_copy[[f'MinkowskiDistance_{self.p}']] = minkowski_distance_gps(X_copy[self.start_lon], X_copy[self.end_lon],
                                       X_copy[self.start_lat], X_copy[self.end_lat], self.p)
        return X_copy[[f'MinkowskiDistance_{self.p}']].reset_index(drop=True)
