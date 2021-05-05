from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
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
