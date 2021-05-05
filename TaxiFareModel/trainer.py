# imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import set_config; set_config(display='diagram')
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient

import pandas as pd
import mlflow


MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "Felix Fähnrich"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.experiment_name=EXPERIMENT_NAME
        self.pipeline = None
        self.X = X
        self.y = y
        self.mlflow_log_metric('rmse', 4.5)
        self.mlflow_log_param('model', 'linear')


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder())
        pipe_distance = make_pipeline(DistanceTransformer(),StandardScaler())

        time_col = ['pickup_datetime']
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        feat_eng_pipeline = ColumnTransformer([
            ('time', pipe_time, time_col),
            ('distance', pipe_distance, dist_cols)
            ])
        model = RandomForestRegressor()
        self.pipeline = Pipeline([
            ('feat_eng', feat_eng_pipeline),
            ('regressor', model)
            ])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        eval_res = compute_rmse(y_pred,y_test)

        return eval_res

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """Save the model into a .joblib format"""



if __name__ == "__main__":
    # get data
    N = 10_000
    df = get_data(nrows=N)
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    trainer.evaluate(X_test, y_test)
