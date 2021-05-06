# imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import set_config; set_config(display='diagram')
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, DistanceFromCenter, CalculationDirection, MinkowskiDistance
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor
import sys

import pandas as pd
import mlflow
import joblib


MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "Felix Fähnrich"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"
#EXPERIMENT_NAME = sys.argv[1]


class Trainer():
    def __init__(self, X, y, model, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.experiment_name=EXPERIMENT_NAME
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = model
        self.dist_to_center = kwargs.get('dist_to_center',False)
        self.calculation_direction = kwargs.get('calculation_direction',False)
        self.calculation_direction = kwargs.get('calculation_direction',False)
        self.manhattan_dist = kwargs.get('manhattan_dist',False)
        self.euclidian_dist = kwargs.get('euclidian_dist',False)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        #Feature Engineering
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder())
        pipe_distance = make_pipeline(DistanceTransformer(),StandardScaler())
        pipe_distance_to_center = make_pipeline(DistanceFromCenter(),StandardScaler())
        pipe_calculation_direction = make_pipeline(CalculationDirection(),StandardScaler())
        pipe_manhattan_dist = make_pipeline(MinkowskiDistance(p=1),StandardScaler())
        pipe_euclidian_dist = make_pipeline(MinkowskiDistance(p=2),StandardScaler())
        time_col = ['pickup_datetime']
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        features = [
            ('time', pipe_time, time_col),
            ('distance', pipe_distance, dist_cols)
            ]
        if self.dist_to_center == True:
            self.mlflow_log_param('feature1', 'distance_to_center')
            features.append(
                ('distance_to_center', pipe_distance_to_center, dist_cols)
                )
        if self.calculation_direction == True:
            self.mlflow_log_param('feature2', 'calculation_direction')
            features.append(
                ('calculation_direction', pipe_calculation_direction, dist_cols)
                )
        if self.manhattan_dist == True:
            self.mlflow_log_param('feature3', 'manhattan_dist')
            features.append(
                ('manhattan_dist', pipe_manhattan_dist, dist_cols)
                )
        if self.euclidian_dist == True:
            self.mlflow_log_param('feature4', 'euclidian_dist')
            features.append(
                ('euclidian_dist', pipe_euclidian_dist, dist_cols)
                )

        feat_eng_pipeline = ColumnTransformer(features)

        # Main Pipeline
        name = str(self.model)[:10]
        self.mlflow_log_param('model', name)
        self.pipeline = Pipeline([
            ('feat_eng', feat_eng_pipeline),
            ('regressor', self.model)
            ])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        print(cross_val_score(self.pipeline, self.X, self.y, cv=5))
        self.pipeline.fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        eval_res = compute_rmse(y_pred,y_test)
        self.mlflow_log_metric('rmse', eval_res)
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
        joblib.dump(self.pipeline, 'pipeline.joblib')


if __name__ == "__main__":
    # get data
    N = 10_000
    df = get_data(nrows=N)
    # clean data
    df = clean_data(df)
    # Delete 1.January 2009
    df = df[df['key'].str.contains("2009-01-01") == False]
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    trainer.evaluate(X_test, y_test)
    #save_model
    trainer.save_model()
