# imports

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder())

    time_col = ['pickup_datetime']
    dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    feat_eng_pipeline = ColumnTransformer([
        ('time', pipe_time, time_col),
        ('distance', pipe_distance, dist_cols)
        ])
    self.pipeline = Pipeline([
        ('feat_eng', feat_eng_pipeline),
        ('regressor', RandomForestRegressor())
        ])
    print(self.pipeline)
    pass

    def run(self):
        """set and train the pipeline"""
        pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pass


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
