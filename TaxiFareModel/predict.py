import joblib
import pandas as pd
from TaxiFareModel.utils import compute_rmse


# load the model from disk
loaded_model = joblib.load('../pipeline.joblib')
test_sample = pd.read_csv('../raw_data/test.csv')
y_pred = loaded_model.predict(test_sample)

test_sample['fare_amount']=pd.Series(y_pred)
eval_df=test_sample[['key','fare_amount']]

eval_df.to_csv('predictions_test_ex.csv', index=False)
