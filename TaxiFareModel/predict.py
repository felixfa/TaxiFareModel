import joblib
import pandas as pd
from google.cloud import storage
from TaxiFareModel.params import STORAGE_LOCATION, BUCKET_NAME


# load the model from disk
client = storage.Client().bucket(BUCKET_NAME)
blob = client.blob(STORAGE_LOCATION + "/model_xgboost.joblib")
blob.download_to_filename("model_xgboost.joblib")
print("=> pipeline downloaded from storage")
loaded_model = joblib.load("model_xgboost.joblib")


#loaded_model = joblib.load('/Users/Felix/Downloads/models_pipeline.joblib')

test_sample = pd.read_csv('../raw_data/test.csv')
y_pred = loaded_model.predict(test_sample)

test_sample['fare_amount']=pd.Series(y_pred)
eval_df=test_sample[['key','fare_amount']]

eval_df.to_csv('predictions_test_ex.csv', index=False)



# gcloud projects get-iam-policy wagon-bootcam-313109-2a767146925e \
# --flatten="bindings[].members" \
# --format='table(bindings.role)' \
# --filter="bindings.members: felix-f-hnrich@wagon-bootcam-313109.iam.gserviceaccount.com

# def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=False):
#     client = storage.Client().bucket(bucket)
#     storage_location = "models/taxifare/final_model.joblib"
#     # storage_location = 'models/{}/versions/{}/{}'.format(
#     #    MODEL_NAME,
#     #    model_directory,
#     #    'model.joblib')
#     blob = client.blob(storage_location)
#     blob.download_to_filename("model.joblib")
#     print("=> pipeline downloaded from storage")
#     model = joblib.load("model.joblib")
#     if rm:
#         os.remove("model.joblib")
#     return model
