from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from google.cloud import storage

BUCKET_NAME="wagon-ml-faehnrich-566"
STORAGE_LOCATION = 'models'


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world Felix"}

@app.get("/predict_fare")
def index(pickup_datetime,
    pickup_longitude,pickup_latitude,
    dropoff_longitude,dropoff_latitude,
    passenger_count):
    array = {"key": 1,
    "pickup_datetime": pickup_datetime,
  "pickup_longitude": pickup_longitude,
  "pickup_latitude": pickup_latitude,
  "dropoff_longitude": dropoff_longitude,
  "dropoff_latitude": dropoff_latitude,
  "passenger_count": passenger_count}

    client = storage.Client().bucket(BUCKET_NAME)
    blob = client.blob(STORAGE_LOCATION + "/model_xgboost.joblib")
    blob.download_to_filename("model_xgboost.joblib")
    print("=> pipeline downloaded from storage")
    loaded_model = joblib.load("model_xgboost.joblib")
    #loaded_model = joblib.load('/Users/Felix/code/felixfa/TaxiFareModel/model_xgboost.joblib')
    X_pred = pd.DataFrame({k: [v] for k, v in array.items()})
    X_pred.iloc[:,2:6] = X_pred.iloc[:,2:6].astype('float64')
    X_pred.iloc[:,6] = X_pred.iloc[:,6].astype('int64')
    print(X_pred)
    y_pred = loaded_model.predict(X_pred)
    print(y_pred)
    return {"prediction":str(y_pred[0])}
