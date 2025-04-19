from fastapi import FastAPI, Body
from pydantic import BaseModel
import mlflow
import joblib
import numpy as np

app = FastAPI()

# Carregar artefatos
scaler = joblib.load('scaler.pkl')
model = mlflow.sklearn.load_model("models:/CaliforniaHousingModel/Production")

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HouseFeatures):
    input_data = np.array([[
        features.MedInc, features.HouseAge, features.AveRooms,
        features.AveBedrms, features.Population, features.AveOccup,
        features.Latitude, features.Longitude
    ]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    return {"previsao": float(prediction[0])}