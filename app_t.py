from fastapi import FastAPI
import mlflow
import joblib
import numpy as np

app = FastAPI()

# Carregar artefatos uma vez ao iniciar
scaler = joblib.load('scaler.pkl')
model = mlflow.sklearn.load_model("models:/CaliforniaHousingModel/Production")

@app.post("/predict")
def predict(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
    Longitude: float
):
    # Converter para array numpy
    input_data = np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])
    
    # Pré-processar e prever
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    return {"previsao": float(prediction[0])}