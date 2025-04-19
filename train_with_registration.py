import mlflow
from sklearn.ensemble import RandomForestRegressor
import joblib

# Carregar dados
X_train = joblib.load('X_train.pkl')
y_train = joblib.load('y_train.pkl')

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("California-Housing-Prices")

with mlflow.start_run():
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Registrar modelo
    mlflow.sklearn.log_model(model, "model")
    result = mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "CaliforniaHousing"
    )
    
    # Marcar como produção
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="CaliforniaHousing",
        version=result.version,
        stage="Production"
    )