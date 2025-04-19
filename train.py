import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Configuração do MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("California-Housing-Regression")

def eval_metrics(y_true, y_pred):
    """Calcula métricas de avaliação detalhadas"""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def log_model_performance(model, X_test, y_test, feature_names):
    """Registra métricas e artefatos do modelo"""
    y_pred = model.predict(X_test)
    metrics = eval_metrics(y_test, y_pred)
    mlflow.log_metrics(metrics)
    
    # Validação cruzada
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
    mlflow.log_metrics({
        "cv_r2_mean": np.mean(cv_scores),
        "cv_r2_std": np.std(cv_scores)
    })
    
    # Importância de features para modelos baseados em árvores
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = dict(zip(feature_names, importances))
        mlflow.log_dict(feat_imp, "feature_importance.json")

def train_and_log_model(model, model_name, params, X_train, y_train, X_test, y_test, feature_names):
    """Treina e registra um modelo no MLflow"""
    with mlflow.start_run(run_name=model_name, nested=True):
        # Treinar modelo
        model.set_params(**params)
        model.fit(X_train, y_train)
        
        # Logging
        mlflow.log_params(params)
        log_model_performance(model, X_test, y_test, feature_names)
        mlflow.sklearn.log_model(model, model_name)
        
        # Registrar no Model Registry se for o melhor modelo
        if model_name == "GradientBoosting":
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
                "CaliforniaHousing"
            )

if __name__ == "__main__":
    # Carregar dados
    X_train = joblib.load('X_train.pkl')
    y_train = joblib.load('y_train.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    df = pd.read_csv('california_housing.csv')
    feature_names = df.columns.drop('MedHouseVal').tolist()

    # Lista de modelos e parâmetros
    models = [
        {
            "name": "LinearRegression",
            "model": LinearRegression(),
            "params": {
                "fit_intercept": True,
                "copy_X": True
            }
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        {
            "name": "GradientBoosting",
            "model": GradientBoostingRegressor(),
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.8,
                "random_state": 42
            }
        }
    ]

    with mlflow.start_run(run_name="Comparative Analysis"):
        # Log dataset metadata
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("features", feature_names)
        
        for model_config in models:
            train_and_log_model(
                model=model_config["model"],
                model_name=model_config["name"],
                params=model_config["params"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names
            )

    print("Treinamento concluído! Acesse o MLflow UI: http://localhost:5000")