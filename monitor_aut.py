import pandas as pd
import numpy as np
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

def check_for_drift(drift_score, rmse_threshold=0.5):
    if drift_score > rmse_threshold:
        print(f"Drift detectado! RMSE aumentou {drift_score:.2f}. Acionando re-treinamento...")
        os.system("python retrain.py")  # Script para re-treinamento automático
    else:
        print(f"Modelo estável. RMSE atual: {drift_score:.2f} (threshold: {rmse_threshold})")

def load_reference_data():
    # Carregar dados de referência (treino)
    X_train = joblib.load('X_train.pkl')
    y_train = joblib.load('y_train.pkl')
    return pd.DataFrame(X_train, columns=joblib.load('scaler.pkl').feature_names_in_), y_train

def load_current_data():
    # Simular novos dados (substituir por sua fonte real)
    df = pd.read_csv('california_housing.csv').sample(1000)
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    return X, y

def simulate_drift(reference_data):
    # Simular drift alterando características importantes
    current_data = reference_data.copy()
    
    # Alterações para simular drift
    current_data['MedInc'] *= 1.2  # Aumento de renda média
    current_data['AveRooms'] += 0.5  # Aumento no tamanho das casas
    current_data['HouseAge'] = np.random.randint(0, 50, current_data.shape[0])  # Idade aleatória
    
    print("Drift simulado nas features principais")
    return current_data

def get_predictions(data):
    # Preparar dados para a API
    scaler = joblib.load('scaler.pkl')
    scaled_data = scaler.transform(data)
    
    # Chamar a API de predição
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    
    # Formatando para o FastAPI
    instances = [{"MedInc": float(row[0]),
                 "HouseAge": float(row[1]),
                 "AveRooms": float(row[2]),
                 "AveBedrms": float(row[3]),
                 "Population": float(row[4]),
                 "AveOccup": float(row[5]),
                 "Latitude": float(row[6]),
                 "Longitude": float(row[7])} for row in scaled_data]
    
    response = requests.post(url, json=instances, headers=headers)
    return np.array([p['predicted_price'] for p in response.json()])

def evaluate_model(reference_data, current_data):
    # Carregar dados atuais
    X_current, y_current = current_data
    
    # Obter previsões
    y_pred = get_predictions(X_current)
    
    # Criar relatório
    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset()
    ])
    
    # Preparar datasets
    reference_dataset = pd.DataFrame(reference_data[0], columns=reference_data[0].columns)
    reference_dataset['target'] = reference_data[1]
    reference_dataset['prediction'] = reference_data[0].values @ np.random.rand(8)  # Simular predições antigas
    
    current_dataset = pd.DataFrame(X_current, columns=reference_data[0].columns)
    current_dataset['target'] = y_current
    current_dataset['prediction'] = y_pred
    
    # Gerar relatórios
    report.run(reference_data=reference_dataset, current_data=current_dataset)
    report.save_html("regression_monitoring_report.html")
    
    # Extrair métricas
    report_dict = report.as_dict()
    
    # Drift de dados
    dataset_drift = report_dict['metrics'][0]['result']['dataset_drift']
    
    # Métricas de regressão
    rmse = report_dict['metrics'][1]['result']['current']['rmse']
    
    return dataset_drift, rmse

def main():
    # Carregar dados de referência
    ref_data = load_reference_data()
    
    # 1. Monitorar com dados atuais
    current_data = load_current_data()
    drift_score, rmse = evaluate_model(ref_data, current_data)
    check_for_drift(rmse)
    
    # 2. Testar com dados simulados com drift
    drifted_data = (simulate_drift(ref_data[0]), ref_data[1])
    drift_score, rmse = evaluate_model(ref_data, drifted_data)
    check_for_drift(rmse)

if __name__ == "__main__":
    main()