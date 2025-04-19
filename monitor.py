import pandas as pd
import numpy as np
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_drift_report():
    """Gera relatório de Data Drift entre dados atuais e de referência"""
    # 1. Carregar dados de referência (treino)
    scaler = joblib.load('scaler.pkl')
    X_train = pd.DataFrame(joblib.load('X_train.pkl'), columns=scaler.feature_names_in_)
    
    # 2. Simular dados atuais com drift
    current_data = X_train.sample(100, random_state=42).copy()
    
    # Aplicar alterações para simular drift
    current_data['MedInc'] *= np.random.uniform(1.3, 1.8, size=len(current_data))
    current_data['AveRooms'] += np.random.randint(1, 3, size=len(current_data))
    
    # 3. Configurar e gerar relatório
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=X_train,
        current_data=current_data,
        column_mapping=None
    )
    
    # 4. Salvar e exibir resultados
    report.save_html("data_drift_report.html")
    print("Relatório gerado: data_drift_report.html")
    
    # Extrair métricas principais
    drift_metrics = report.as_dict()["metrics"][0]["result"]
    print(f"\nDrift Score: {drift_metrics['dataset_drift_score']:.2%}")
    print(f"Variáveis com drift: {drift_metrics['number_of_drifted_columns']}")

if __name__ == "__main__":
    generate_drift_report()