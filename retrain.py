import mlflow
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from preprocess import load_and_preprocess_data
from train import train_and_log_model
from promote import main as promote_model

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)

def retrain_model():
    """Executa o pipeline completo de retreinamento"""
    try:
        logging.info("Iniciando processo de retreinamento...")
        
        # 1. Carregar e pré-processar dados atualizados
        logging.info("Carregando e pré-processando dados...")
        df = pd.read_csv('california_housing.csv')
        
        # Atualizar dados de treino/teste
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']
        
        # Divisão treino/teste (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Salvar novos dados pré-processados
        joblib.dump(X_train, 'X_train.pkl')
        joblib.dump(X_test, 'X_test.pkl')
        joblib.dump(y_train, 'y_train.pkl')
        joblib.dump(y_test, 'y_test.pkl')

        # 2. Treinar novos modelos
        logging.info("Treinando novos modelos...")
        mlflow.set_experiment("California-Housing-Regression-Retrained")
        
        # Parâmetros para retreinamento
        retrain_params = {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6
        }

        with mlflow.start_run(run_name=f"retrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # 3. Treinar e registrar modelo
            model = train_and_log_model(
                model=GradientBoostingRegressor(),
                model_name="Retrained-GradientBoosting",
                params=retrain_params,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=X.columns.tolist()
            )
            
            # 4. Promover melhor modelo
            logging.info("Verificando promoção do novo modelo...")
            promote_model()
            
            logging.info("Retreinamento concluído com sucesso!")
            return True

    except Exception as e:
        logging.error(f"Falha no retreinamento: {str(e)}")
        return False

if __name__ == "__main__":
    # Executar retreinamento e capturar status
    success = retrain_model()
    
    # Forçar saída com código de erro se falhar
    if not success:
        exit(1)