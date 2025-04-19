# MLOps Pipeline para Previsão de Preços de Imóveis

## Descrição
Este projeto implementa um pipeline completo de MLOps para previsão de preços de imóveis na Califórnia, seguindo práticas do MLops, versionamento, rastreabilidade e monitoramento contínuo. Através de scripts , o pipeline cobre desde a obtenção e pré-processamento dos dados até o deploy em produção via API, monitoramento de data drift e re-treinamento condicional.

## Tecnologias Utilizadas
- **Linguagem:** Python 3.10 (conflito de versões se for outra versão)
- **Plataforma:** Windows 11
- **Framework API:** FastAPI
- **Monitoramento:** Evidently AI
- **Versionamento e Experimentos:** MLflow (SQLite como backend)
- **Manipulação de Dados:** pandas, NumPy
- **Machine Learning:** scikit-learn, XGBoost
- **Serialização:** joblib
- **Visualização:** matplotlib, seaborn

## Pré-requisitos
1. Python 3.10 instalado no Windows 11
2. Git instalado para clonar o repositório
3. Ambiente virtual (`venv`) para isolar dependências

## Instalação e Configuração
1. **Clone o repositório**
   ```powershell
   git clone https://github.com/lucaswodtke/Mlops
   ```

2. **Crie e ative um ambiente virtual**
   ```powershell
   python -m venv .\.venv
   .\.venv\Scripts\activate
   ```

3. **Instale as dependências**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure o MLflow**
   - O backend padrão está configurado em `sqlite:///mlflow.db`.
   - Para iniciar a interface web do MLflow:
     ```powershell
     mlflow ui --backend-store-uri sqlite:///mlflow.db
     ```
   - Acesse em seu navegador: `http://localhost:5000`

## Estrutura de Diretórios
```
├── app.py                  # API FastAPI para previsões
├── load_data.py            # Fetch e export do dataset original
├── preprocess.py           # EDA e pré-processamento
├── train.py                # Treinamento de múltiplos modelos e registro no MLflow
├── promote.py              # Script de promoção automática no MLflow Model Registry
├── monitor.py              # Monitoramento de data drift e re-treinamento condicional (simplificado)
├── requirements.txt        # Dependências do projeto
├── california_housing.csv  # Dataset bruto gerado por load_data.py
├── scaler.pkl              # Artefato do StandardScaler
├── X_train.pkl, X_test.pkl # Dados pré-processados
├── mlflow.db               # Banco de dados SQLite para MLflow
└── regression_monitoring_report.html # Relatório de monitoramento Evidently
├── monitor_aut.py              # Monitoramento de data drift e re-treinamento condicional automatico (FINS DE TESTE)
├── train_with_registration.py # Exemplo de treino e registro no Model Registry (FINS DE TESTE)
├── trainlocal.py  # Exemplo de Configuração do MLflow para armazenamento local (FINS DE TESTE)
├── test_model.py  # Teste do modelo em produção (FINS DE TESTE)
├── models.py  # Consulta aos modelos registrados no MLflow (FINS DE TESTE)
```

## Como Executar o Pipeline
1. **Obter e salvar dados**
   ```powershell
   python load_data.py
   ```

2. **Executar pré-processamento**
   ```powershell
   python preprocess.py
   ```

3. **Treinar e registrar modelos**
   ```powershell
   python train.py
   ```

4. **Promover o melhor modelo para produção**
   ```powershell
   python promote.py
   ```

5. **Iniciar API de predição**
   ```powershell
   uvicorn app:app --reload
   ```
   - URL de produção local: `http://127.0.0.1:8000/predict`

6. **Monitorar drift e re-treinamento**
   ```powershell
   python monitor.py
   ```
   - O relatório HTML será salvo como `regression_monitoring_report.html`.



## Autor
Lucas Henrique Gonçalves Wodtke

