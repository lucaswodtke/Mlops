import mlflow

# Carregar modelo pelo Model Registry
model = mlflow.sklearn.load_model("models:/CaliforniaHousingModel/Production")
print("Modelo carregado com sucesso!")