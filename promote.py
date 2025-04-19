import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
model_name = "CaliforniaHousing"  # Nome do seu modelo

# Definir os limites de R² para Staging e Production
staging_threshold = 0.7  # Apenas modelos acima deste R² vão para Staging
production_threshold = 0.75  # Limite mínimo para produção

# Buscar todas as versões do modelo
versions = client.search_model_versions(f"name='{model_name}'")

best_model = None  # Para armazenar o modelo Champion
best_r2 = -float('inf')  # Inicializar com valor baixo

for version in versions:
    run_id = version.run_id
    metrics = client.get_run(run_id).data.metrics
    
    # Verificar se a métrica R² existe para esta versão
    if "r2" in metrics:
        current_r2 = metrics["r2"]
        
        # Mover para Staging se atender ao critério
        if current_r2 > staging_threshold:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Staging"
            )
            print(f"Versão {version.version} movida para Staging (R² = {current_r2:.3f})")
        
        # Atualizar melhor modelo se superar o threshold de produção
        if current_r2 > production_threshold and current_r2 > best_r2:
            best_r2 = current_r2
            best_model = version.version

# Promover melhor modelo para Production
if best_model:
    client.transition_model_version_stage(
        name=model_name,
        version=best_model,
        stage="Production"
    )
    print(f"\n🔥 Novo modelo em Production: Versão {best_model} (R² = {best_r2:.3f})")
else:
    print("\n⚠️ Nenhum modelo atendeu aos critérios para produção!")

# Listar status final
print("\nStatus final dos modelos:")
for version in versions:
    if version.current_stage:
        print(f"Versão {version.version}: {version.current_stage} (R² = {client.get_run(version.run_id).data.metrics.get('r2','N/A'):.3f})")