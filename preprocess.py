import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Carregar dados
df = pd.read_csv('california_housing.csv')

# =================================================
# Análise Exploratória dos Dados (EDA)
# =================================================

print("\n" + "="*40)
print("Análise Exploratória dos Dados (EDA)")
print("="*40 + "\n")

# 1. Visualização inicial dos dados
print("1. Primeiras linhas do dataset:")
print(df.head())
print("\n" + "-"*40 + "\n")

# 2. Estrutura do dataset
print(f"2. Dimensões do dataset: {df.shape}")
print(f"Número de registros: {df.shape[0]}")
print(f"Número de variáveis: {df.shape[1]}")
print("\n" + "-"*40 + "\n")

# 3. Tipos de dados e valores ausentes
print("3. Tipos de dados e valores ausentes:")
print(df.info())
print("\nValores ausentes por coluna:")
print(df.isnull().sum())
print("\n" + "-"*40 + "\n")

# 4. Estatísticas descritivas
print("4. Estatísticas descritivas:")
print(df.describe().transpose())
print("\n" + "-"*40 + "\n")

# 5. Distribuição da variável target
plt.figure(figsize=(10, 6))
sns.histplot(df['MedHouseVal'], kde=True)
plt.title('Distribuição dos Preços Médios das Casas (MedHouseVal)')
plt.xlabel('Preço (unidades de $100,000)')
plt.ylabel('Frequência')
plt.savefig('target_distribution.png')
plt.close()
print("5. Gráfico de distribuição do target salvo como 'target_distribution.png'")
print("-"*40 + "\n")

# 6. Correlação entre variáveis
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.savefig('correlation_matrix.png')
plt.close()
print("6. Matriz de correlação salva como 'correlation_matrix.png'")
print("-"*40 + "\n")

# 7. Distribuição das features
numerical_features = df.columns.drop('MedHouseVal')
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribuição de {col}')
plt.tight_layout()
plt.savefig('features_distribution.png')
plt.close()
print("7. Distribuição das features salva como 'features_distribution.png'")
print("-"*40 + "\n")

# =================================================
# Pré-processamento
# =================================================

print("\n" + "="*40)
print("Iniciando Pré-processamento")
print("="*40 + "\n")

# Separar features (X) e target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Dividir em treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Normalizar dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvar artefatos
joblib.dump(X_train_scaled, 'X_train.pkl')
joblib.dump(X_test_scaled, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Pré-processamento concluído! ✅")
print("Artefatos salvos:")
print("- X_train.pkl\n- X_test.pkl\n- y_train.pkl\n- y_test.pkl\n- scaler.pkl")