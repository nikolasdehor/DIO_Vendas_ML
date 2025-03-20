"""
Modelo Simplificado para Previsão de Vendas de Sorvete
Versão de backup que não depende de MLflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando estilo de visualização
sns.set(style="whitegrid")

def gerar_dados(n_amostras=100, seed=42):
    """Gera dados simulados de temperatura e vendas"""
    np.random.seed(seed)
    data = pd.DataFrame({
        'temperatura': np.random.uniform(20, 40, n_amostras),
    })
    data['vendas'] = 10 + 2.5 * data['temperatura'] + np.random.normal(0, 5, n_amostras)
    return data

def dividir_dados(data, test_size=0.2, random_state=42):
    """Divide os dados em conjuntos de treino e teste"""
    X = data[['temperatura']]
    y = data['vendas']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def treinar_modelo(X_train, y_train):
    """Treina o modelo de regressão linear"""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo treinado e retorna métricas"""
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metricas = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
    }
    
    print("\n===== Avaliação do Modelo =====")
    print(f"Coeficiente (inclinação): {modelo.coef_[0]:.4f}")
    print(f"Intercepto: {modelo.intercept_:.4f}")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.4f}")
    
    return y_pred, metricas

def visualizar_resultados(X_test, y_test, y_pred, metricas, modelo=None):
    """Visualiza os resultados do modelo"""
    plt.figure(figsize=(12, 10))
    
    # Gráfico de dispersão com linha de regressão
    plt.subplot(2, 1, 1)
    plt.scatter(X_test, y_test, alpha=0.7, label="Dados Reais")
    plt.plot(X_test, y_pred, color='red', linewidth=2, label="Previsão")
    
    plt.title("Previsão de Vendas de Sorvete baseada na Temperatura")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Vendas")
    plt.legend()
    
    # Gráfico de resíduos
    plt.subplot(2, 1, 2)
    residuos = y_test - y_pred
    sns.histplot(residuos, kde=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Resíduos")
    plt.title("Distribuição dos Resíduos")
    
    # Adicionar texto com as métricas
    metricas_texto = "\n".join([f"{k}: {v:.4f}" for k, v in metricas.items()])
    plt.figtext(0.15, 0.05, metricas_texto, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salvar gráfico
    plt.savefig("resultados_modelo.png")
    plt.show()
    
    print("\nGráfico de resultados salvo como 'resultados_modelo.png'")

def exportar_modelo(modelo, filename="modelo_sorvete.pkl"):
    """Exporta o modelo treinado para um arquivo"""
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(modelo, file)
    print(f"\nModelo salvo como '{filename}'")

def main():
    print("==== Previsão de Vendas de Sorvete com Machine Learning ====")
    
    # Gerar dados
    print("\nGerando dados simulados...")
    data = gerar_dados(n_amostras=100)
    print(f"Dados gerados: {len(data)} amostras")
    
    # Dividir dados
    print("\nDividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = dividir_dados(data)
    print(f"Tamanho do conjunto de treino: {len(X_train)} amostras")
    print(f"Tamanho do conjunto de teste: {len(X_test)} amostras")
    
    # Treinar modelo
    print("\nTreinando modelo de regressão linear...")
    modelo = treinar_modelo(X_train, y_train)
    
    # Avaliar modelo
    print("\nAvaliando modelo...")
    y_pred, metricas = avaliar_modelo(modelo, X_test, y_test)
    
    # Visualizar resultados
    print("\nVisualizando resultados...")
    visualizar_resultados(X_test, y_test, y_pred, metricas, modelo)
    
    # Exportar modelo
    exportar_modelo(modelo)
    
    print("\nProcesso finalizado com sucesso!")

if __name__ == "__main__":
    main()