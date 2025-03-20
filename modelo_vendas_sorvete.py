import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=100, seed=42):
    """
    Gera dados simulados de temperatura e vendas.
    
    Args:
        n_samples: Número de amostras a serem geradas
        seed: Semente para reprodutibilidade
        
    Returns:
        DataFrame com dados simulados
    """
    np.random.seed(seed)
    data = pd.DataFrame({
        'temperatura': np.random.uniform(20, 40, n_samples),  # Temperaturas entre 20 e 40 graus
    })
    # Relação linear com ruído
    data['vendas'] = 10 + 2.5 * data['temperatura'] + np.random.normal(0, 5, n_samples)
    return data

def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Pré-processa os dados, separando em conjuntos de treino e teste e aplicando normalização.
    
    Args:
        data: DataFrame com os dados
        test_size: Proporção do conjunto de teste
        random_state: Semente para reprodutibilidade
        
    Returns:
        Tupla contendo dados de treino e teste (X_train, X_test, y_train, y_test, scaler)
    """
    # Dividindo os dados
    X = data[['temperatura']]
    y = data['vendas']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalizando os dados (geralmente importante para modelos mais complexos)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertendo arrays para DataFrames mantendo os nomes das features
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Treina o modelo e calcula métricas de avaliação.
    
    Args:
        X_train, X_test, y_train, y_test: Dados de treino e teste
        
    Returns:
        Tupla contendo o modelo treinado e um dicionário com as métricas
    """
    # Criando e treinando o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Avaliação com validação cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Fazendo previsões
    y_pred = model.predict(X_test)
    
    # Calculando métricas
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "CV_R2_mean": np.mean(cv_scores),
        "CV_R2_std": np.std(cv_scores)
    }
    
    return model, metrics, y_pred

def plot_results(X_test, y_test, y_pred, metrics, save_path=None):
    """
    Visualiza os resultados do modelo.
    
    Args:
        X_test: Features de teste
        y_test: Valores reais
        y_pred: Previsões do modelo
        metrics: Dicionário com métricas de avaliação
        save_path: Caminho para salvar a figura (opcional)
    """
    # Configurando o estilo
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Gráfico principal
    plt.subplot(2, 1, 1)
    plt.scatter(X_test, y_test, alpha=0.7, label="Dados Reais")
    plt.plot(X_test, y_pred, color='red', linewidth=2, label="Previsão")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Vendas")
    plt.title("Previsão de Vendas de Sorvete baseada na Temperatura")
    plt.legend()
    
    # Gráfico de resíduos
    plt.subplot(2, 1, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True)
    plt.xlabel("Resíduos")
    plt.title("Distribuição dos Resíduos")
    
    # Adicionar texto com as métricas
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.figtext(0.15, 0.05, metrics_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Figura salva em {save_path}")
    
    plt.show()

def log_model_to_mlflow(model, metrics, scaler=None, experiment_name="previsao_vendas_sorvete"):
    """
    Registra o modelo e métricas no MLflow.
    
    Args:
        model: Modelo treinado
        metrics: Dicionário com métricas de avaliação
        scaler: StandardScaler usado para normalização (opcional)
        experiment_name: Nome do experimento no MLflow
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Registrando parâmetros
        mlflow.log_param("model_type", type(model).__name__)
        
        # Registrando métricas
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # Salvando o modelo
        mlflow.sklearn.log_model(model, "modelo_sorvete")
        
        # Salvando o scaler se fornecido
        if scaler is not None:
            mlflow.sklearn.log_model(scaler, "scaler")
        
        # Salvando gráfico
        plot_results(X_test, y_test, y_pred, metrics, save_path="model_performance.png")
        mlflow.log_artifact("model_performance.png")
        
        logger.info(f"Modelo registrado no MLflow com R²: {metrics['R2']:.4f}")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    # Gerando dados simulados
    data = generate_sample_data(n_samples=100)
    
    # Pré-processando os dados
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Treinando e avaliando o modelo
    model, metrics, y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Visualizando resultados
    plot_results(X_test, y_test, y_pred, metrics)
    
    # Registrando no MLflow
    log_model_to_mlflow(model, metrics, scaler)