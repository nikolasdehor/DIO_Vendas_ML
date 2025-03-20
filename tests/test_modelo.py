"""
Testes unitários para o modelo de previsão de vendas de sorvete
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Adicionando o diretório raiz ao path para importação dos módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tentativa de importar do arquivo original
try:
    from modelo_vendas_sorvete import generate_sample_data, preprocess_data, train_and_evaluate_model
    USANDO_ARQUIVO_ORIGINAL = True
except ImportError:
    # Se falhar, importa do arquivo de backup
    from modelo_vendas_sorvete_simples import gerar_dados as generate_sample_data
    from modelo_vendas_sorvete_simples import dividir_dados as preprocess_data
    from modelo_vendas_sorvete_simples import treinar_modelo, avaliar_modelo
    USANDO_ARQUIVO_ORIGINAL = False

class TestModelo(unittest.TestCase):
    """Classe de testes para o modelo de previsão de vendas"""
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        self.seed = 42
        self.n_samples = 50
        
        # Gerando dados para teste
        self.data = generate_sample_data(n_samples=self.n_samples, seed=self.seed)
        
        # Verificando o tipo de retorno da função de pré-processamento
        if USANDO_ARQUIVO_ORIGINAL:
            self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = preprocess_data(self.data)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(self.data)
    
    def test_data_generation(self):
        """Testa se a geração de dados está funcionando corretamente"""
        # Verificando se o DataFrame foi criado com o número correto de amostras
        self.assertEqual(len(self.data), self.n_samples)
        
        # Verificando se as colunas esperadas existem
        self.assertIn('temperatura', self.data.columns)
        self.assertIn('vendas', self.data.columns)
        
        # Verificando se os valores de temperatura estão no intervalo esperado
        self.assertTrue(all(self.data['temperatura'] >= 20))
        self.assertTrue(all(self.data['temperatura'] <= 40))
    
    def test_data_preprocessing(self):
        """Testa se o pré-processamento de dados está funcionando corretamente"""
        # Verificando se a divisão treino/teste foi feita corretamente
        expected_test_size = int(0.2 * self.n_samples)
        expected_train_size = self.n_samples - expected_test_size
        
        self.assertEqual(len(self.X_train), expected_train_size)
        self.assertEqual(len(self.X_test), expected_test_size)
        self.assertEqual(len(self.y_train), expected_train_size)
        self.assertEqual(len(self.y_test), expected_test_size)
    
    def test_model_training(self):
        """Testa se o treinamento do modelo está funcionando corretamente"""
        # Treinando modelo
        if USANDO_ARQUIVO_ORIGINAL:
            model, _, _ = train_and_evaluate_model(self.X_train, self.X_test, self.y_train, self.y_test)
        else:
            model = treinar_modelo(self.X_train, self.y_train)
            
        # Verificando se o modelo foi treinado corretamente
        self.assertIsInstance(model, LinearRegression)
        
        # Verificando se os coeficientes foram calculados
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
        
        # Testando previsões
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_model_evaluation(self):
        """Testa se a avaliação do modelo está funcionando corretamente"""
        # Treinando modelo
        if USANDO_ARQUIVO_ORIGINAL:
            model, metrics, predictions = train_and_evaluate_model(self.X_train, self.X_test, self.y_train, self.y_test)
            
            # Verificando se as métricas esperadas foram calculadas
            expected_metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'CV_R2_mean', 'CV_R2_std']
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
                
            # Verificando se o R² está em um intervalo razoável (geralmente > 0.5 para este modelo)
            self.assertGreater(metrics['R2'], 0.5)
        else:
            model = treinar_modelo(self.X_train, self.y_train)
            predictions, metrics = avaliar_modelo(model, self.X_test, self.y_test)
            
            # Verificando se as métricas esperadas foram calculadas
            expected_metrics = ['MAE', 'MSE', 'RMSE', 'R2']
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
                
            # Verificando se o R² está em um intervalo razoável (geralmente > 0.5 para este modelo)
            self.assertGreater(metrics['R2'], 0.5)
            
        # Verificando se as previsões têm o formato esperado
        self.assertEqual(len(predictions), len(self.y_test))

if __name__ == '__main__':
    unittest.main()