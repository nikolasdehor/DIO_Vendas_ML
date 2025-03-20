# 🍦 Prevendo Vendas de Sorvete com Machine Learning

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## 📋 Sobre o Projeto

Este projeto faz parte do desafio da Digital Innovation One (DIO) para demonstrar como utilizar Machine Learning para prever vendas de sorvete com base na temperatura. O projeto implementa um modelo de regressão linear para estabelecer a relação entre a temperatura ambiente e o volume de vendas, utilizando MLflow para o tracking de experimentos.

## 🚀 Funcionalidades

- **Geração de Dados Simulados**: Criação de um dataset que simula a relação entre temperatura e vendas de sorvete
- **Pré-processamento de Dados**: Divisão em conjuntos de treino/teste e normalização
- **Treinamento de Modelo**: Implementação de regressão linear com scikit-learn
- **Validação Cruzada**: Avaliação robusta do desempenho do modelo
- **Visualização de Resultados**: Gráficos informativos com matplotlib e seaborn
- **Tracking de Experimentos**: Integração com MLflow para registro de métricas e artefatos
- **Logging**: Sistema de logs para monitoramento da execução

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem de programação principal
- **Pandas & NumPy**: Manipulação e processamento de dados
- **Scikit-learn**: Implementação do modelo de machine learning
- **Matplotlib & Seaborn**: Visualização de dados
- **MLflow**: Tracking de experimentos e versionamento de modelos
- **Logging**: Registro de eventos durante a execução

## 📊 Estrutura do Projeto

```
DIO_Vendas_ML/
│
├── modelo_vendas_sorvete.py       # Script principal com o modelo
├── README.md                      # Documentação do projeto
├── requirements.txt               # Dependências do projeto
├── mlruns/                        # Armazena logs e artefatos do MLflow
└── notebooks/                     # Jupyter notebooks para exploração
    └── exploracao_dados.ipynb     # Notebook de análise exploratória
```

## 💻 Como Usar

### Pré-requisitos

```bash
# Clone este repositório
git clone https://github.com/seu-usuario/DIO_Vendas_ML.git

# Entre na pasta do projeto
cd DIO_Vendas_ML

# Instale as dependências
pip install -r requirements.txt
```

### Executando o Projeto

```bash
# Execute o script principal
python modelo_vendas_sorvete.py
```

### Visualizando Experimentos no MLflow

```bash
# Inicie o servidor MLflow UI
mlflow ui
```
Acesse `http://localhost:5000` para visualizar os experimentos.

## 📈 Fluxo de Trabalho do Modelo

1. **Geração de Dados Simulados**:
   - Temperaturas entre 20°C e 40°C
   - Vendas calculadas com uma relação linear e ruído gaussiano

2. **Pré-processamento**:
   - Divisão em dados de treino (80%) e teste (20%)
   - Normalização das features usando StandardScaler

3. **Treinamento do Modelo**:
   - Regressão Linear para modelar a relação temperatura-vendas
   - Validação cruzada com 5 folds para avaliar robustez

4. **Avaliação**:
   - Métricas: MAE, MSE, RMSE, R²
   - Análise de resíduos para verificar pressupostos do modelo

5. **Registro no MLflow**:
   - Parâmetros do modelo
   - Métricas de desempenho
   - Artefatos (modelo treinado, scaler, gráficos)

## 📊 Métricas Avaliadas

- **MAE (Mean Absolute Error)**: Erro médio absoluto das previsões
- **MSE (Mean Squared Error)**: Erro quadrático médio
- **RMSE (Root Mean Squared Error)**: Raiz quadrada do erro quadrático médio
- **R²**: Coeficiente de determinação (qualidade do ajuste)
- **Validação Cruzada**: Média e desvio padrão do R² em 5 folds

## 🔍 Resultados e Insights

O modelo demonstra uma forte correlação entre a temperatura e as vendas de sorvete, como esperado. A visualização mostra:

- Relação linear clara entre temperatura e vendas
- Distribuição dos resíduos próxima à normal
- R² típico acima de 0.8, indicando bom ajuste

## 🚀 Próximos Passos

- [ ] Implementar modelos mais complexos (Random Forest, XGBoost)
- [ ] Adicionar mais variáveis preditivas (dia da semana, feriados, etc.)
- [ ] Criar uma API para disponibilizar o modelo usando FastAPI
- [ ] Implementar CI/CD para automação de testes e deploy
- [ ] Adicionar dados reais de vendas para substituir os simulados

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Faça commit das suas alterações (`git commit -m 'Adiciona nova feature'`)
4. Faça push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- Digital Innovation One (DIO) pelo desafio
- Comunidade de Data Science e Machine Learning pelos recursos e inspiração

---

Desenvolvido como parte do desafio da DIO para aplicar conceitos de Machine Learning 🚀

```bash
# Para dúvidas ou sugestões
email@exemplo.com
```