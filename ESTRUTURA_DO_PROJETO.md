# Estrutura do Projeto

Este documento descreve a estrutura do projeto e a função de cada arquivo.

```
DIO_Vendas_ML/
│
├── modelo_vendas_sorvete.py         # Script principal com modelo avançado (usa MLflow)
├── modelo_vendas_sorvete_simples.py # Versão simplificada do modelo (sem MLflow)
├── requirements.txt                 # Dependências do projeto
├── README.md                        # Documentação principal do projeto
├── ESTRUTURA_DO_PROJETO.md          # Este arquivo
├── run.sh                           # Script para executar o projeto (Linux/Mac)
├── run.bat                          # Script para executar o projeto (Windows)
│
├── tests/                           # Diretório de testes
│   └── test_modelo.py               # Testes unitários para o modelo
│
├── notebooks/                       # Jupyter notebooks
│   └── exploracao_dados.ipynb       # Notebook de análise exploratória
│
└── mlruns/                          # Diretório gerado pelo MLflow (criado automaticamente)
```

## Descrição dos Arquivos

### Scripts Principais

- **modelo_vendas_sorvete.py**: Implementação completa do modelo usando MLflow para tracking de experimentos, com funcionalidades avançadas como normalização de dados, validação cruzada, e visualizações detalhadas.

- **modelo_vendas_sorvete_simples.py**: Versão simplificada que não depende do MLflow. Útil como backup ou quando o MLflow não está disponível.

### Scripts de Execução

- **run.sh**: Script para sistemas Linux/Mac que configura o ambiente e executa o projeto.
- **run.bat**: Script equivalente para sistemas Windows.

### Testes

- **tests/test_modelo.py**: Contém testes unitários para verificar o funcionamento correto do modelo. Adapta-se automaticamente para testar tanto a versão completa quanto a versão simplificada.

### Documentação

- **README.md**: Documentação principal com visão geral, instruções de uso e detalhes do projeto.
- **ESTRUTURA_DO_PROJETO.md**: Este arquivo, que descreve a organização dos arquivos.

### Exploração de Dados

- **notebooks/exploracao_dados.ipynb**: Notebook Jupyter para análise exploratória dos dados, visualização das relações e entendimento do problema.

### Dependências

- **requirements.txt**: Lista todas as bibliotecas e dependências necessárias para executar o projeto.

## Como Usar Esta Estrutura

1. **Para Iniciar**: Execute `run.sh` (Linux/Mac) ou `run.bat` (Windows) para configurar automaticamente o ambiente e executar o modelo.

2. **Para Desenvolvimento**: Trabalhe principalmente com `modelo_vendas_sorvete.py` para implementar novas funcionalidades.

3. **Para Exploração**: Use o notebook `exploracao_dados.ipynb` para análises interativas.

4. **Para Testes**: Execute `python -m unittest discover tests` para rodar os testes unitários.

5. **Para MLflow**: Após a execução, acesse a interface do MLflow em `http://localhost:5000` para visualizar experimentos (se disponível).

## Fluxo de Trabalho Recomendado

1. Inicie explorando os dados no notebook
2. Modifique o modelo conforme necessário
3. Execute os testes para verificar a correção
4. Use o script principal para treinar e registrar o modelo
5. Visualize os resultados na interface do MLflow