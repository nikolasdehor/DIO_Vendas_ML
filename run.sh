#!/bin/bash

# Script para executar o projeto de previsão de vendas

# Cores para saída
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}MLflow não encontrado. Executando versão simplificada...${NC}"
    python modelo_vendas_sorvete_simples.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}Erro ao executar modelo.${NC}"
        exit 1
    else
        echo -e "${GREEN}Modelo executado com sucesso (versão simplificada).${NC}\n"
    fi
fi

# Executando testes
echo -e "${YELLOW}Executando testes...${NC}"
python -m unittest discover tests
if [ $? -ne 0 ]; then
    echo -e "${RED}Alguns testes falharam. Verifique os erros acima.${NC}"
else
    echo -e "${GREEN}Todos os testes passaram com sucesso.${NC}\n"
fi

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Projeto executado com sucesso!${NC}"
echo -e "${GREEN}Os resultados e gráficos foram salvos no diretório atual.${NC}"
echo -e "${GREEN}==================================================${NC}"

# Desativando ambiente virtual
deactivate===== Projeto de Previsão de Vendas de Sorvete =====${NC}\n"

# Verificando se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Erro: Python3 não está instalado.${NC}"
    exit 1
fi

# Verificando se o ambiente virtual já existe
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Criando ambiente virtual...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Erro ao criar ambiente virtual.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Ambiente virtual criado com sucesso.${NC}\n"
fi

# Ativando ambiente virtual
echo -e "${YELLOW}Ativando ambiente virtual...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao ativar ambiente virtual.${NC}"
    exit 1
fi
echo -e "${GREEN}Ambiente virtual ativado.${NC}\n"

# Instalando dependências
echo -e "${YELLOW}Instalando dependências...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao instalar dependências.${NC}"
    exit 1
fi
echo -e "${GREEN}Dependências instaladas com sucesso.${NC}\n"

# Verificando se MLflow está disponível
if command -v mlflow &> /dev/null; then
    echo -e "${YELLOW}Executando modelo com MLflow...${NC}"
    python modelo_vendas_sorvete.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Erro ao executar com MLflow. Tentando versão simplificada...${NC}"
        python modelo_vendas_sorvete_simples.py
    else
        echo -e "${GREEN}Modelo executado com sucesso usando MLflow.${NC}\n"
        
        # Iniciando servidor MLflow
        echo -e "${YELLOW}Iniciando servidor MLflow UI...${NC}"
        echo -e "${YELLOW}Acesse http://localhost:5000 para visualizar os experimentos.${NC}"
        echo -e "${YELLOW}Pressione Ctrl+C para encerrar o servidor quando terminar.${NC}\n"
        mlflow ui
    fi
else
    echo -e "${YELLOW}