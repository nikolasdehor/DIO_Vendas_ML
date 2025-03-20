@echo off
REM Script para executar o projeto de previsão de vendas em Windows

echo ===== Projeto de Previsão de Vendas de Sorvete =====

REM Verificando se Python está instalado
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Erro: Python nao esta instalado.
    exit /b 1
)

REM Verificando se o ambiente virtual já existe
if not exist venv (
    echo Criando ambiente virtual...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Erro ao criar ambiente virtual.
        exit /b 1
    )
    echo Ambiente virtual criado com sucesso.
)

REM Ativando ambiente virtual
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Erro ao ativar ambiente virtual.
    exit /b 1
)
echo Ambiente virtual ativado.

REM Instalando dependências
echo Instalando dependencias...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Erro ao instalar dependencias.
    exit /b 1
)
echo Dependencias instaladas com sucesso.

REM Verificando se MLflow está disponível
where mlflow >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Executando modelo com MLflow...
    python modelo_vendas_sorvete.py
    if %ERRORLEVEL% NEQ 0 (
        echo Erro ao executar com MLflow. Tentando versao simplificada...
        python modelo_vendas_sorvete_simples.py
    ) else (
        echo Modelo executado com sucesso usando MLflow.
        
        REM Iniciando servidor MLflow
        echo Iniciando servidor MLflow UI...
        echo Acesse http://localhost:5000 para visualizar os experimentos.
        echo Pressione Ctrl+C para encerrar o servidor quando terminar.
        mlflow ui
    )
) else (
    echo MLflow nao encontrado. Executando versao simplificada...
    python modelo_vendas_sorvete_simples.py
    if %ERRORLEVEL% NEQ 0 (
        echo Erro ao executar modelo.
        exit /b 1
    ) else (
        echo Modelo executado com sucesso (versao simplificada).
    )
)

REM Executando testes
echo Executando testes...
python -m unittest discover tests
if %ERRORLEVEL% NEQ 0 (
    echo Alguns testes falharam. Verifique os erros acima.
) else (
    echo Todos os testes passaram com sucesso.
)

echo ==================================================
echo Projeto executado com sucesso!
echo Os resultados e graficos foram salvos no diretorio atual.
echo ==================================================

REM Desativando ambiente virtual
call deactivate