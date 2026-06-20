@echo off
echo ====================================================
echo Iniciando Servidor Local de MLflow (Costo Cero)
echo ====================================================
echo Dashboard disponible en: http://127.0.0.1:5000
echo ====================================================
cd /d "%~dp0"
..\.venv\Scripts\mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
pause
