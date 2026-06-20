# Usar imagen base ligera de Python
FROM python:3.11-slim

# Establecer variables de entorno para evitar buffers y escritura de bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecer directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema mínimas necesarias (por ejemplo, libgomp para ONNX Runtime en CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt e instalar dependencias de Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del servicio de inferencia
COPY . /app/

# Exponer el puerto por defecto de FastAPI
EXPOSE 8000

# Comando para iniciar la aplicación con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
