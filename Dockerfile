FROM python:3.10-slim

WORKDIR /app

# Instalar utilidades necesarias, incluyendo wget
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    wget \
    ca-certificates \
    && apt-get clean

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Descargar el modelo TinyLlama
RUN mkdir -p models && \
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf \
         -O models/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
