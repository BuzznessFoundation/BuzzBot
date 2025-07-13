FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Instala compiladores necesarios
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    && apt-get clean

# Copiar archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente primero (sin models/)
COPY . .

# Luego descarga el modelo
RUN mkdir -p models && \
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf \
         -O models/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf

# Exponer puerto
EXPOSE 8000

# Comando de ejecución
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]