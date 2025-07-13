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

# Copiar todo el contenido, incluyendo la carpeta models/
COPY . .

# Exponer puerto
EXPOSE 7860

# Comando de ejecución
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]