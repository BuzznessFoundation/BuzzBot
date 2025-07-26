FROM python:3.10-slim

WORKDIR /app

# Instalar utilidades necesarias, incluyendo wget
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    wget \
    libopenblas-dev \
    libomp-dev \
    && apt-get clean

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
