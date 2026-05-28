FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5001 
CMD ["python", "app.py"]