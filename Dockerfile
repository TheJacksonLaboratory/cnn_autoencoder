# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install pip requirements
RUN apt-get update && apt-get install -y \
    libblosc1 \
    wget \
    unzip \
    build-essential

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Install CompressAI
RUN wget https://github.com/InterDigitalInc/CompressAI/archive/refs/heads/master.zip && \
    unzip master.zip && \
    rm master.zip && \
    python -m pip install -e 'CompressAI-master/.[dev]'

# Install LC algorithm
RUN wget https://github.com/UCMerced-ML/LC-model-compression/archive/refs/heads/master.zip && \
    unzip master.zip && \
    rm master.zip && \
    python -m pip install -e 'LC-model-compression-master'

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["python", "src/train_cae_ms.py"]