# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN apt-get update && apt-get install build-essential git -y
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN git clone https://github.com/TheJacksonLaboratory/zarrdataset.git
RUN python -m pip install -e ./zarrdataset

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "src/train_cae_ms.py"]
