FROM python:3.10.10-slim
# COPY ./app app
WORKDIR /final_project_mlops/src
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libssl-dev && rm -rf /var/lib/apt/lists/*

COPY . /final_project_mlops/
RUN pip install --no-cache-dir -r /final_project_mlops/requirements.txt
CMD ["python", "train.py"]