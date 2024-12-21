FROM python:3.10.10-slim
# COPY ./app app
WORKDIR /app/src
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
CMD ["python", "train.py"]