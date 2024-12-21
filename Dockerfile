FROM python:3.10.10-slim
# COPY ./app app
WORKDIR /final_project_mlops/src
# COPY . /app/
COPY . /final_project_mlops/
RUN pip install -r /final_project_mlops/requirements.txt
CMD ["python", "train.py"]