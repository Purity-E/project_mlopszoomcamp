FROM python:3.9-slim

RUN pip install -U pip

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN python -m nltk.downloader popular

COPY predict.py .

COPY ["CV.pkl", "model.pkl", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]