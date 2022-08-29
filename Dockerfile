FROM continuumio/miniconda3

WORKDIR /app

RUN conda update -n base -c defaults conda

COPY nlp_project.yml .

RUN conda env create -f nlp_project.yml

SHELL ["conda", "run", "-n", "nlp_project", "/bin/bash", "-c"]

COPY ["predict.py", "models/CV.pkl", "./"]

EXPOSE 9696

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]




