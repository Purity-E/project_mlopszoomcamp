# project_mlopszoomcamp
This repository contains all the code and everything about my project for the mlopszoomcamp course.
# Fake News Detection
## Problem Description
The internet today has become a very useful tool in getting information. The information we get usually plays a major role in defining our perception on matters such as political, health, culture things that in one way or another affects our daily lives and our relationship with other people. So it is important that the information we get is true and is not misleading in any sort of way.

But this has not been the case in quite a number of instances. Every now and then we get to hear of cases of people spreading misinformation through social media posts or blog posts. And this has been on rise of late
Quite a number of techniques have been used to curb this problem. One of the major way has been to use machine models to detect fake news articles.

In this project, we are going to build a classification model that will classify a news article as either fake or not fake. We will make use of varoius Natural language processing techniques and machine learning algorithms. This is an end-to-end machine learning project that will involve all the necessary steps that is usually carried out in machine learning project, from building the models to model registry to deploying the models to model monitoring.
## Data Used
We will make use data from the following sources;
- https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- https://www.kaggle.com/datasets/rchitic17/real-or-fake

These  datasets are for training, testing and  validation purposes. 

## Step 1 - Create notebook
## Step 2 - Experiment tracking
- Install mlflow
- Run the command 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to set a tracking server.  This command must run in the directory that contains the notebook.
- In the notebook import mlflow and run the following codes;
mlflow.set_tracking_uri("sqlite:///mlflow.db") - Connects to a tracking URI. URI provided should match the one provided in the command
mlflow.set_experiment("experiment-name") - Will set our experiment as active, if experiment doesn't exit, it will create a new one.
- Run linear regression, record and track info using mlfow
- Hyperparameter tune Decision trees and Random Forest with hyperopt. Record and track info using mlflow
- Note: Because fmin() tries to minimize the objective, the objective function must return the negative accuracy for the case of a classification algorithm.
- Do retraining with the best model and best parameters, then autolog the model.
- Model Registry: Register the models, then stage the models
- Tests the models to promote to production
## Step 3 Workflow Orchestration
- Install Prefect
- Turn the notebook into a python script
- Clean the script
- Host prefect on VM (AWS). Run command ' prefect config set PREFECT_ORION_UI_API_URL="http://<external-ip>:4200/api" '
- Start prefect orion with 'prefect orion start --host 0.0.0.0'
- From local machine, configure to hit the API with ' prefect config set PREFECT_API_URL="http://<external-ip>:4200/api" '
- Create a main function on the script and wrap it with a @flow decorator
- Add @task decorator around task functions and make sure to add .result() to it in the main function
- Run 'prefect orion start' command to access the server
- Create a local storage 'prefect storage create' (local storage)
- Deploy runs 'prefect deployment create file_name.py'
- Create agents and works queues to run scheduled runs
## Step 4 Model Deployment
* Deploy model as a webservice
* Install flask
* Install gunicorn
* Create file 'predict.py' for the flask application.
* Get the model from mlflow registry
* Create file 'test.py' for testing
* Put 'predict.py' into a flask app and test it with test.py
* Run the application with gunicorn "gunicorn --bind=0.0.0.0:9696 predict:app"
* Package app to a docker container
* Create a dockerfile



## Quick solutions
- How to shut down mlflow ui
fuser -k 5000/tcp
-How turn jupyter notebook to python script
jupyter nbconvert --to script notebookname (make sure to install nbconvert)
- How to comment a block of code in python
crtl + / (On windows)
- How to check for prefect configuration
prefect config view
- How to unset prefect configuration
prefect config unset
- How to fix '400 Bad Request'
Install latest version of prefect 2.0
- How to check prefect storage
prefect storage ls

