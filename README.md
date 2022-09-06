# Course Project
 This project involves creating a machine learning model to detect fake news. It is an end to end machine learning project that involves all the steps from building a model to deploying the model.
 This repository contains all the code and all information about my project for the mlopszoomcamp course.

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
- Import the neccessary libraries
- Load the datasets
- Clean and prepare the data; check duplicates, missing values, imbalanced dataset
- Clean the text data; normalize the text, remove stopwords, lemmatize the words
- Split the data into train and validation set
- Encode the data
- Train and test the different models; logistic regression, Decision Trees, Random Forest
- Evaluate the models
## Step 2 - Experiment tracking
- Install mlflow
- Run the command 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to set a tracking server.  This command must run in the directory that contains the notebook.
- In the notebook import mlflow and run the following codes;
   - mlflow.set_tracking_uri("sqlite:///mlflow.db") - Connects to a tracking URI. URI provided should match the one provided in the command
   - mlflow.set_experiment("experiment-name") - Will set our experiment as active, if experiment doesn't exit, it will create a new one.
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
* Install Flask
* Install gunicorn
    ## Part A (Creating a docker file with conda environment)
    * Create file 'predict.py' for the flask application.
    * Get the model from mlflow registry
    * Load the count vectorizer from the local folder
    * Create file 'test.py' for testing
    * Put 'predict.py' into a flask app and test it with test.py
    * Run the application with gunicorn "gunicorn --bind=0.0.0.0:9696 predict:app"
    * Package app to a docker container
    * Create a dockerfile
    * Create .yml file that contain environment dependencies 'conda env export > nlp_project.yml'
    * Build docker image '  sudo /usr/bin/docker build -t fake-news-classification:v1 . '
    ## Part B
    * Create Model Deployment folder
    * copy the 'predict.py' file
    * modify so that both the count vectorizer and model are loaded from the local file
    * Run the application with gunicorn "gunicorn --bind=0.0.0.0:9696 predict:app"
    * Package app to a docker container
    * Get requirements.txt file
    * Create a dockerfile
    * Build docker image '  sudo /usr/bin/docker build -t fake-news-classification:v2 . '
    * Then run 'sudo /usr/bin/docker run -it --rm -p 9696:9696  fake-news-classification:v2'

* NOTE: Part B worked perfectly. Part A still has some issues
## Step 5 Model monitoring
* Install pymongo, evidently, pyarrow
* Prepare a prediction service
- Save to MongoDB and Evidently service
## Step 6 Best practices
* install pytest
* configure pytest
* create folder 'Best practices'
* Create 'batch.py' script
* Add model and count vectorizer to the folder
* create a test folder inside 'Best practices'
* Add '__init__.py' to test folder to let python know it's a python package
* create 'batch_test.py'
* Write unit test for normalizing text
* Write unit test for preparing features
* Linting
    - Install pylint
    - Run 'pylint batch.py' to check batch.py
    - Run 'pylint --recursive=y .' to check for all the files in the working directory
    - To view suggestion on VScode, select pylint as a linter
    - Configure pylint and disable some suggestions
    - create pyproject.toml file to disable some suggestions
* Formating
    - Install black and isort
    - Run 'black --diff .'
    - Run 'black .' to apply formating
    - Run 'pylint --recursive=y .' to check for the problems
    - use isort for sorting import problems
    - Run 'isort --diff .'
    - Run 'isort .'
* Git pre-commit hooks
    - Install pre-commit 'pip install pre-commit'
    - Run 'pre-commit install' to add pre-commit to the git hooks
    - Create a config file '.pre-commit-config.yaml'
* Makefiles and make
    - create a Makefile
    - Run 'make run'


## Quick solutions
- How to shut down mlflow ui
    - fuser -k 5000/tcp
- How turn jupyter notebook to python script
    - jupyter nbconvert --to script notebookname (make sure to install nbconvert)
- How to comment a block of code in python
    - crtl + / (On windows)
- How to check for prefect configuration
    - prefect config view
- How to unset prefect configuration
    - prefect config unset
- How to fix '400 Bad Request'
    - Install latest version of prefect 2.0
- How to check prefect storage
    - prefect storage ls
- How to handle the 'Docker no space left on the device error'
    - 'docker system prune --all --force --volumes' to delete unused volumes
    - 'docker volume ls' to get the list of volumes

