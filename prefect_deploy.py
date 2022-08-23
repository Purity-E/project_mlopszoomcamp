#!/usr/bin/env python
# coding: utf-8


# importing libraries
from webbrowser import get
import pandas as pd
import re
import numpy as np
import spacy
import contractions
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, roc_auc_score,  accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.pyll import scope
from prefect import flow, task
from nltk.stem import WordNetLemmatizer


# loading datasets
@task
def get_data(path1, path2, path3):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    data = pd.concat([df1, df2, df3])
    return data


# normalizing the documents
def normalize_document(doc):
    # remove special characters\whitespaces
    pattern = r'[^a-zA-Z0-9\s]'
    doc = re.sub(pattern, '', doc)
    doc = doc.strip()
    doc = contractions.fix(doc)
    doc = doc.lower()
    return doc


# removing stopwords 
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
def remove_stop(doc):
    #tokenize words
    word_tokens = word_tokenize(doc)
    #removing stopwords
    filtered_tokens = [token for token in word_tokens if token not in stopwords]
    return filtered_tokens


# function for converting tags
def pos_tag_wordnet(tagged_tokens):
    tag_map = {'j': wordnet.ADJ, 'v': wordnet.VERB, 'n': wordnet.NOUN, 'r': wordnet.ADV}
    new_tagged_tokens = [(word, tag_map.get(tag[0].lower(), wordnet.NOUN))
                            for word, tag in tagged_tokens]
    return new_tagged_tokens


# lematizing words
wnl = WordNetLemmatizer()
def lemmatize(doc):
    #POS tagging
    tagged_tokens = nltk.pos_tag(doc)
    # converting the tags
    wordnet_tokens = pos_tag_wordnet(tagged_tokens)
    #lemmatizing
    filtered_doc = ' '.join(wnl.lemmatize(word, tag) for word, tag in wordnet_tokens)
    return filtered_doc

# Data preprocessing
@task
def data_cleaning(df):
    # dropping duplicates
    df.drop_duplicates(subset ="title", keep = 'first', inplace = True)
    # cleaning title
    df['clean_title'] = df['title'].apply(lambda x: normalize_document(x)) # normalize text
    df['clean_title'] = df['clean_title'].apply(lambda x: remove_stop(x)) # remove stopwords
    df['clean_title'] = df['clean_title'].apply(lambda x: lemmatize(x)) # lemmatize text
    # turning the label to int
    df['label'] = (df.label == 'FAKE').astype(int)
    return df


#creating a function for encoding text data
def transform_text(data, df_train):
    #transforming with count vectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(df_train['clean_title'])
    cv = vectorizer.transform(data.clean_title)
    return cv

@task
def data_split_encode(df):
    # Splitting data into train and validation set
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=1)
    #creating the label
    y_train = df_train.label.values
    y_val = df_val.label.values
    print(df_train.shape), print(y_train.shape), print(df_val.shape), print(y_val.shape)

    # encoding data
    #transforming train data
    train_data = transform_text(df_train, df_train)
    #transforming validation data
    val_data = transform_text(df_val, df_train)

    return train_data, y_train, val_data, y_val


# Experiment tracking

# 1. Experiment run for linear regression

# experiment run for linear regression
@task
def model_lr(train_data, y_train, val_data, y_val):
    with mlflow.start_run():

        # setting tag for model
        mlflow.set_tag("model", "linear regression")

        # fitting the train data
        lr = LogisticRegression(max_iter=500, solver='lbfgs')
        lr.fit(train_data, y_train)

        # validating data
        y_pred = lr.predict(val_data)

        # evaluation
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val,y_pred)
        # logging metrics
        mlflow.log_metric("auc score", auc)
        mlflow.log_metric("f1 score", f1)
        mlflow.log_metric("accuracy", accuracy)

    return f1, auc, accuracy


# 2. Hyperparameter tuning for Random Forest and Decision Trees
@task
def model_trees(num_trials, train_data, y_train, val_data, y_val):
    def objective(params):
            classifier_type = params['type'] # defining model type
            del params['type'] # deleting model type from params dict
            with mlflow.start_run():
                if classifier_type == 'rf':
                    mlflow.set_tag("model", "random forest")
                    mlflow.log_params(params)
                    rf = RandomForestClassifier(**params)
                    rf.fit(train_data, y_train)
                    y_pred = rf.predict(val_data)
                elif classifier_type == 'dt':
                    mlflow.set_tag("model", "Decision trees")
                    mlflow.log_params(params)
                    dt = DecisionTreeClassifier(**params)
                    dt.fit(train_data, y_train)
                    y_pred = dt.predict(val_data)
                else:
                    y_pred = 0

                # evaluation
                f1 = f1_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred)
                accuracy = accuracy_score(y_val,y_pred)
                
                # logging metrics
                metrics = {"auc score": auc, "f1 score":f1, "accuracy":accuracy}
                mlflow.log_metrics(metrics)
            return {'loss': -f1, 'status': STATUS_OK}
    # defining search space
    search_space = hp.choice('classifier_type', [
        {
        'type': 'rf',
        'max_depth': scope.int(hp.quniform('max_depth', 100, 800, 100)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 10)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
        },
        {
        'type': 'dt',
        'max_depth': scope.int(hp.quniform('max_depth_dt', 100, 800, 100)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf_dt', 1, 4, 1)),
        'random_state': 42
        }])

    rstate = np.random.default_rng(42)  # for reproducible results
    # creating the fmin function
    best_result = fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=num_trials,
                trials=Trials(),
                rstate=rstate
                )
    print(space_eval(search_space, best_result))

    return best_result




# Retraining and Autologging with the best model and parameters

# def train_best_model(train_data, y_train):

#     best_params = {
#             'max_depth': 700,
#             'n_estimators': 40,
#             'min_samples_split': 9,
#             'min_samples_leaf': 1,
#             'random_state': 42
#             }

#     mlflow.sklearn.autolog()

#     rf = RandomForestClassifier(**best_params)
#     rf.fit(train_data, y_train)


# autologging linear regression

# def autolog_lr(train_data, y_train):
#     mlflow.sklearn.autolog()

#     lr = LogisticRegression(max_iter=500, solver='lbfgs')
#     lr.fit(train_data, y_train)


# creating main function
@flow
def main(path1='./data/news_a1.csv',path2='./data/news_a2.csv', path3='./data/news2.csv', num_trials=20):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("course_project")
    # getting data
    df = get_data(path1, path2, path3)
    # cleaning data
    df = data_cleaning(df)
    # split and encode data
    train_data, y_train, val_data, y_val = data_split_encode(df).result()
    # experiment tracking for linear regression
    lr_results = model_lr(train_data, y_train, val_data, y_val).result()
    print(f'f1, auc, accuracy; {lr_results}')
    # experiment tracking for Decision trees & random forest
    best_result = model_trees(num_trials, train_data, y_train, val_data, y_val)
    print(best_result)
    # training best model


# Deployment
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

 # define deploymentspec
DeploymentSpec(
  flow=main,
  name="model_training",
  schedule=IntervalSchedule(interval=timedelta(minutes=5)), 
  flow_runner=SubprocessFlowRunner(),
  tags=["ml"]
)



