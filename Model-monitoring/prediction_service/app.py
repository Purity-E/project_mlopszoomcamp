# importing libraries
from dataclasses import dataclass
import re
import spacy
import contractions
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
import mlflow
from nltk.stem import WordNetLemmatizer
import pickle
from flask import Flask, request, jsonify
import os
from pymongo import MongoClient
import json
import requests

# getting the mongodb address
MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://127.0.0.1:27017')
# getting the evidently address
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')

# getting  count vectorizer and model
CV_FILE = os.getenv('CV_FILE', 'CV.pkl')
MODEL_FILE = os.getenv('MODEL_FILE', 'model.pkl')


# loading count vectorizer and model
with open('CV_FILE', "rb") as f_in:
        cv = pickle.load(f_in)

with open('MODEL_FILE', "rb") as f_in:
        model = pickle.load(f_in)


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
def prepare_feature(x):
    # cleaning title
    x =  normalize_document(x) # normalize text
    x =  remove_stop(x) # remove stopwords
    x = lemmatize(x) # lemmatize text
    doc = []
    doc.append(x)
    return doc

# prediction 
def predict(doc):
    data = cv.transform(doc)
    y = model.predict(data)
    if y == 1:
        pred = 'FAKE'
    elif y == 0:
        pred = 'REAL'
    return pred


app = Flask('News Classification')

# saving to mongodb
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("news_classification")
collection = db.get_collection("data")


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/news", json=[rec])

# creating function for the web request
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    
    record = request.get_json()

    doc = prepare_feature(record['title'])
    prediction = predict(doc)

    result = {
        'News classification':prediction
    }

    save_to_db(record, prediction)
    send_to_evidently_service(record, prediction)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)