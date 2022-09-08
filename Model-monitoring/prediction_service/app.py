# importing libraries
from dataclasses import dataclass
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


# loading count vectorizer and model
with open('CV.pkl', "rb") as f_in:
        cv = pickle.load(f_in)

with open('model.pkl', "rb") as f_in:
        model = pickle.load(f_in)



app = Flask('News Classification')

# saving to mongodb
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("news_classification")
collection = db.get_collection("data")


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = list(prediction)
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = list(prediction)
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/news", json=[rec])

# creating function for the web request
@app.route('/predict', methods=['POST'])
def predict_endpoint():

    record = request.get_json()

    data = cv.transform(record)
    prediction = model.predict(data)

    result = {
        'News classification': list(prediction)
    }

    save_to_db(record, prediction)
    send_to_evidently_service(record, prediction)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)