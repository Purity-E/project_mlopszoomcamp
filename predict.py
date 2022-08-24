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




mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("course_project")

# loading the model from mlflow
model_name = "FakeNews_Classifier"
stage = "Production"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

# loading count vectorizer
with open('models/CV.pkl', "rb") as f_in:
        cv = pickle.load(f_in)


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


# creating function for the web request
@app.route('/predict', methods=['POST'])
def predict_endpoint():

    news = request.get_json()

    doc = prepare_feature(news['title'])
    predictions = predict(doc)

    result = {
        'News classification':predictions
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


