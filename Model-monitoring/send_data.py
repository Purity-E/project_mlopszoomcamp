import json
import uuid
from datetime import datetime
from time import sleep
import re
import pandas as pd
import requests
import contractions
import spacy
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pyarrow as pa

df = pd.read_csv("/home/ubuntu/my_mlops_code/project_mlopszoomcamp/data/news3.csv")
table = pa.Table.from_pandas(df) # converting pandas dataframe to pyarrow table
data = table.to_pylist()


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



class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", 'w') as f_target:
    for row in data:
        row['id'] = str(uuid.uuid4())
        clean_title = prepare_feature(row['title'])
        f_target.write(f"{row['id']},{clean_title}\n")
        resp = requests.post("http://127.0.0.1:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(row, cls=DateTimeEncoder)).json()
        print(f"prediction: {resp['clean_title']}")
        sleep(1)
