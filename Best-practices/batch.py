# importing libraries
import pickle
import re
import sys

import contractions
import nltk
import spacy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# normalizing the documents
def normalize_document(doc):
    # remove special characters\whitespaces
    pattern = r"[^a-zA-Z0-9\s]"
    doc = re.sub(pattern, "", doc)
    doc = doc.strip()
    doc = contractions.fix(doc)
    doc = doc.lower()
    return doc


# removing stopwords
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words


def remove_stop(doc):
    # tokenize words
    word_tokens = word_tokenize(doc)
    # removing stopwords
    filtered_tokens = [token for token in word_tokens if token not in stopwords]
    return filtered_tokens


# function for converting tags
def pos_tag_wordnet(tagged_tokens):
    tag_map = {"j": wordnet.ADJ, "v": wordnet.VERB, "n": wordnet.NOUN, "r": wordnet.ADV}
    new_tagged_tokens = [
        (word, tag_map.get(tag[0].lower(), wordnet.NOUN)) for word, tag in tagged_tokens
    ]
    return new_tagged_tokens


# lematizing words
wnl = WordNetLemmatizer()


def lemmatize(doc):
    # POS tagging
    tagged_tokens = nltk.pos_tag(doc)
    # converting the tags
    wordnet_tokens = pos_tag_wordnet(tagged_tokens)
    # lemmatizing
    filtered_doc = " ".join(wnl.lemmatize(word, tag) for word, tag in wordnet_tokens)
    return filtered_doc


# Data preprocessing
def prepare_feature(text):
    # cleaning title
    text = normalize_document(text)  # normalize text
    text = remove_stop(text)  # remove stopwords
    text = lemmatize(text)  # lemmatize text
    doc = []
    doc.append(text)
    return doc


# prediction
def predict(doc, cv, model):
    data = cv.transform(doc)
    y = model.predict(data)
    if y == 1:
        pred = "FAKE"
    elif y == 0:
        pred = "REAL"
    return pred


# creating function for the web request
def main(sent):
    news = {}
    news["title"] = f"{sent}"

    # loading count vectorizer and model
    with open("CV.pkl", "rb") as f_in:
        cv = pickle.load(f_in)

    with open("model.pkl", "rb") as f_in:
        model = pickle.load(f_in)

    # preparing features
    doc = prepare_feature(news["title"])

    # making predictions
    predictions = predict(doc, cv, model)

    return predictions


if __name__ == "__main__":
    sentence = sys.argv[1:]
    prediction = main(sentence)
    print(f"News classification: {prediction}")
