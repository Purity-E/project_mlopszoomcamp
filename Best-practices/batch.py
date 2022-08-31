# importing libraries
from dataclasses import dataclass
import re
import spacy
import contractions
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pickle
import sys


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
def predict(doc, cv, model):
    data = cv.transform(doc)
    y = model.predict(data)
    if y == 1:
        pred = 'FAKE'
    elif y == 0:
        pred = 'REAL'
    return pred


# creating function for the web request
def main(sent):
    news = {}
    news['title'] = f'{sent}'

    # loading count vectorizer and model
    with open('CV.pkl', "rb") as f_in:
            cv = pickle.load(f_in)

    with open('model.pkl', "rb") as f_in:
            model = pickle.load(f_in)

    # preparing features
    doc = prepare_feature(news['title'])

    # making predictions
    predictions = predict(doc, cv, model)

    return predictions


if __name__ == "__main__":
    sent = sys.argv[1:]
    pred = main(sent)
    print(f'News classification: {pred}')


