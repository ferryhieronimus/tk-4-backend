"""
    Kode diadaptasi dari 
    https://colab.research.google.com/drive/1zsmcwN5fNBrVQzvE1YPEn8gJQIHXL8wa?usp=sharing#scrollTo=vKAwu1KvxqRF
    dengan beberapa modifikasi dan penambahan
"""
import os
import random
import re
import lightgbm as lgb
import numpy as np
from collections import defaultdict
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

import joblib
import pickle

# untuk stemming kalimat
stemmer = PorterStemmer()
stop_words_set = set(stopwords.words('english'))

# function untuk mendapatkan isi dari dokumen berdasarkan relative pathnya
def open_documents(file_path):
    all_words = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            words = word_tokenize(line)
            stemmed_words = [stemmer.stem(word) for word in words]
            stemmed_text = " ".join(stemmed_words)
            terms = re.findall(r'\w+', stemmed_text)
            terms_without_stopwords = [
                term for term in terms if term not in stop_words_set]
            all_words.extend(terms_without_stopwords)
    return all_words

# function untuk membuat representasi vektor dari query atau dokumen
def vector_rep(model, dictionary, text):
    rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
    return rep if len(rep) == model.num_topics else [0.] * model.num_topics


# representasi vector dari gabungan query + document + cosine distance + jaccard similarity
def features(query, doc, model, dictionary):
    v_q = vector_rep(model, dictionary, query)
    v_d = vector_rep(model, dictionary, doc)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)
    return v_q + v_d + [jaccard] + [cosine_dist]

# untuk mendapatkan nilai rank (score) dari query
def rank(query, documents_path, lsi_model, ranker_model, dictionary):
    doc = open_documents(documents_path)
    doc = []
    with open(documents_path, encoding="utf-8") as file:
        content = file.read()
        for line in content:
            words = word_tokenize(line)
            stemmed_words = [stemmer.stem(word) for word in words]
            stemmed_text = " ".join(stemmed_words)
            terms = re.findall(r'\w+', stemmed_text)
            terms_without_stopwords = [
                term for term in terms if term not in stop_words_set]
            doc.extend(terms_without_stopwords)
    filename = os.path.basename(documents_path)
    title_path = f"./dataset/title/title{filename}"
    with open(title_path,  encoding="utf-8") as file:
        title = file.read()
    X_unseen = []
    X_unseen.append(features(query.split(), doc, lsi_model, dictionary))
    result = ranker_model.predict(X_unseen)
    return (result, content, title)