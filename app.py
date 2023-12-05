from flask import Flask, request, jsonify
from flask_cors import CORS

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import joblib
import pickle

from gensim.models import LsiModel
from bsbi import BSBIIndex

from compression import VBEPostings

from letor2 import rank

app = Flask(__name__)
CORS(app)

def get_search(query):
        bm25_letor_score_doc = []
        for (score, doc) in BSBI_instance.retrieve_bm25(query, k=30):
                letor_result = rank(query, "./dataset/document/" + doc, lsi_model, ranker_model, dictionary)
                bm25_letor_score_doc.append(letor_result)

        sorted_bm25_letor_score_doc = sorted(bm25_letor_score_doc, key=lambda x: x[0], reverse=True)
        return sorted_bm25_letor_score_doc

@app.route('/')
def get_results():
    query = request.args.get('q')

    search_results = get_search(query)

    response_data = [
        {
            'title': result[2], 
            'content': result[1]  
        }
        for result in search_results
    ]

    return jsonify(response_data)

if __name__ == '__main__':
    stemmer = PorterStemmer()
    stop_words_set = set(stopwords.words('english'))

    lsi_model = LsiModel.load("lsi.model")
    dictionary = ""
    with open("lsi.dict", 'rb') as f:
            dictionary = pickle.load(f)
    ranker_model = joblib.load("model.pkl")

    BSBI_instance = BSBIIndex(data_dir='collections',
                                postings_encoding=VBEPostings,
                                output_dir='index')

    BSBI_instance.load()

    BSBI_instance = BSBIIndex(data_dir='collections',
                            postings_encoding=VBEPostings,
                            output_dir='index')

    BSBI_instance.load()

    app.run(debug=True)
