from letor2 import rank
from compression import VBEPostings
from bsbi import BSBIIndex
from gensim.models import LsiModel
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


app = Flask(__name__)
CORS(app)


def get_search(query):
    bm25_letor_score_doc = []
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
        letor_result = rank(query, "./dataset/document/" +
                            doc, lsi_model, ranker_model, dictionary)
        bm25_letor_score_doc.append(letor_result)

    sorted_bm25_letor_score_doc = sorted(
        bm25_letor_score_doc, key=lambda x: x[0], reverse=True)
    return sorted_bm25_letor_score_doc


@app.route('/')
def get_results():
    query = request.args.get('q')
    if not query:
        return jsonify([])

    try:
        search_results = get_search(query)

        response_data = [{
            'title': result[2],
            'content': result[1]
        } for result in search_results
        ]

        return jsonify(response_data)
    except Exception as e:
        print(f"Error in get_results: {e}")
        return jsonify({'error': 'An error occurred while processing the request'})


# Custom 404 handler
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Page not found'}), 404


stemmer = PorterStemmer()
stop_words_set = set(stopwords.words('english'))

lsi_model = LsiModel.load("lsi.model")
dictionary = ""
with open("lsi.dict", 'rb') as f:
    dictionary = pickle.load(f)

ranker_model = ""
with open("model3.pkl", 'rb') as f:
    ranker_model = pickle.load(f)

BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

BSBI_instance.load()

if __name__ == '__main__':
    from waitress import serve

    serve(app, host='0.0.0.0', port=5000)
