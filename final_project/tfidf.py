import pickle
import numpy as np
import pymorphy2
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_NUMBER = 10
vectorizer = TfidfVectorizer()
tokenizer = RegexpTokenizer(r'[А-Яа-яё]+')
stops = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()


def getting_data(path_to_questions, path_to_answers):
    with open(path_to_questions, 'rb') as q:
        questions = pickle.load(q)
    with open(path_to_answers, 'rb') as a:
        answers = pickle.load(a)
    return questions, answers


def preprocessing_data(data):
    preprocessed_data = []
    if isinstance(data, str):
        data = [data]
    for i in range(len(data)):
        tokens = tokenizer.tokenize(data[i])
        preprocessed_data.append(
            ' '.join([morph.parse(token.lower())[0].normal_form for token in tokens if token.lower() not in stops]))
    return preprocessed_data


def indexing_corpus(corpus):
    vectorized_corpus = vectorizer.fit_transform(corpus)
    return vectorized_corpus


def indexing_query(query):
    vectorized_query = vectorizer.transform(query)
    return vectorized_query


def calculating_similarity(corpus, query):
    return cosine_similarity(corpus, query)


def getting_results(scores, indexes, n):
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    if np.argmax(scores) == 0:
        return []
    else:
        corpus_doc_names = np.array(indexes)
        return list(corpus_doc_names[sorted_scores_indx.ravel()][:n])


def main(query):
    questions, answers = getting_data('./data/questions_preprocessed.pickle', './data/answers.pickle')
    preprocessed_query = preprocessing_data(query)
    vectorized_corpus = indexing_corpus(questions)
    vectorized_query = indexing_query(preprocessed_query)
    scores = calculating_similarity(vectorized_corpus, vectorized_query)
    results = getting_results(scores, answers, RESULTS_NUMBER)
    return results
