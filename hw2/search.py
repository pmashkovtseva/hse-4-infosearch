import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculating_similarity(texts, query):
    return cosine_similarity(texts, query)


def getting_results(scores, filenames, n):
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus_doc_names = np.array(filenames)
    return corpus_doc_names[sorted_scores_indx.ravel()][:n]
