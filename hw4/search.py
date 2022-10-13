import numpy as np


def calculating_similarity(corpus_embeddings, query_embeddings):
    return np.dot(corpus_embeddings.numpy(), np.transpose(query_embeddings.numpy()))


def getting_results(scores, corpus, n):
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus_doc_names = np.array(corpus)
    return corpus_doc_names[sorted_scores_indx.ravel()][:n]
