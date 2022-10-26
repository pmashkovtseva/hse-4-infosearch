import pickle
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

RESULTS_NUMBER = 10
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


def getting_data(path_to_questions, path_to_answers):
    with open(path_to_questions, 'rb') as q:
        questions = pickle.load(q)
    with open(path_to_answers, 'rb') as a:
        answers = pickle.load(a)
    return questions, answers


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def indexing_query(query):
    encoded_input = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return query_embeddings


def calculating_similarity(corpus_embeddings, query_embeddings):
    return cosine_similarity(corpus_embeddings, query_embeddings)


def getting_results(scores, corpus, n):
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus_doc_names = np.array(corpus)
    return list(corpus_doc_names[sorted_scores_indx.ravel()][:n])


def main(query):
    questions, answers = getting_data('./data/questions_embeddings.pickle', './data/answers.pickle')
    query_embeddings = indexing_query(query)
    scores = calculating_similarity(questions, query_embeddings)
    results = getting_results(scores, answers, RESULTS_NUMBER)
    return results
