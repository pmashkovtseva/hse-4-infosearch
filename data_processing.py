import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

CORPUS_SIZE = 10000


def getting_data(path):
    with open(path, 'r') as f:
        raw_data = list(f)[:CORPUS_SIZE]
    questions = []
    answers = []
    for i in range(len(raw_data)):
        questions.append(json.loads(raw_data[i])['question'])
        for j in range(len(json.loads(raw_data[i])['answers'])):
            answers.append(json.loads(raw_data[i])['answers'][j]['text'])
    return answers


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def indexing_documents(model, corpus):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    encoded_input = tokenizer(corpus[:CORPUS_SIZE], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return corpus_embeddings


def indexing_query(model, query):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    encoded_input = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return query_embeddings


def calculating_similarity(corpus_embeddings, query_embeddings):
    return np.dot(corpus_embeddings.numpy(), np.transpose(query_embeddings.numpy()))


def getting_results(scores, corpus, n):
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus_doc_names = np.array(corpus)
    return corpus_doc_names[sorted_scores_indx.ravel()][:n]
