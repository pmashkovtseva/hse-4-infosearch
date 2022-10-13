import torch
from transformers import AutoTokenizer, AutoModel

from data_reading import CORPUS_SIZE


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def indexing_documents(corpus, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    encoded_input = tokenizer(corpus[:CORPUS_SIZE], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return corpus_embeddings


def indexing_query(query, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    encoded_input = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return query_embeddings
