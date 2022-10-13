import json

CORPUS_SIZE = 10


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
