import argparse
from pathlib import Path

from data_indexing import indexing_documents, indexing_query
from data_reading import getting_data, CORPUS_SIZE
from search import calculating_similarity, getting_results

parser = argparse.ArgumentParser()


def parsing_arguments():
    parser.add_argument("-d", "--data", type=Path, required=True, help="path to a .jsonl document")
    parser.add_argument("-m", "--model", type=Path, required=False, help="path to a bert model", default="sberbank-ai"
                                                                                                         "/sbert_large_nlu_ru")
    parser.add_argument("-q", "--query", type=str, required=True, help="search query as a string")
    parser.add_argument("-n", "--number_of_results", type=int, required=False, default=CORPUS_SIZE,
                        help="number of results sorted by "
                             "relevancy")
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    answers = getting_data(arguments.data)
    corpus_embeddings = indexing_documents(answers, arguments.model)
    query_embeddings = indexing_query(arguments.query, arguments.model)
    scores = calculating_similarity(corpus_embeddings, query_embeddings)
    results = getting_results(scores, answers, arguments.number_of_results)
    print(results)


if __name__ == '__main__':
    arguments = parsing_arguments()
    main(arguments)
