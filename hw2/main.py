import argparse
from pathlib import Path

from data_indexing import indexing_corpus, indexing_query
from data_processing import reading_data, preprocessing_text
from search import calculating_similarity, getting_results

parser = argparse.ArgumentParser()


def parsing_arguments():
    parser.add_argument("-q", "--query", type=str, required=True, help="search query as a string")
    parser.add_argument("-p", "--path", type=Path, required=False, help="path to text files", default="./friends-data")
    parser.add_argument("-n", "--number_of_results", type=int, required=False, default=10,
                        help="number of results sorted by "
                             "relevancy")
    args = parser.parse_args()
    return args


def main(args):
    texts, filenames = reading_data(args.path)
    preprocessed_texts = preprocessing_text(texts)
    preprocessed_query = preprocessing_text(args.query)
    vectorized_corpus = indexing_corpus(preprocessed_texts)
    vectorized_query = indexing_query(preprocessed_query)
    scores = calculating_similarity(vectorized_corpus, vectorized_query)
    results = getting_results(scores, filenames, args.number_of_results)
    print(results)


if __name__ == '__main__':
    arguments = parsing_arguments()
    main(arguments)
