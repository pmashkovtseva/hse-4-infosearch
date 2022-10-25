from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()


def indexing_corpus(corpus):
    vectorized_corpus = vectorizer.fit_transform(corpus)
    return vectorized_corpus


def indexing_query(query):
    vectorized_query = vectorizer.transform(query)
    return vectorized_query
