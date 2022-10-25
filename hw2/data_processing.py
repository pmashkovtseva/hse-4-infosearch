import os

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy2

tokenizer = RegexpTokenizer(r'[А-Яа-яё]+')
stops = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()
SPLIT = ' - '
NAME_INDEX = 2
STRIP = '.ru.txt'
ENCODING = 'utf-8'


def reading_data(path):
    texts = []
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filenames.append(file.split(SPLIT)[NAME_INDEX].strip(STRIP))
            with open(os.path.join(root, file), 'r', encoding=ENCODING) as f:
                text = f.read()
                texts.append(text)
    return texts, filenames


def preprocessing_text(text):
    preprocessed_text = []
    if isinstance(text, str):
        text = [text]
    for i in range(len(text)):
        tokens = tokenizer.tokenize(text[i])
        preprocessed_text.append(
            ' '.join([morph.parse(token.lower())[0].normal_form for token in tokens if token.lower() not in stops]))
    return preprocessed_text
