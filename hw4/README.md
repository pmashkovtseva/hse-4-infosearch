## hw4

`data_reading.py` отвечает за чтение *.jsonl*-файла и сбор корпусов вопросов и ответов.

`data_indexing.py` отвечает за индексацию корпуса вопросов и запроса к поисковику.

`search.py` отвечает за подсчёт близости между документами корпуса и запроса и формирование поисковой выдачи.

`main.py` отвечает за сбор необходимых параметров поиска и выполнение поиска в соответствии с ними. параметры:

1. `-d` (`--data`): путь к *.jsonl*-файлу с корпусом. обязателен.</li>
2. `-m` (`--model`): путь к используемой модели для индексации корпуса. по умолчанию - *sberbank-ai/sbert_large_nlu_ru*.</li>
3. `-q` (`--query`): поисковой запрос в виде строки. обязателен.</li>
4. `-n` (`--number_of_results`): топ-n результатов поисковой выдачи. по умолчанию равен размеру всего корпуса ответов.</li>
