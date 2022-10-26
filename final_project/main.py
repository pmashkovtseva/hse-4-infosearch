import importlib
import time

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/search', methods=['get'])
def searching():
    start_time = time.time()
    if not request.args:
        return redirect(url_for('main'))
    query = request.args.get('query')
    metrics = request.args.get('metrics')
    results = importlib.import_module(metrics).main(query)
    if len(results) == 0:
        return render_template('not_found.html', query=query)
    else:
        exectution_time = round((time.time() - start_time), 3)
        return render_template('search.html', query=query, results=results, time=exectution_time)


@app.route('/faq')
def faq():
    return render_template('faq.html')


if __name__ == '__main__':
    app.run()
