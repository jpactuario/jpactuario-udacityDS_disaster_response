import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as pl_go
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Extract data needed for visuals

    # 1) Extract list of categories & sum up the number of messages by category
    df_summary = df[df.columns[4:]].sum().sort_values()
    categories_names = df_summary.index.tolist()
    message_count = df_summary.tolist()

    # 2) Extract the length of messages
    message_lengths = df.message.str.len().tolist()

    # capping the length at 99th percentile to truncate the tail,
    # making the histogram visualization easier to see.
    message_length_99p = int(np.percentile(message_lengths, 99))
    message_lengths_capped = np.minimum(message_lengths, message_length_99p)

    # create visuals

    graph_one = []
    trace_one = pl_go.Bar(x=message_count, y=categories_names,
                        orientation='h')
    graph_one.append(trace_one)
    layout_one = {
                  'title': 'Count of messages by Response Type',
                  'xaxis': dict(title = 'Message Count'),
                  'height': 1000
                 }

    graph_two = []
    trace_two = pl_go.Histogram(x=message_lengths_capped)
    graph_two.append(trace_two)
    layout_two = {
                  'title': 'Histogram of Message Length (capped at {})'.format(
                    message_length_99p
                  ),
                  'xaxis': dict(title = 'Message Length')
                 }

    # compiling all visuals

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
