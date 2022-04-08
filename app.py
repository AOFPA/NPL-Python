from flask import Flask,render_template,request,url_for

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

app = Flask(__name__)

#train nlp
thai_stopwords = list(thai_stopwords())
df = pd.read_csv('review_shopping.csv', sep='\t', names=['text', 'sentiment'], header=None)

def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                    if word.lower not in thai_stopwords)
    return final

df['text_tokens'] = df['text'].apply(text_process)

X = df[['text_tokens']]
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
cvec.fit_transform(X_train['text_tokens'])

train_bow = cvec.transform(X_train['text_tokens'])

lr = LogisticRegression()
lr.fit(train_bow, y_train)
test_bow = cvec.transform(X_test['text_tokens'])


#flask
@app.route('/')
def index():
    return render_template('index.html',predic="")

@app.route('/process', methods=['POST'])
def nlp():
    if request.method == 'POST':
        message = request.form['message']
        if message == '':
            return render_template('index.html')
        else:
            my_text = [message]
            my_tokens = text_process(my_text)
            my_bow = cvec.transform(pd.Series([my_tokens]))
            my_predictions = lr.predict(my_bow)

            return render_template('index.html', predic = my_predictions[0],msg = my_text[0])


if __name__ == '__main__':
    app.run(debug=True)