import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv('review_shopping.csv', sep='\t', names=['text', 'sentiment'], header=None)

#ดึง array ของ stopwords หรือคำที่ไม่ค่อยสื่อความหมาย
thai_stopwords = list(thai_stopwords())

#ทำการตัดคำ (Word Tokenize) ลบ stopword และ punctuation
def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                     if word.lower not in thai_stopwords)
    return final

df['text_tokens'] = df['text'].apply(text_process)


#Split ข้อมูลเป็น Train (70%) Test (30%)
X = df[['text_tokens']]
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Word Vectorizer และ Bag-of-Words (BoW)
cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
cvec.fit_transform(X_train['text_tokens'])

#เราจะใช้ BoW นี้ในการฝึกฝนแบบจำลอง 
train_bow = cvec.transform(X_train['text_tokens'])

#สร้างแบบจำลอง Logistic Regression เพื่อจำแนกความรู้สึก positive หรือ negative
lr = LogisticRegression()
lr.fit(train_bow, y_train)

#ใช้ sklearn ในการทดสอบแบบจำลองว่ามีความแม่นยำมากน้อยแค่ใหน
test_bow = cvec.transform(X_test['text_tokens'])
test_predictions = lr.predict(test_bow)
print(classification_report(test_predictions, y_test))

#ทดสอบกับข้อความที่เราสร้างขึ้นเอง
my_text = 'ตรงปกส่งไวครับ'
my_tokens = text_process(my_text)
my_bow = cvec.transform(pd.Series([my_tokens]))
my_predictions = lr.predict(my_bow)
print(my_predictions)