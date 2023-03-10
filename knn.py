import numpy as np
import pandas as pd
import re
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from string import digits

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import model_selection, naive_bayes
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plot

import seaborn as sns

sns.set(style="darkgrid")


def preprocess(text):
    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Allah', 'God', 'Lord', 'Allaah', 'He',
                         'She', 'A', 'They', '(h)', 'We', 'It']
    '''
    for i in custom_stop_words:
        stop_words.append(i)
    '''

    text = text.translate(str.maketrans('', '', digits))  # removing digits

    text = ', '.join([word for word in text.split() if word not in custom_stop_words])  # removing stop words

    return text


def tokenization(text):
    pattern = re.compile(
        r"((ar-|al-)?[A-Z][A-Za-z-']+)\s?(bin\s|ibn\s|binte\s|bint\s)?((al-|ar-)?([A-Z][A-Za-z-']+\s?)|(bin\s|ibn\s|binte\s|bint\s))*")

    matches = pattern.finditer(text)

    tokens = []

    for match in matches:
        tokens.append(match.group())

    # print(tokens)
    '''
    names = list(map(" ".join, tokens))
    for i in names:
        i.strip()

    #print(names)
    '''
    return tokens


def main():
    # loading  Data

    category = ['Accepted', 'Rejected']
    doc_to_data = skd.load_files('Dataset/', description=None, categories=category, load_content=True,
                                 encoding='ISO-8859-1', random_state=24)

    # print(doc_to_data.data)
    # print(doc_to_data.target)

    # Splitting Data

    X_train, X_test, y_train, y_test = train_test_split(doc_to_data.data, doc_to_data.target, test_size=0.05,
                                                        random_state=24)

    # vector = CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, stop_words='english')

    # xtrain_cv = vector.fit_transform(raw_documents=X_train)
    # print(vector.get_feature_names())
    # print(xtrain_cv.shape)
    # print(xtrain_cv.toarray())
    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'And', 'Aisha', 'Division', 'Allah',
                         'God',
                         'Lord', 'Allaah', 'He', 'She', 'A', 'They', '(h)', 'We', 'It']

    text_clf = Pipeline([
        ('vector',
         CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization,
                         min_df=2, max_df=0.5, stop_words=custom_stop_words)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier', KNeighborsClassifier(n_neighbors=3))

    ])

    text_clf = text_clf.fit(X_train, y_train)
    print(text_clf.score(X_test, y_test))
    predicted = text_clf.predict(X_test)

    print(metrics.accuracy_score(y_test, predicted))
    print(metrics.precision_score(y_test, predicted))
    print(metrics.recall_score(y_test, predicted))
    print(metrics.f1_score(y_test, predicted))
    print(metrics.classification_report(y_test, predicted, target_names=doc_to_data.target_names))

    # Testing a positive Case
    sen = ['''
        Tell us Hnad, told us Abu Sid, Aloamc, from Abu Wael, Amr ibn al-Harith bin Mustaliq, the son of my brother,
        Zainab woman Abdullah Zainab, a woman Abdullah bin Masood said, The Messenger of Allah, peace be upon him said
    ''']
    '''
    print(text_clf.predict(sen))
    print(text_clf.predict_proba(sen))

    # Confusion Matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    print(cm)

    # Scatter
    x_ax = list(range(0, len(X_test), 1))
    y_ = list(text_clf.predict_proba(X_test)[:, 1])

    # Histogram
    plot.hist(y_)
    plot.show();

    y_id = list(np.argsort(y_))
    y_ax = []
    for i in y_id:
        y_ax.append(y_[i])

    plot.axhline(.5, color='.5')
    plot.ylim(-.1, 1.1)
    plot.scatter(x_ax, y_ax, color="blue")
    plot.show();
    '''

main()