import numpy as np
import pandas as pd
import re
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from string import digits

from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2


from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plot

import seaborn as sns

sns.set(style="darkgrid")


def preprocess(text):
    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Allah', 'God', 'Lord', 'Allaah', 'He',
                         'She', 'A', 'They', '(h)', 'We', 'It']


    text = text.translate(str.maketrans('', '', digits))  # removing digits

    text = ' '.join([word for word in text.split() if word not in custom_stop_words])  # removing stop words

    return text


def tokenization(text):
    pattern = re.compile(
        r"((ar-|al-)?[A-Z][A-Za-z-']+)\s?(bin\s|ibn\s|binte\s|bint\s)?((al-|ar-)?([A-Z][A-Za-z-']+\s?)|(bin\s|ibn\s|binte\s|bint\s))*")

    matches = pattern.finditer(text)

    tokens = []

    for match in matches:
        tokens.append(match.group())

    # print(tokens)

    return tokens


def main():
    # loading  Data

    category = ['Accepted', 'Rejected']
    doc_to_data = skd.load_files('Dataset/', description=None, categories=category, load_content=True,
                                 encoding='ISO-8859-1', random_state=24)


    # Splitting Data

    X_train, X_test, y_train, y_test = train_test_split(doc_to_data.data, doc_to_data.target, test_size=0.05,
                                                        random_state=24)
    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'And', 'Aisha', 'Division', 'Allah',
                         'God',
                         'Lord', 'Allaah', 'He', 'She', 'A', 'They', '(h)', 'We', 'It']

    text_clf = Pipeline([
        ('vector',
         CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization,
                         min_df=2, max_df=0.5, stop_words=custom_stop_words)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier', LinearSVC())

    ])

    vectorizer = text_clf.named_steps['vector']
    transformer = text_clf.named_steps['transformer']
    classifier = text_clf.named_steps['classifier']

    text_clf = text_clf.fit(X_train, y_train)
    #print(text_clf.score(X_test, y_test))
    predicted = text_clf.predict(X_test)
    joblib.dump(text_clf, 'lin-svc.pkl')
    #print(metrics.accuracy_score(y_test, predicted))
    print(metrics.precision_score(y_test, predicted))
    print(metrics.recall_score(y_test, predicted))
    #print(np.mean(predicted == y_test))
    print(metrics.f1_score(y_test,predicted))
    print(metrics.classification_report(y_test, predicted, target_names=doc_to_data.target_names))

    # Testing a positive Case
    sen = ['''
        Tell us Mohammed Rafi, told us Abdul Razak bin Hammam, told us Muammar bin Rashid bin Hammam alarm, 
        my brother gave Ben alarm said this is what told us Abu Huraira from Muhammad, the Messenger of Allah, peace be upon him
    ''']
    #print(text_clf.predict(sen))

    # Confusion Matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    #print(cm)



main()