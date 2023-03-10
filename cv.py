from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import re
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from string import digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, naive_bayes

from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score


import matplotlib.pyplot as plot

import seaborn as sns

sns.set(style="darkgrid")


def preprocess(text):


    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Aisha', 'Division', 'Allah', 'God', 'Lord', 'Allaah','He', 'She', 'A', 'They','(h)', 'We', 'It']


    text = text.translate(str.maketrans('', '', digits))    #removing digits

    text = ', '.join([word for word in text.split() if word not in custom_stop_words])  #removing stop words

    return text


def tokenization(text):

    pattern = re.compile(r"((ar-|al-)?[A-Z][A-Za-z-']+)\s?(bin\s|ibn\s|binte\s|bint\s)?((al-|ar-)?([A-Z][A-Za-z-']+\s?)|(bin\s|ibn\s|binte\s|bint\s))*")

    matches = pattern.finditer(text)

    tokens = []

    for match in matches:
        tokens.append(match.group())

    return tokens



def main():
    # loading  Data

    category = ['Accepted', 'Rejected']
    doc_to_data = skd.load_files('Dataset/', description=None, categories=category, load_content=True,
                                 encoding='ISO-8859-1', random_state=24)

    # print(doc_to_data.data)
    # print(doc_to_data.target)
    X_train, X_test, y_train, y_test = train_test_split(doc_to_data.data, doc_to_data.target, test_size=0.05,
                                                        random_state=24)
    # Splitting Data
    zippedList = list(zip(X_train,y_train))
    df = pd.DataFrame(zippedList, columns=['Isnad', 'Class'])

    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Aisha', 'Division', 'Allah', 'God',
                         'Lord', 'Allaah', 'He', 'She', 'A', 'They', '(h)', 'We', 'It']

    vector = TfidfVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, min_df=2,max_df=0.5,stop_words=custom_stop_words,
                             sublinear_tf=True, use_idf=True, smooth_idf=True)

    #counts = vector.fit(X_train)
    # print(vector.get_feature_names())

    #transformer = TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)

    features = vector.fit_transform(X_train).toarray()
    labels = df.Class


    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        #SVC(C=1.0, kernel='linear', degree=3, gamma='auto'),
        naive_bayes.BernoulliNB(),
        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None),
        LogisticRegression(penalty= 'l2', solver='liblinear',C=1, class_weight='balanced', random_state=24, tol=0.000001),
        KNeighborsClassifier(n_neighbors=3)

    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    print(cv_df.groupby('model_name').accuracy.mean())
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plot.show();





main()