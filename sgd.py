import numpy as np
import pandas as pd
import re
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from string import digits

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier

import pickle
import joblib
from sklearn.feature_selection import SelectKBest, chi2


from sklearn import metrics
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plot

import seaborn as sns

sns.set(style="darkgrid")


def preprocess(text):
    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Allah', 'God', 'And' 'Lord', 'Allaah', 'He',
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

    return tokens


def main():
    # loading  Data

    category = ['Accepted', 'Rejected']
    doc_to_data = skd.load_files('Dataset/', description=None, categories=category, load_content=True,
                                 encoding='ISO-8859-1', random_state=24)


    # Splitting Data

    X_train, X_test, y_train, y_test = train_test_split(doc_to_data.data, doc_to_data.target, test_size=0.05,
                                                        random_state=24)

    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Allah', 'God', 'And' 'Lord', 'Allaah',
                         'He',
                         'She', 'A', 'They', '(h)', 'We', 'It']

    text_clf = Pipeline([
        ('vector',
         CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization,
                         min_df=2, max_df=0.5, stop_words=custom_stop_words)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier',
         SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))

    ])

    clf = text_clf.named_steps['classifier']
    text_clf = text_clf.fit(X_train, y_train)

    print(text_clf.score(X_test, y_test))

    predicted = text_clf.predict(X_test)

    joblib.dump(text_clf, 'sgd.pkl')
    print(metrics.accuracy_score(y_test, predicted))

    print(metrics.precision_score(y_test,predicted))
    print(metrics.recall_score(y_test,predicted))
    print(metrics.f1_score(y_test, predicted))

    print(metrics.classification_report(y_test, predicted, target_names=doc_to_data.target_names))

    # Testing a positive Case
    sen = ['''
        Tell us Koutaiba bin Said, told us Abu Awana, Sammaak ibn Harb, h and told us Hnad, told us Wakee, for Israel,
         for the fishmonger, for Musab bin Saad, son of Omar, the Prophet, peace be upon him, peace be upon him he said 
    ''']
    #print(text_clf.predict(sen))

    # Confusion Matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    #print(cm)

    '''
    param_grid = [
        {'vector__ngram_range': [(1, 1), (1, 2)],
         'vector__min_df': [1, 2, 3],
         'vector__max_df': [0.5, 0.7]}
    ]
    search = GridSearchCV(estimator=text_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=True)
    best = search.fit(X_train, y_train)
    print(best.best_params_)
    print(best.best_score_)

    # print(vector.get_feature_names())
    # print(vector.vocabulary_)

    
    # print(classifier.intercept_)

    # Create a zipped list of tuples from above lists
    zippedList = list(zip(X_test, y_test))
    df = pd.DataFrame(zippedList, columns=['Isnad', 'Class'])
    # print(df.head(5))

    sns.catplot(x="Class", data=df, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
    plot.show()
    
    # Matrix Plot
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index=np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plot.figure(figsize=(40, 35))
    sns.set(font_scale=2)
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})

    plot.show();
    '''



main()