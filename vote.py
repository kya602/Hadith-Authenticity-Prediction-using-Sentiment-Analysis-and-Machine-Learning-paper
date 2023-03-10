
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import re
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from string import digits
import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

def preprocess(text):


    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Aisha', 'Division', 'Allah', 'God', 'Lord', 'Allaah','He', 'She', 'A', 'They','(h)', 'We', 'It']
    '''
    for i in custom_stop_words:
        stop_words.append(i)
    '''

    text = text.translate(str.maketrans('', '', digits))    #removing digits

    text = ', '.join([word for word in text.split() if word not in custom_stop_words])  #removing stop words

    return text


def tokenization(text):

    pattern = re.compile(r"((ar-|al-)?[A-Z][A-Za-z-']+)\s?(bin\s|ibn\s|binte\s|bint\s)?((al-|ar-)?([A-Z][A-Za-z-']+\s?)|(bin\s|ibn\s|binte\s|bint\s))*")

    matches = pattern.finditer(text)

    tokens = []

    for match in matches:
        tokens.append(match.group())

    #print(tokens)
    '''
    names = list(map(" ".join, tokens))
    for i in names:
        i.strip()

    #print(names)
    '''
    return tokens

def main():
#loading  Data

    category = ['Rejected', 'Accepted']
    doc_to_data = skd.load_files('Dataset/', description = None, categories = category, load_content = True, encoding ='ISO-8859-1', random_state=24)


#Splitting Data

    X_train, X_test, y_train, y_test = train_test_split(doc_to_data.data, doc_to_data.target, test_size=0.1, random_state=24)


    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Aisha', 'Division', 'Allah', 'God', 'Lord',
                     'Allaah', 'He', 'She', 'A', 'They', '(h)', 'We', 'It']

    clf1 = Pipeline([
        ('vector', CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, stop_words=custom_stop_words, min_df=2, max_df=0.5)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier', LogisticRegression(penalty= 'l2', solver='liblinear',C=1, class_weight='balanced', random_state=24, tol=0.000001))

    ])
    clf2 = Pipeline([
        ('vector', CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, stop_words=custom_stop_words, min_df=2, max_df=0.5)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=24))

    ])
    clf3 = Pipeline([
        ('vector', CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, stop_words=custom_stop_words, min_df=2, max_df=0.5)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier', BernoulliNB())

    ])

    clf4 = Pipeline([
        ('vector', CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, stop_words=custom_stop_words, min_df=2, max_df=0.5)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier', KNeighborsClassifier(n_neighbors=3))

    ])


    X = X_train
    y = y_train

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('bnb', clf3), ('knn', clf4)],
                            voting='soft',
                            weights=[1, 1, 5, 3])

    # predict class probabilities for all classifiers
    probas = [c.fit(X, y).predict_proba(X_test) for c in (clf1, clf2, clf3,clf4, eclf)]

    # get class probabilities for the first sample in the dataset
    class1_1 = [pr[0, 0] for pr in probas]
    class2_1 = [pr[0, 1] for pr in probas]


    # plotting

    N = 5  # number of groups
    ind = np.arange(N)  # group positions
    width = 0.3  # bar width

    fig, ax = plt.subplots()

    # bars for classifier 1-3
    p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
                color='green', edgecolor='k')
    p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
                color='lightgreen', edgecolor='k')

    # bars for VotingClassifier
    p3 = ax.bar(ind, [0, 0, 0, 0, class1_1[-1]], width,
                color='blue', edgecolor='k')
    p4 = ax.bar(ind + width, [0, 0, 0, 0, class2_1[-1]], width,
                color='steelblue', edgecolor='k')

    # plot annotations
    #plt.axvline(2.8, color='k', linestyle='dashed')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(['LogisticRegression',
                        'BernoulliNB',
                        'RandomForestClassifier',
                        'KNeighborsClassifier',
                        'VotingClassifier\n(average probabilities)'],
                       rotation=40,
                       ha='right')
    plt.ylim([0, 1])
    plt.title('Class probabilities for sample 1 by different classifiers')
    plt.legend([p1[0], p2[0]], ['Accepted', 'Rejected'], loc='upper left')
    plt.tight_layout()
    plt.show();

main()