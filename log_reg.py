import numpy as np
import pandas as pd
import re
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from string import digits
import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plot
import mglearn
import seaborn as sns
sns.set(style='darkgrid')
from sklearn import linear_model
from scipy.special import expit


def preprocess(text):


    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Aisha', 'Division',
                         'Allah', 'God', 'And', 'Lord', 'Allaah','He', 'She', 'A', 'They','(h)', 'We', 'It']


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
#loading  Data

    category = ['Rejected', 'Accepted']
    doc_to_data = skd.load_files('Dataset/', description = None, categories = category, load_content = True, encoding ='ISO-8859-1', random_state=24)

    #print(doc_to_data.data)
    #print(doc_to_data.target)

#Splitting Data

    X_train, X_test, y_train, y_test = train_test_split(doc_to_data.data, doc_to_data.target, test_size=0.05, random_state=24)

    custom_stop_words = ['Tell', 'Tell us','Narrated', 'Messenger', 'Prophet', 'Aisha', 'Division',
                         'Allah', 'God', 'Lord', 'Allaah','He', 'She', 'A', 'They','(h)', 'We', 'It','Cu']
    #vector = CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, stop_words=custom_stop_words, min_df=2, max_df=0.5)

    #xtrain_cv = vector.fit(raw_documents=X_train)
    #print(vector.get_feature_names())
    #print(xtrain_cv.shape)
    #print(len(xtrain_cv.vocabulary_))
    #pri#nt(xtrain_cv.toarray())

    #custom_stop_words = ['Tell', 'Tell us','Narrated', 'Messenger', 'Prophet', 'Aisha', 'Division', 'Allah', 'God', 'Lord', 'Allaah','He', 'She', 'A', 'They','(h)', 'We', 'It','Cu']

    text_clf = Pipeline([
        ('vector', CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization, stop_words=custom_stop_words, min_df=2, max_df=0.5)),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True)),
        ('classifier', LogisticRegression(penalty= 'l2', solver='liblinear',C=1, class_weight='balanced', random_state=24, tol=0.000001))

    ])
    
    text_clf.fit(X_train,y_train)
    vectorizer = text_clf.named_steps['vector']
    transformer = text_clf.named_steps['transformer']
    classifier = text_clf.named_steps['classifier']


    predicted = text_clf.predict(X_test)
    #skplt.metrics.plot_confusion_matrix(y_test, predicted, text_fontsize=25, figsize=(25,25))
    #plt.show();

    
    print(text_clf.score(X_test,y_test))
    predicted = text_clf.predict(X_test)
    joblib.dump(text_clf,'log-reg.pkl')


    #x_cv = vectorizer.fit_transform(raw_documents = X_train)
    #xtf = transformer.fit_transform(x_cv)
    #chi = SelectKBest(chi2,k=200).fit(x)

    feature_names = vectorizer.get_feature_names()
    #feat = [feature_names[i] for i in chi.get_support(indices=True)]
    #feat = np.asarray(feat)
    
    #for i, label in enumerate(doc_to_data.target_names):
        #top10 = np.argsort(classifier.coef_[i])[-10:]
        #print("%s : %s" % (label, ", ".join(feat[top10])))

    #plot_coefficients(classifier,feature_names)
    #mglearn.tools.visualize_coefficients(classifier.coef_,feature_names, n_top_features=15)
    #plot.show();
    print(metrics.accuracy_score(y_test, predicted))
    print(metrics.precision_score(y_test,predicted))
    print(metrics.recall_score(y_test,predicted))
    print(metrics.f1_score(y_test,predicted))


    #print(np.mean(predicted == y_test))
    print(metrics.classification_report(y_test, predicted, target_names=doc_to_data.target_names))

# Testing a negative Case
    text = ["Tell us Mohammed Rafi, told us Abdul Razak bin Hammam, told us Muammar bin Rashid bin Hammam alarm, my brother gave Ben alarm said this is what told us Abu Huraira from Muhammad, the Messenger of Allah, peace be upon him"]

    print(text_clf.predict_proba(text))
#Confusion Matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    #print(cm)

    #print(len(vectorizer.get_feature_names()))


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

    vectorizer = text_clf.named_steps['vector']
    transformer = text_clf.named_steps['transformer']
    classifier = text_clf.named_steps['classifier']

    
    # Create a zipped list of tuples from above lists
    zippedList = list(zip(X_train,y_train))
    df = pd.DataFrame(zippedList, columns=['Isnad', 'Class'])
    #print(df.head(5))
    
    sns.catplot(x="Class", data=df, kind="count", height=6, aspect=1.5)
    plot.show();


    zipped = list(zip(X_test, y_test))
    df2 = pd.DataFrame(zipped, columns=['Isnad', 'Class'])
    # print(df.head(5))

    sns.catplot(x="Class", data=df2, kind="count", height=6, aspect=1.5)
    plot.show();



main()