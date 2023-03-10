from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import random
import matplotlib.pyplot as plot
import re
from itertools import chain
import pandas as pd
import numpy as np
import os
import sys
import string
from string import digits
from nltk.tokenize import word_tokenize

#df = pd.read_excel("Dataset.xlsx", "Sheet1")
#df.head(2)

def preprocess(text):

    #stop_words = list(stopwords.words("english"))

    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Allah', 'Allaah']
    '''
    for i in custom_stop_words:
        stop_words.append(i)
    '''
    #text = text.translate(str.maketrans('','', string.punctuation))
    text = text.translate(str.maketrans('', '', digits))

    text = ' '.join([word for word in text.split() if word not in custom_stop_words])

    return text

def tokenization(text):

    y = re.split(r':|,',text)
    names = []
    for i in y:
        x = re.findall(r"(?P<name>[A-Z][A-Za-z'-]+)+(?:(?P<surname>bin|binte|ibn|bint)|\s|(?P=name))*", i)
        print(x)
    #matched = re.findall(pattern, text, flags=0)

    #tokens = [list(elem) for elem in matched]

    '''
    for t in tokens:
        for i in t:
            while '' in i:
                i.remove('')
    '''
    #names = list(map(" ".join, tokens))
    #list(chain.from_iterable(tokens))


    return y

sentence = "3 - Abdullah ibn Sa'd Ar-Rocky, Bakr ibn Muhammad"
print(tokenization(preprocess(sentence)))
