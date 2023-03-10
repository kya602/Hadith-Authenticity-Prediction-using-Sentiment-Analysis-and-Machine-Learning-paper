import re
from string import digits
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

ex = ['''12 - Abu al-Yaman  the ruling bin Nafi said: Shoaib us al-Zuhri said: 
                        Obaidullah bin Abdullah bin Ataba bin Masood, told me that Abdullah bin Abbas told him,
                        that Abu Sufyan bin Harb told him: Saad ar-Raky''']


def preprocess(text):


    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Allah', 'God', 'Lord', 'Allaah','He', 'She', 'A', 'They','(h)', 'We', 'It']
    '''
    for i in custom_stop_words:
        stop_words.append(i)
    '''

    text = text.translate(str.maketrans('', '', digits))    #removing digits

    text = ' '.join([word for word in text.split() if word not in custom_stop_words])  #removing stop words

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

vector = CountVectorizer(encoding='ISO-8859-1', lowercase=False, preprocessor=preprocess, tokenizer=tokenization)

x = vector.fit(raw_documents=ex)
print(vector.vocabulary_)
    #print(xtrain_cv.shape)
    #print(xtrain_cv.toarray())