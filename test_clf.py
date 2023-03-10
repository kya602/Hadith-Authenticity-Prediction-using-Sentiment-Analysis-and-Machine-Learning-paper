import joblib
import re
from string import digits

def preprocess(text):
    custom_stop_words = ['Tell', 'Tell us', 'Narrated', 'Messenger', 'Prophet', 'Allah', 'God', 'Lord', 'Allaah', 'He',
                         'She', 'A', 'They', '(h)', 'We', 'It']

    text = ' '.join([word for word in text.split() if word not in custom_stop_words])  # removing stop words
    text = text.translate(str.maketrans('', '', digits))  # removing digits

    res = re.search(r'[-."}{?]',text)
    separator = res.group()
    head, sep, tail = text.partition(separator)
    #print(head)


    return head


def tokenization(text):
    pattern = re.compile(
        r"((ar-|al-)?[A-Z][A-Za-z-']+)\s?(bin\s|ibn\s|binte\s|bint\s)?((al-|ar-)?([A-Z][A-Za-z-']+\s?)|(bin\s|ibn\s|binte\s|bint\s))*")

    matches = pattern.finditer(text)

    tokens = []

    for match in matches:
        tokens.append(match.group())

    return tokens

#clf = joblib.load('sgd.pkl')
s = "Tell us Abu Bakr ibn Abi Shaybah, told us Ismail Ibn attic, for Jerira, Qais bin Abaya, told me son of Abdullah bin Mutt, from his father, he said, and rarely seen more by men in Islam, an event of it Vsmni I read {In the name of God the Merciful} And he said, What are the sons of thee and the event? Ravenous say If I read, say {Praise be to Allah, the Lord of the Worlds}."
#print(clf.predict(s))
print(preprocess(s))