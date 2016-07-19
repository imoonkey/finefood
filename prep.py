__author__ = 'moonkey'

import string
import nltk
from bs4 import BeautifulSoup as BS
import cPickle as pickle


class Prep(object):
    @staticmethod
    def read_data(filename='finefoods.txt'):
        """
        read data from file
        :param filename:
        :return: list of dicts
        """
        # productId, userId, username, helpfulness, score, time, summary, text
        res = []
        with open(filename, 'rb') as data_file:
            data = data_file.read().split('\n\n')
            data = [Prep._extract_useful_fields(Prep._row2dict(r))
                    for r in data if r]
            data = [d for d in data if d]
        return data

    @staticmethod
    def _row2dict(text):
        """
        turning each data record into a dict with keys:
        ['text', 'userId', 'summary', 'score', 'helpfulness',
        'time', 'profileName', 'productId']
        :param text:
        :return: dict of data
        """
        sd = [row.split(': ') for row in text.split('\n')]
        try:
            sd = {row[0].split('/')[1]: ' '.join(row[1:])
                  for row in sd if len(row) >= 2}
            # assert (len(sd) == 8)

        except Exception as e:
            print e
            for i in sd: print i
            print '\n\n'
        return sd

    @staticmethod
    def _extract_useful_fields(sd):
        """
        :param sd: dict from _row2dict()
        :return: only retain fields ['helpfulness', 'tokens']
        """
        result = {}
        h = [int(e) for e in sd['helpfulness'].split('/')]
        result['helpfulness'] = float(h[0]) / h[1] if h[1] > 0 else None
        # print sd['helpfulness'], result['helpfulness']
        if result['helpfulness'] is None:
            return None

        result['tokens'] = (Prep.tokenize(sd['summary']) or []) + \
                           (Prep.tokenize(sd['text']) or [])
        return result

    @staticmethod
    def tokenize(text, stemmer=nltk.stem.PorterStemmer()):
        """
            tokenize/ to lower case
        :param text: list of tokens
        :param stemmer:
        :return: returned type is the same as input text.
        """
        text = BS(text).text
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords += list(string.punctuation)

        if not text:
            return None
        if type(text) is str or type(text) is unicode:
            original_tokens = nltk.word_tokenize(text)
        elif type(text) is list:  # tokens
            original_tokens = text
        else:
            original_tokens = None

        stemmed_tokens = [
            stemmer.stem(t.lower()) for t in original_tokens
            if (t.lower() not in stopwords  # remove stop words
                and not all(c in string.punctuation for c in t)  # punctuations
                and not (t.startswith("'") and len(t) <= 2))
        ]

        return stemmed_tokens

    @staticmethod
    def split_train_test(data_list, train_ratio=0.8):
        train_num = int(len(data_list) * train_ratio)
        return data_list[:train_num], data_list[train_num:]

if __name__ == '__main__':
    food_data = Prep.read_data('finefoods.txt')
    print len(food_data)
    with open('finefoods.pkl', 'wb') as tokenfile:
        pickle.dump(food_data, tokenfile)

    train, test = Prep.split_train_test(food_data, train_ratio=0.8)
    with open('train.pkl', 'wb') as train_file:
        pickle.dump(train, train_file)
    with open('test.pkl', 'wb') as test_file:
        pickle.dump(test, test_file)
