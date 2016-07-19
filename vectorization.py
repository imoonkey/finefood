__author__ = 'moonkey'

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import cPickle as pickle
from scipy import sparse, io
import numpy as np
import logging


class Vectorization(object):
    def __init__(self):
        self.vectorizer = CountVectorizer(min_df=100)
        # min_df=3: token count=33068
        # min_df=10: token count=16777
        # min_df=20: token count=11733
        # min_df=50: token count=7479
        # min_df=100: token count=5298

    def vectorize(self, train_data, test_data):
        train_sents = [' '.join(t['tokens']) for t in train_data]
        test_sents = [' '.join(t['tokens']) for t in test_data]
        logging.info('Token to sentence complete.')

        self.vectorizer.fit(train_sents)
        logging.info('Vectorizer ready.')
        logging.info(
            'token count=' + str(len(self.vectorizer.get_feature_names())))
        # logging.info(
        #     'tokens=' + str(self.vectorizer.get_feature_names()))

        with open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)
        train_matrix = self.vectorizer.transform(train_sents)
        test_matrix = self.vectorizer.transform(test_sents)
        logging.info('Feature matrices got.')

        return train_matrix, test_matrix

    @staticmethod
    def save_sparse_csr(filename, array):
        io.mmwrite(filename, array)
        # np.savez(filename, data=array.data, indices=array.indices,
        # indptr=array.indptr, shape=array.shape)

    @staticmethod
    def load_sparse_csr(filename):
        return io.mmread(filename)
        # loader = np.load(filename)
        # return sparse.csr_matrix(
        # (loader['data'], loader['indices'], loader['indptr']),
        # shape=loader['shape'])


def vectorize_finefood(train_data, test_data):
    vec = Vectorization()
    train_x, test_x = vec.vectorize(train_data, test_data)
    vec.save_sparse_csr(filename='train_x.mtx', array=train_x)
    vec.save_sparse_csr(filename='test_x.mtx', array=test_x)

    train_y = np.array([t['helpfulness'] for t in train_data])
    test_y = np.array([t['helpfulness'] for t in train_data])
    np.save('train_y', train_y)
    np.save('test_y', test_y)
    logging.info('Fine food tokens vectorized.')
    return train_x, train_y, test_x, test_y


def load_finefood():

    train_x = Vectorization.load_sparse_csr('train_x.mtx').tocsr()
    train_y = np.load('train_y.npy')
    test_x = Vectorization.load_sparse_csr('test_x.mtx').tocsr()
    test_y = np.load('test_y.npy')

    logging.info('Fine food data loaded.')
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename='predict.log'
    )
    vectorize_finefood(train_data=pickle.load(open('train.pkl', 'rb')),
                       test_data=pickle.load(open('test.pkl', 'rb')))
