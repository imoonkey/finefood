__author__ = 'moonkey'

from vectorization import load_finefood
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import logging
import cPickle as pickle


class HelpfulnessPredictor(object):
    def __init__(self):
        # self.predictor = RandomForestRegressor(
        #     random_state=0, n_estimators=10, verbose=10)
        self.predictor = LinearRegression()

    def train(self, x, y):
        self.predictor.fit(x, y)

    def test(self, x, y):
        score = self.predictor.score(x, y)
        logging.info("Score: " + str(score))
        return score

    def predict(self, x):
        return self.predictor.predict(x)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.predictor, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.predictor = pickle.load(f)
        return self.predictor


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        # filename='predict.log'
    )

    train_x, train_y, test_x, test_y = load_finefood()
    logging.info('Data loaded.')

    h_pred = HelpfulnessPredictor()
    h_pred.train(train_x, train_y)
    logging.info('Model trained.')
    h_pred.save('pred')
    logging.info('Model saved.')

    h_pred.load('pred')
    logging.info('Model loaded.')
    h_pred.test(test_x, test_y)
    logging.info('Model trained.')