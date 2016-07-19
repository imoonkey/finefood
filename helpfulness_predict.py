__author__ = 'moonkey'

from vectorization import load_finefood

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import confusion_matrix, classification_report

import logging
import cPickle as pickle


class HelpfulnessPredictor(object):
    def __init__(self, predictor_model=LinearRegression()):
        self.predictor = predictor_model

    def train(self, x, y):
        self.predictor.fit(x, y)

    def test(self, x, y):
        pred_y = self.predict(x)

        # Classification analysis
        y_bin = [i > 0.5 for i in y]
        pred_y_bin = [i > 0.5 for i in pred_y]

        conf_mtx = confusion_matrix(y_bin, pred_y_bin)
        logging.info('confusion matrix:' + str(conf_mtx))
        logging.info('classification report')
        logging.info(classification_report(
            y_bin, pred_y_bin, target_names=['not helpful', 'helpful']))

        return pred_y

    def predict(self, x):
        return self.predictor.predict(x)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.predictor, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.predictor = pickle.load(f)
        return self.predictor


def pipeline(predictor_model, modelname):
    h_pred = HelpfulnessPredictor(predictor_model)

    h_pred.train(train_x, train_y)
    logging.info('Model trained.')
    h_pred.save(modelname)
    logging.info('Model saved.')

    h_pred.load(modelname)
    logging.info('Model loaded.')
    h_pred.test(test_x, test_y)
    logging.info('Model trained.')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename='predict.log'
    )

    train_x, train_y, test_x, test_y = load_finefood()

    zoo = {
        "LR": LinearRegression(),
        "RFR": RandomForestRegressor(random_state=0,
                                     n_estimators=10, verbose=10),
        "SVR_L": SVR(kernel='linear'),
        "SVR_R": SVR(kernel='rbf'),
    }
    for m_name in zoo:
        logging.info('model name: ' + m_name)
        pipeline(predictor_model=zoo[m_name], modelname=m_name + '.pkl')