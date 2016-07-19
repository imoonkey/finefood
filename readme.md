Language:
Python 2.7+

Required packages (can be installed using pip):
bs4, nltk, numpy, scipy, sklearn, cPickle

To reproduce the intermediate outputs:

run `prep.py` for finefoods.pkl, train.pkl, test.pkl

run `vectorization.py` for train_x.mtx, train_y.npy, test_x.mtx, test_y.npy

Confusion matrix: (Y: helpful, N: not helpful)
 | Predicted: Y| Predicted: N
---| --- | --- 
 ** Actual: Y ** | 0.5 | 0.5
 ** Actual: N ** | 0.5 | 0.5