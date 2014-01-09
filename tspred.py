#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# vim: ts=4 sts=4 sw=4 tw=79 sta et
"""%prog [options]
Python source code - @todo
"""

__author__ = 'Patrick Butler'
__email__ = 'pbutler@killertux.org'
__version__ = '0.0.1'


import numpy as np
from sklearn import base

class PredTS(base.BaseEstimator, base.TransformerMixin):
    """
    Using a regression predictor for each n data points train on the previous
    n-1 and attempt to predict the nth

    :param recency: fit using only the last n periods
    :param lag: predict data points this many periods into the future
    :param retrain_period: retrain every n periods
    :param verbose: True/False, whether to print extra information

    """
    def __init__(self, clf, recency=None, lag=0, retrain_period=1, verbose=False):
        self.clf = clf
        self.recency = recency
        self.lag = lag
        self.verbose = verbose
        self.retrain_period = retrain_period

    def fit(self, X, y):
        """
        semi-internal function use predict
        """
        X = np.asarray(X)
        y = np.asarray(y)
        recency = self.recency
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched arrays")
        if recency is not None:  # and X.shape[0]
            X = X[-recency:, :]
            y = y[-recency:]
        if self.verbose:
            print X, "fits", y
        #print X.shape, y.shape
        self.clf.fit(X, y)
        return self

    def predict(self, X, y):
        """
        pts.predict(X, y)
        :param X: features
        :param y: target
        """
        X = np.asarray(X)
        y = np.asarray(y)
        lag = self.lag
        y_pred = np.zeros(y.shape[0] + 1)
        n = X.shape[0]
        y_pred = np.zeros(n + lag - 1)
        y_pred[:] = np.NaN
        for i in range(1, X.shape[0]):
            #print y[lag:lag + i]
            if (i - 1) % self.retrain_period == 0:
                if X[:i, :].shape[0] == y[lag:lag + i].shape[0]:
                    self.fit(X[:i, :], y[lag:lag + i])

            pidx = i + lag - 1
            y_pred[pidx] = self.clf.predict(X[i])
            if self.verbose:
                print "y_pred_%d(%s) = %f" % (pidx, X[i], y_pred[i + lag - 1]),
                print self.clf.coef_, self.clf.intercept_
            #if i == 2:
            #    break
        return y_pred


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="don't print status messages to stdout")
    #parser.add_argument("-n", "--nfeatures", default=10, type=int)
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + __version__)
    #parser.add_argument('args', metavar='args', type=str, nargs='*',
    #                    help='an integer for the accumulator')
    options = parser.parse_args()

    nfeatures = options.nfeatures

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))

def test_basic():
    from sklearn import linear_model
    nfeatures = 2
    nsamples = 10
    test_data = np.arange(nsamples * nfeatures).reshape((-1, nfeatures))
    test_target = np.dot(test_data, np.arange(nfeatures) + 1)
    print test_target
    pts = PredTS(linear_model.LinearRegression(False), lag=0, recency=2,
                 verbose=True)
    result = pts.predict(test_data, test_target)
    assert np.sum(pts.clf.coef_ - np.arange(nfeatures) - 1) < 1e-5
