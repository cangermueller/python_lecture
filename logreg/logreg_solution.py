#!/usr/bin/env python

import numpy as np
import sklearn.metrics as met
import sklearn.datasets as ds
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import time


class LogisticRegressionBase(object):
    """ Abstract class for logistic regression.

    NOTE: Do not modify!
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.lambda_ = 0.05

    def f(self, theta):
        """ Evaluate objective function f at theta. """
        return None

    def fprime(self, theta):
        """ Compute first derivative of f at theta. """
        return None

    def _init_params(self):
        """ Initialize model parameters. """
        self.theta0 = np.random.randn(self.n_features)

    def fit(self):
        """ Fit theta via gradient descent. """
        self._init_params()
        self.theta = fmin_bfgs(f=self.f, x0=self.theta0,
                               fprime=self.fprime)
        self._f0 = self.f(self.theta0)
        self._f = self.f(self.theta)
        print('f(theta0): %.3f' % self._f0)
        print('f(theta): %.3f' % self._f)
        print('Training accuracy: %.3f' % met.accuracy_score(self.y, self.predict(hard=True)))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X=None, hard=False):
        if X is None:
            X = self.X
        p = self.sigmoid(X.dot(self.theta))
        return np.round(p) if hard else p


class LogisticRegressionYours(LogisticRegressionBase):
    """ Your implementation of logistic regression."""

    def f(self, theta):
        z = self.X.dot(theta)
        rv = -np.sum(self.y * z - np.log(1 + np.exp(z)))
        rv += 0.5 * self.lambda_ * theta.dot(theta)
        return rv

    def fprime(self, theta):
        div = self.y -  1 / (1 + np.exp(-self.X.dot(theta)))
        return -np.sum(self.X * div.reshape(-1, 1), axis=0) + self.lambda_ * theta


def evaluate(lr):
    """ Evaluate performance of logistic regression model. """

    t0 = time.clock()
    lr.fit()
    t1 = time.clock()

    print('Fitting time (sec): %d' % (t1 - t0))
    print('Accuracy: %.3f' % (met.accuracy_score(lr.y, lr.predict(hard=True))))
    print('AUC: %.3f' % (met.roc_auc_score(lr.y, lr.predict())))

    fpr, tpr, thr = met.roc_curve(lr.y, lr.predict())
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(fpr, tpr, linewidth=3)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.grid()
    fig.savefig('roc.png')



if __name__ == '__main__':
    # Randomly generate classification dataset
    np.random.seed(0)
    n_samples = 1000
    n_features = 100
    X, y = ds.make_classification(n_samples=n_samples,
                            n_features=n_features,
                            n_informative=int(n_features / 2))
    evaluate(LogisticRegressionYours(X, y))
