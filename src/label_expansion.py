# python expand_labels.py out.synthetic.tsv
"""
one-class learning to expand T_prox


"""
from scipy import stats
import pandas as pd
from collections import Counter, defaultdict
import sys
import re
import sklearn
from sklearn import feature_extraction
import itertools
from sklearn import metrics
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
import random
import util




class PUClassifier(object):
    """
    Learning classifiers from only positive and unlabeled data (Noto et al., 2008)

        self.inner_estimator: g(x), a traditional classifier for p(s=1 | x) and c computation
        self.outer_estimator: f(x), classifier learned from c-based data reweighting

    example:
        s = y.copy()
        s[:n_unlabeled] = 0  # overwrite some 1's
        pu = PUClassifier(n_folds=5)
        pu.fit(X, s)
        ypred = pu.predict(X[s==0])

    """
    def __init__(self, n_folds=5, inner_alpha='optimal', outer_alpha='optimal'):
        self.inner_alpha = inner_alpha
        # if inner_alpha == 'optimal':
        #     self.inner_estimator = GridSearchCV(SGDClassifier(loss="log", penalty="l2"), param_grid={"alpha": np.logspace(-4, 0, 10)})
        # else:
        self.inner_estimator = SGDClassifier(loss="log", penalty="l2", alpha=inner_alpha)

        self.outer_alpha = outer_alpha
        self.n_folds = n_folds
        self.fitted = False

    # for sklearn compatibility
    def get_params(self, deep=True):
        return {"n_folds": self.n_folds, 'inner_alpha': self.inner_alpha, 'outer_alpha': self.outer_alpha}    

    # for sklearn compatibility
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def compute_c(self, X, s):
        c = np.zeros(self.n_folds)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        for i, (itr, ite) in enumerate(skf.split(s, s)):
            self.inner_estimator.fit(X[itr], s[itr])
            c[i] = self.inner_estimator.predict_proba(X[ite][s[ite]==1])[:,1].mean()
        self.c = c.mean()


    def sample(self, X, s):
        if not hasattr(self, "c"):
            self.compute_c(X, s)
        X_positive = X[s==1]
        X_unlabeled = X[s==0]
        n_positive = X_positive.shape[0]
        n_unlabeled = X_unlabeled.shape[0]

        X_train = np.r_[X_positive, X_unlabeled, X_unlabeled]
        y_train = np.concatenate([
            np.repeat(1, n_positive), 
            np.repeat(1, n_unlabeled), 
            np.repeat(0, n_unlabeled)])

        self.inner_estimator.fit(X, s)
        p_unlabeled = self.inner_estimator.predict_proba(X_unlabeled)[:,1]
        w_positive = ((1 - self.c) / self.c) * (p_unlabeled / (1 - p_unlabeled))
        w_negative = 1 - w_positive
        sample_weight = np.concatenate(
            [np.repeat(1.0, n_positive), 
            w_positive, 
            w_negative])
        return X_train, y_train, sample_weight


    def find_best_alpha(self, X_train, y_train, sample_weight):
        # do a lil hyperparam search
        alphas = np.logspace(-4, 0, 10)
        class_weights = [{1:1}]
        n_folds = 3
        best_score = -np.inf
        best_alpha = None
        best_class_weight = None
        for alpha, class_weight in itertools.product(alphas, class_weights):
            scores = np.zeros(n_folds)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
            for i, (itr, ite) in enumerate(skf.split(y_train, y_train)):
                clf = SGDClassifier(
                    loss="log",  #hinge for svm
                    penalty="l2", 
                    alpha=alpha, 
                    class_weight=class_weight).fit(
                    X_train[itr], y_train[itr], sample_weight=sample_weight[itr])
                ypred = clf.predict(X_train[ite])
                scores[i] = metrics.accuracy_score(y_train[ite], ypred, sample_weight=sample_weight[ite])
            this_score = scores.mean()
            if this_score > best_score:
                best_score = this_score
                best_alpha = alpha
                best_class_weight = class_weight
        return best_alpha

    def fit(self, X, s):
        if not hasattr(self, "c"):
            self.compute_c(X, s)

        X_train, y_train, sample_weight = self.sample(X, s)

        if self.outer_alpha == 'optimal':
            alpha = self.find_best_alpha(X_train, y_train, sample_weight)
        else:
            alpha = self.outer_alpha

        self.outer_estimator = SGDClassifier(
            loss="log", 
            penalty="l2", 
            alpha=alpha, 
            class_weight={1:1}).fit(
            X_train, y_train, sample_weight=sample_weight)

        self.fitted = True


    def predict_proba(self, X):
        if not self.fitted:
            raise Exception('not fitted yet!')

        return self.outer_estimator.predict_proba(X)

    def predict(self, X, threshold=0.8):
        if not self.fitted:
            raise Exception('not fitted yet!')
        raise NotImplementedError
        # print(stats.zscore(self.predict_proba(X)[:, 1]))
        # TODO USE SOMETHING OTHER THAN SVM SO TAHT YOU CAN GET PROBS
        # return self.outer_estimator.predict(X)
        return np.array([1. if p > threshold else 0. for p in self.predict_proba(X)[:, 1]])


def expand_variable(estimator, X, T, threshold=0.8, only_zeros=True):
    if only_zeros:
        X_subset = X[T == 0]
    else:
        X_subset = X

    probs = estimator.predict_proba(X_subset)[:, 1]
    probs = stats.zscore(probs)
    T_new = np.array([1. if p > threshold else 0. for p in probs])

    T_expanded = T.copy()
    if only_zeros:
        T_expanded[T == 0] = T_new
    else:
        T_expanded = T_new

    return T_expanded

