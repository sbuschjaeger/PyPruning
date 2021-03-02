from abc import ABC, abstractmethod 
import copy

import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

class PruningClassifier(ABC, BaseEstimator, ClassifierMixin): 
    
    def __init__(self, n_estimators = 5, base_estimator = None, n_jobs = 8):
        self.weights_ = None
        self.estimators_ = None
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.n_classes_ = None
        self.n_jobs = n_jobs

        assert base_estimator is None or isinstance(base_estimator, (RandomForestClassifier, ExtraTreesClassifier)), "If you want to train a model prior to please supply {{RandomForestClassifier, ExtraTreesClassifier}} for training. If you want to prune a custom classifier, pass None and call prune manually"
        assert n_jobs >= 1, "n_jobs must be at-least 1"

    @abstractmethod
    def prune_(self, proba, target):
        pass

    def prune(self, X, y, estimators):
        if self.n_classes_ is None:
            classes = [e.n_classes_ for e in estimators]
            if (len(set(classes)) > 1):
                print("WARNING: Detected a different number of classes for each learner")
                self.n_classes_ = max(classes)
            else:
                self.n_classes_ = classes[0]

        # TODO parallelize this
        proba = []
        for h in estimators:
            proba.append(h.predict_proba(X))
        proba = np.array(proba)

        self.estimators_ = copy.deepcopy(estimators)
        idx, weights = self.prune_(proba, y)        
        estimators_ = []
        for i in idx:
            estimators_.append(self.estimators_[i])
        
        self.estimators_ = estimators_
        self.weights_ = weights 
        
        return self

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        model = self.base_estimator.fit(X,y)
        self.prune_(X, model.estimators_)

        return self

    def _individual_proba(self, X):
        assert self.estimators_ is not None, "Call prune before calling predict_proba!"
        all_proba = []

        for e in self.estimators_:
            tmp = np.zeros(shape=(X.shape[0], self.n_classes_), dtype=np.float32)
            tmp[:, e.classes_.astype(int)] += e.predict_proba(X)
            all_proba.append(tmp)

        return np.array(all_proba)

    def predict_proba(self, X):
        all_proba = self._individual_proba(X)
        scaled_prob = np.array([w * p for w,p in zip(all_proba, self.weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)