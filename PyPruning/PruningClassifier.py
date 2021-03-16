from abc import ABC, abstractmethod 
import copy

import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

#BaseEstimator, ClassifierMixin
class PruningClassifier(ABC): 
    
    def __init__(self):
        self.weights_ = None
        self.estimators_ = None
        self.n_classes_ = None
        #self.n_estimators = n_estimators
        #self.base_estimator = base_estimator
        #self.n_jobs = n_jobs

        #assert base_estimator is None or isinstance(base_estimator, (RandomForestClassifier, ExtraTreesClassifier)), "If you want to train a model prior to please supply {{RandomForestClassifier, ExtraTreesClassifier}} for training. If you want to prune a custom classifier, pass None and call prune manually"
        #assert n_jobs >= 1, "n_jobs must be at-least 1"

    @abstractmethod
    def prune_(self, proba, target, data = None):
        pass
    
    def prune(self, X, y, estimators, classes = None, n_classes = None):
        """
        classes: Contains the class mappings of each base learner in the order which is returned. Usually this should be something like
            [0,1,2,3,4] for a 5 class problem. However, sometimes weird stuff happens and the mapping might be [2,1,0,3,4]. 
            In this case, you can manually supply the list of mappings
        n_classes: The total number of classes. Usually, this it should be n_classes = len(classes). However, sometimes estimators are only fitted on 
            a subset of data (e.g. during cross validation) and the prune set might contain classes which are not in the oirignal training set and 
            vice-versa. In this case its best to supply n_classes beforehand. 
        """
        if classes is None:
            classes = [e.n_classes_ for e in estimators]
            if (len(set(classes)) > 1):
                raise RuntimeError("Detected a different number of classes for each learner. Please make sure that all learners have their n_classes_ field set to the same value. Alternativley, you may supply a list of classes via the classes parameter to avoid this error.")
                #self.n_classes_ = max(classes)
            else:
                self.classes_ = estimators[0].classes_
                self.n_classes_ = classes[0]
            
            if len(set(y)) > self.n_classes_:
                raise RuntimeError("Detected more classes in the pruning set then the estimators were originally trained on. This usually results in errors or unpredicted classification errors. You can supply a list of classes via the classes parameter. Classes should be arrays / lists containing all possible class labels starting from 0 to C, where C is the number of classes. Please make sure that these are integers as they will be interpreted as such.")
        else:
            self.classes_ = classes
            self.n_classes_ = n_classes

        # Okay this is a bit crazy, but has its reasons. This basically implements the code snippet below, but also takes care of the case where a single estimator did not receive all the labels. In this case predict_proba returns vectors with less than n_classes entries. This can happen in ExtraTrees, but also in RF, especially with unfavourable cross validation splits or large class imbalances. 
        # Anyway, this code construct the desired matrix and copies all predictions to the corresponding locations based on e.classes_. This **should** be correct for numeric classes staring by 0 and also anything which is mapped via the SKLearns LabelEncoder.  
        proba = np.zeros(shape=(len(estimators), X.shape[0], self.n_classes_), dtype=np.float32)
        for i, e in enumerate(estimators):
            proba[i, :, self.classes_.astype(int)] = e.predict_proba(X).T

            
        # proba = []
        # for h in estimators:
        #     proba.append(h.predict_proba(X))
        # proba = np.array(proba)

        self.estimators_ = copy.deepcopy(estimators)
        idx, weights = self.prune_(proba, y, X)        
        estimators_ = []
        for i in idx:
            estimators_.append(self.estimators_[i])
        
        self.estimators_ = estimators_
        self.weights_ = weights 
        
        return self

    # def fit(self, X, y):
    #     self.n_classes_ = len(set(y))
    #     model = self.base_estimator.fit(X,y)
    #     self.prune_(X, model.estimators_)

    #     return self

    def _individual_proba(self, X):
        assert self.estimators_ is not None, "Call prune before calling predict_proba!"
        all_proba = []

        for e in self.estimators_:
            tmp = np.zeros(shape=(X.shape[0], self.n_classes_), dtype=np.float32)
            tmp[:, self.classes_.astype(int)] += e.predict_proba(X)
            all_proba.append(tmp)

        if len(all_proba) == 0:
            return np.zeros(shape=(1, X.shape[0], self.n_classes_), dtype=np.float32)
        else:
            return np.array(all_proba)

    def predict_proba(self, X):
        all_proba = self._individual_proba(X)
        scaled_prob = np.array([w * p for w,p in zip(all_proba, self.weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(proba.argmax(axis=1), axis=0)