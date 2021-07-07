from abc import ABC, abstractmethod 
import copy

import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

class PruningClassifier(ABC): 
    """ This abstract class forms the basis of all pruning methods and offers a unified interface. New pruning methods must extend this class and implement the prune_ method as detailed below. 

    Attributes
    ----------
    weights_: numpy array 
        An array of weights corresponding to each classifier in self.estimators_
    estimators_ : list
        A list of estimators
    n_classes_ : int 
        The number of classes the pruned ensemble supports.
    """
    def __init__(self):
        self.weights_ = None
        self.estimators_ = None
        self.n_classes_ = None

    @abstractmethod
    def prune_(self, proba, target, data = None):
        """
        Prunes the ensemble using the ensemble predictions proba and the pruning data targets / data. If the pruning method requires access to the original ensemble members you can access these via self.estimators_. Note that self.estimators_ is already a deep-copy of the estimators so you are also free to change the estimators in this list if you want to.

        Parameters
        ----------
        proba : numpy matrix
            A (N,M,C) matrix which contains the individual predictions of each ensemble member on the pruning data. Each ensemble prediction is generated via predict_proba. N is size of the pruning data, M the size of the base ensemble and C is the number of classes
        
        target: numpy array of ints 
            A numpy array or list of N integers where each integer represents the class for each example. Classes should start with 0, so that for C classes the integer 0,1,...,C-1 are used
        
        data:  numpy matrix, optional
            The data points in a (N, M) matrix on which the proba has been computed, where N is the pruning set size and M is the number of classifier in the original ensemble. This can be used by a pruning method if required, but most methods do not require the actual data points but only the individual predictions. 
        
        Returns
        -------
        A tuple of indices and weights (idx, weights) with the following properties:
        idx : numpy array / list of ints
            A list of integers which classifier should be selected from self.estimators_. Any changes made to self.estimators_ are also reflected here, so make sure that the order of classifier in proba and self.estimators_ remains the same (or you return idx accordingly)
        
        weights: numpy array / list of floats
            The individual weights for each selected classifier. The size of this array should match the size of idx (and not the size of the original base ensemble). 
        """
        pass
    
    def prune(self, X, y, estimators, classes = None, n_classes = None):
        """
        Prunes the given ensemble on the supplied dataset. There are a few assumptions placed on the behavior of the individual classifiers in `estimators`. If you use scikit-learn classifier and any classifier implementing their interface they should work without a problem. The detailed assumptions are listed below:
         
        - `predict_proba`: Each estimator should offer a predict_proba function which returns the class probabilities for each class on a batch of data
        - `n_classes_`: Each estimator should offer a field on the number of classes it has been trained on. Ideally, this should be the same for all classifier in the ensemble but might differ e.g. due to different bootstrap samples. This field is not accessed if you manually supply `n_classes` as parameter to this function
        - `classes_`: Each estimator should offer a class mapping which shows the order of classes returned by predict_proba. Usually this should simply be [0,1,2,3,4] for 5 classes, but if your classifier returns class probabilities in a different order, e.g. [2,1,0,3,4] you should store this order in `classes_`. This field is not accessed if you manually supply `classes` as parameter to this function

        For pruning this function calls `predict_proba` on each classifier in `estimators` and then calls `prune_` of the implementing class. After pruning, it extracts the selected classifiers from `estimators` with their corresponding weight and stores them in `self.weights_` and `self.estimators_`

        Parameters
        ----------
        X : numpy matrix
            A (N, d) matrix with the datapoints used for pruning where N is the number of data points and d is the dimensionality
        
        Y : numpy array / list of ints
            A numpy array or list of N integers where each integer represents the class for each example. Classes should start with 0, so that for C classes the integer 0,1,...,C-1 are used
        
        estimators : list
            A list of estimators from which the pruned ensemble is selected.
        
        classes : numpy array / list of ints
            Contains the class mappings of each base learner in the order which is returned by predict_proba. Usually this should be something like [0,1,2,3,4] for a 5 class problem. However, sometimes weird stuff happens and the mapping might be [2,1,0,3,4]. In this case, you can manually supply the list of mappings
        
        n_classes: int
            The total number of classes. Usually, this it should be n_classes = len(classes). However, sometimes estimators are only fitted on a subset of data (e.g. during cross validation or bootstrapping) and the prune set might contain classes which are not in the original training set and vice-versa. In this case its best to supply n_classes beforehand. 

        Returns
        -------
        The pruned ensemble (self).
        """
        if classes is None:
            classes = [e.n_classes_ for e in estimators]
            if (len(set(classes)) > 1):
                raise RuntimeError("Detected a different number of classes for each learner. Please make sure that all learners have their n_classes_ field set to the same value. Alternatively, you may supply a list of classes via the classes parameter to avoid this error.")
                #self.n_classes_ = max(classes)
            else:
                self.classes_ = estimators[0].classes_
                self.n_classes_ = classes[0]
            
            if len(set(y)) > self.n_classes_:
                raise RuntimeError("Detected more classes in the pruning set then the estimators were originally trained on. This usually results in errors or unpredicted classification errors. You can supply a list of classes via the classes parameter. Classes should be arrays / lists containing all possible class labels starting from 0 to C, where C is the number of classes. Please make sure that these are integers as they will be interpreted as such.")
        else:
            self.classes_ = classes
            self.n_classes_ = n_classes

        # Okay this is a bit crazy, but has its reasons. This basically implements the for-loop below, but also takes care of the case where a single estimator did not receive all the labels. In this case predict_proba returns vectors with less than n_classes entries. This can happen in ExtraTrees, but also in RF, especially with unfavorable cross validation splits or large class imbalances. 
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

    def _individual_proba(self, X):
        """ Predict class probabilities for each individual learner in the ensemble without considering the weights.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,C)
            The predicted class probabilities for each learner.
        """
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
        """ Predict class probabilities using the pruned model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,C)
            The predicted class probabilities. 
        """
        all_proba = self._individual_proba(X)
        scaled_prob = np.array([w * p for w,p in zip(all_proba, self.weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict(self, X):
        """ Predict classes using the pruned model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted classes. 

        """
        proba = self.predict_proba(X)
        return self.classes_.take(proba.argmax(axis=1), axis=0)