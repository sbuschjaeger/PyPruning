import numpy as np

from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from joblib import Parallel,delayed

from .PruningClassifier import PruningClassifier

# Paper:   Margin & Diversity based ordering ensemble pruning
# Authors: Guo et al. 2018
#
def individual_margin_diversity(i, ensemble_proba, target, alpha = 0.2):
    iproba = ensemble_proba[i,:,:]
    n = iproba.shape[0]

    predictions = iproba.argmax(axis=1)
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
    V = V.sum(axis=0)

    #V = jproba #all_proba.sum(axis=0)#.argmax(axis=1)
    MDM = 0
    
    for j in range(n):
        if (predictions[j] == target[j]):
            
            # special case for margin: prediction for label with majority of votes
            if(predictions[j] == np.argmax(V[j,:])):
                # calculate margin with second highest number of votes
                sortedArray = np.sort(np.copy(V[j,:]))
                
                # check whether 1. and 2. max vot counts are equal! (margin = 0)
                if(sortedArray[-2] == np.max(V[j,:])):
                    margin = (  V[j, target[j]]  - (sortedArray[-2] -1)   ) / n
                else:
                    margin = (  V[j, target[j]]  - sortedArray[-2]   ) / n
                   
            else:
                # usual case for margin: prediction not label with majority of votes
                margin = (  V[j, target[j]]  - np.max(V[j,:])   ) / n
            
            
            # somehow theres still a rare case for margin == 0
            if(margin == 0):
                margin = 0.01
            
            fm = np.log(abs(margin))
            fd = np.log(V[j, target[j]] / n)
            MDM = MDM + (alpha*fm) + ((1-alpha)*fd)
    return - 1.0 * MDM

# Paper:   Ensemble Pruning via individual contribution
# Authors: Lu et al. 2010
#
def individual_contribution(i, ensemble_proba, target):
    iproba = ensemble_proba[i,:,:]
    n = iproba.shape[0]

    predictions = iproba.argmax(axis=1)
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
    V = V.sum(axis=0)

    IC = 0
    #V = all_proba.argmax(axis=2)
    #predictions = iproba.argmax(axis=1)
    #V = all_proba.sum(axis=0)#.argmax(axis=1)

    for j in range(n):
        if (predictions[j] == target[j]):
            
            # case 1 (minority group)
            # label with majority votes on datapoint  = np.argmax(V[j, :]) 
            if(predictions[j] != np.argmax(V[j,:])):
                IC = IC + (2*(np.max(V[j,:])) - V[j, predictions[j]])
                
            else: # case 2 (majority group)
                # calculate second largest nr of votes on datapoint i
                sortedArray = np.sort(np.copy(V[j,:]))
                IC = IC + (sortedArray[-2])
                
        else:
            # case 3 (wrong prediction)
            IC = IC + (V[j, target[j]]  -  V[j, predictions[j]] - np.max(V[j,:]) )
    return - 1.0 * IC

def individual_error(i, ensemble_proba, target):
    iproba = ensemble_proba[i,:,:]
    return (iproba.argmax(axis=1) != target).mean()

# Area under the ROC-Curve (AUC) metric for ensemble pruning
# Uses the sklearn-implementation
def individual_neg_auc(i, ensemble_proba, target):
    iproba = ensemble_proba[i,:,:]
    if(iproba.shape[1] == 2):
        iproba = iproba.argmax(axis=1)
        return - 1.0 * roc_auc_score(target, iproba)
    else:
        return - 1.0 * roc_auc_score(target, iproba, multi_class="ovr")

# Paper:   Pruning Adaptive Boosting at ICML 1997
# Authors: Margineantu and Dietterich 
def individual_kappa_statistic(i, ensemble_proba, target):
    scores = []
    iproba = ensemble_proba[i,:,:].argmax(axis=1)

    for j, jproba in enumerate(ensemble_proba):
        if j != i:
            # See https://github.com/scikit-learn/scikit-learn/issues/14256
            # and https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
            with np.errstate(divide='ignore',invalid='ignore'):
                score = cohen_kappa_score(iproba, jproba.argmax(axis=1))
                if np.isnan(score):
                    scores.append(0.0)
                else:
                    scores.append(score)
    return min(scores)

# Paper:   Pruning in Ordered Bagging Ensembles
# Authors: Martinez-Munoz and Suarez 
def reference_vector(i, ensemble_proba, target):
    ref = 2 * (ensemble_proba.argmax(axis=1) == target) - 1.0
    ipred = 2 * (ensemble_proba[i,:].argmax(axis=1) == target) - 1.0

    # Note: The paper describes a slightly different distance metric which constructs the projection of ipred to a reference vector. Unfortunatly, the specific implementation of this reference vector is not epxlained in detail in the paper.
    # However, the authors also note two things:
    # (1) They use all classifier with an angle <= pi/2 which can lead to more than n_estimator classifier. Thus we need to present an ordering based on the angles and pick the first n_estimator.
    # (2) "The classifiers are ordered by increasing values of the angle between the signature vectors of the individual classifiers and the reference vector". 
    # 
    # ref and ipred follow the exact definitions as presented in the paper (eq. 3) and the cosine_similary is the most direct implementation of "the angle between signature and reference vector" 
    # 
    return -1.0 * cosine_similarity(ipred, ref)

class RankPruningClassifier(PruningClassifier):

    def __init__(self, 
        n_estimators = 5, 
        metric = individual_error,
        n_jobs = 8):

        super().__init__()

        assert metric is not None, "You must provide a valid metric!"
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.metric = metric

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        
        single_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self.metric) (i, proba, target) for i in range(n_received)
        )
        single_scores = np.array(single_scores)

        return np.argpartition(single_scores, self.n_estimators)[:self.n_estimators], [1.0 / self.n_estimators for _ in range(self.n_estimators)]
        