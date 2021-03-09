import numpy as np
from sklearn.metrics import roc_auc_score

from joblib import Parallel,delayed
from sklearn.utils import axis0_safe_slice

from .PruningClassifier import PruningClassifier

def error(i, ensemble_proba, sub_proba, target):
    iproba = ensemble_proba[i,:,:]
    pred = 1.0 / (1 + len(sub_proba)) * (sub_proba.sum(axis=0) + iproba)
    return (pred.argmax(axis=1) != target).mean() 

def neg_auc(i, ensemble_proba, sub_proba, target):
    iproba = ensemble_proba[i,:,:]
    pred = 1.0 / (1 + len(sub_proba)) * (sub_proba.sum(axis=0) + iproba)

    if(sub_proba.shape[1] == 2):
        pred = pred.argmax(axis=1)
        return - 1.0 * roc_auc_score(target, pred)
    else:
        return - 1.0 * roc_auc_score(target, pred, multi_class="ovr")

# Paper:   Pruning in Ordered Bagging Ensembles
# Authors: Martinez-Munoz and Suarez 
def complementariness(i, ensemble_proba, sub_proba, target):
    iproba = ensemble_proba[i,:,:]
    b1 = (iproba.argmax(axis=1) == target)
    b2 = (sub_proba.sum(axis=0).argmax(axis=1) != target)
    return - 1.0 * np.sum(np.logical_and(b1, b2))

# Paper:   Pruning in Ordered Bagging Ensembles
# Authors: Martinez-Munoz and Suarez 
def margin_distance(i, ensemble_proba, sub_proba, target):
    iproba = ensemble_proba[i,:,:]
    c_refs = []

    for sub in sub_proba:
        c_refs.append( 2 * (sub.argmax(axis=1) == target) - 1.0)
    
    c_refs.append(2 * (iproba.argmax(axis=1) == target) - 1.0)
    c_refs = np.mean(c_refs)

    p = np.random.uniform(0, 0.25, len(target))
    return np.mean((p - c_refs)**2)

# Paper:   Diversity Regularized Ensemble Pruning
# Authors: Li, Yu and Zhou 2012
def drep(i, ensemble_proba, sub_proba, target):
    iproba = ensemble_proba[i,:,:].argmax(axis=1)
    
    if len(sub_proba) == 0:
        return (iproba != target).mean()
    else:
        sproba = sub_proba.mean(axis=0).argmax(axis=1)

        # This implements a multi-class version of eq (9) from the paper. Originally, the paper considers binary classification problems with labels {-1,1}. It counts the number of the same predictions as the difference between two classifier. 
        diff = (iproba == sproba).sum()
        return diff

class GreedyPruningClassifier(PruningClassifier):

    def __init__(self, 
        n_estimators = 5, 
        metric = error, 
        n_jobs = 8):

        super().__init__()

        assert metric is not None, "You did not provide a valid metric for model selection. Please do so"
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.metric = metric

    # I assume that Parallel keeps the order of evaluations regardless of its backend (see eg. https://stackoverflow.com/questions/56659294/does-joblib-parallel-keep-the-original-order-of-data-passed)
    # But for safty measures we also return the index of the current model
    def _metric(self, i, ensemble_proba, sub_proba, target):
        return (i, self.metric(i, ensemble_proba, sub_proba, target))

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]

        not_seleced_models = list(range(n_received))
        selected_models = [ ]

        for _ in range(self.n_estimators):
            scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._metric) ( i, proba, proba[selected_models, :, :], target) for i in not_seleced_models
            )

            best_model, _ = min(scores, key = lambda e: e[1])
            not_seleced_models.remove(best_model)
            selected_models.append(best_model)

        return selected_models, [1.0 / len(selected_models) for _ in selected_models]