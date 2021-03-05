import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score

from joblib import Parallel,delayed

from .PruningClassifier import PruningClassifier

from .Metrics import error

class RankPruningClassifier(PruningClassifier):

    def __init__(self, 
        n_estimators = 5, 
        metric = error,
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
            delayed(self.metric) (iproba, proba, target) for iproba in proba
        )
        single_scores = np.array(single_scores)

        return np.argpartition(single_scores, self.n_estimators)[:self.n_estimators], [1.0 / self.n_estimators for _ in range(self.n_estimators)]
        