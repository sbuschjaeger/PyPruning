import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score

from joblib import Parallel,delayed

from .PruningClassifier import PruningClassifier

from .Metrics import error

class OrderPruningClassifier(PruningClassifier):

    def __init__(self, 
        n_estimators = 5, 
        first_round_metric = None,
        metric = error, 
        n_jobs = 8):

        super().__init__()

        assert metric is not None, "You did not provide a valid metric for model selection. Please do so"
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.first_round_metric = first_round_metric
        self.metric = metric
        
        if first_round_metric is None:
            self.first_round_metric = metric

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]

        single_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self.first_round_metric) (iproba, proba.sum(axis=0), target) for iproba in proba
        )
        single_scores = np.array(single_scores)

        best_model = np.argmin(single_scores)
        not_seleced_models = list(range(n_received))
        not_seleced_models.remove(best_model)
        selected_models = [ best_model ]

        for _ in range(self.n_estimators - 1):
            selected_proba = proba[selected_models,:].sum(axis=0)
            scores = []
            for i in not_seleced_models:
                pairwise_score = self.metric(proba[i, :], selected_proba, target)
                scores.append( (i, pairwise_score) ) 
            best_model, _ = min(scores, key = lambda e: e[1])
            not_seleced_models.remove(best_model)
            selected_models.append(best_model)
        
        return selected_models, [1.0 / len(selected_models) for _ in selected_models]