import numpy as np

from PruningClassifier import PruningClassifier

class RandomPruningClassifier(PruningClassifier):

    def __init__(self, 
        n_estimators = 5, 
        base_estimator = None, 
        n_jobs = 8):
        
        super().__init__(n_estimators, base_estimator, n_jobs)

    def prune_(self, proba, target):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        else:
            return np.random.choice(range(0, n_received),size=self.n_estimators), [1.0 / self.n_estimators for _ in range(self.n_estimators)]

        