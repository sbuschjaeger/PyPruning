import numpy as np

from .PruningClassifier import PruningClassifier

# TODO Add random seed
class RandomPruningClassifier(PruningClassifier):

    def __init__(self, n_estimators = 5):
        
        super().__init__()
        self.n_estimators = n_estimators

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        else:
            return np.random.choice(range(0, n_received),size=self.n_estimators), [1.0 / self.n_estimators for _ in range(self.n_estimators)]

        