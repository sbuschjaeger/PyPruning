import numpy as np

from .PruningClassifier import PruningClassifier

class RandomPruningClassifier(PruningClassifier):
    """ Random pruning.
    This pruning method implements a random pruning which randomly selects n_estimators from the original ensemble and assigns equal weights to each of the classifiers.

    Attributes
    ----------
    n_estimators : int, default is 5
        The number of estimators which should be selected.
    seed : int, optional, default is None
        The random seed for the random selection
    """

    def __init__(self, n_estimators = 5, seed = None):
        """
        Creates a new RandomPruningClassifier.

        Parameters
        ----------

        n_estimators : int, default is 5
            The number of estimators which should be selected.
        seed : int, optional, default is None
            The random seed for the random selection
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.seed = seed

    def prune_(self, proba, target, data = None):
        # TODO  It seems that numpy changed the way it handles randomization. We should maybe adapt their new interface
        np.random.seed(self.seed)
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        else:
            return np.random.choice(range(0, n_received),size=self.n_estimators), [1.0 / self.n_estimators for _ in range(self.n_estimators)]

        