
Extending PyPruning
===================

.. toctree::
   :maxdepth: 3
   :hidden:

   self

If you want to implement your own pruning method then there are two ways:

Implementing a custom metric
****************************

You can implement your own metric for :class:`~PyPruning.GreedyPruningClassifier`, :class:`~PyPruning.MIQPPruningClassifier` or a :class:`~PyPruning.RankPruningClassifier` you simply have to implement a python function that should be **minimized**. The specific interface required by each method slightly differs so please check out the specific documentation for the method of your choice. In all cases, each method expects functions with at-least three parameters

- ``i`` (int): The classifier which should be rated
- ``ensemble_proba`` (A (M, N, C) matrix ): All N predictions of all M classifier in the entire ensemble for all C classes
- ``target`` (list / array): A list / array of class targets.

Note that ``ensemble_proba`` contains all class probabilities predicted by all members in the ensemble. So in order to get individual class predictions for the i-th classifier you can access it via ``ensemble_proba[i,:,:]``. A complete example which simply computes the error of each method would be

.. code-block:: python

   def individual_error(i, ensemble_proba, target):
      iproba = ensemble_proba[i,:,:]
      return (iproba.argmax(axis=1) != target).mean()

Implementing a custom pruner
****************************

You can implement your own pruner as a well. In this case you just have to implement the :class:`~PyPruning.PruningClassifier` class. To do so, you just need to implement the ``prune_(self, proba, target)`` function which receives a list of all predictions of all classifiers as well as the corresponding data and targets. The function is supposed to return a list of indices corresponding to the chosen estimators as well as the corresponding weights. If you need access to the estimators as well (and not just their predictions) you can access ``self.estimators_`` which already contains a copy of each classier. For more details have a look at the :class:`PruningClassifier.py` interface. An example implementation could be:

.. code-block:: python

   class RandomPruningClassifier(PruningClassifier):

      def __init__(self):
         super().__init__()

      def prune_(self, proba, target, data = None):
         n_received = len(proba)
         if self.n_estimators >= n_received:
               return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
         else:
               return np.random.choice(range(0, n_received),size=self.n_estimators), [1.0 / self.n_estimators for _ in range(self.n_estimators)]

PyPruning.PruningClassifier module
----------------------------------

.. automodule:: PyPruning.PruningClassifier
   :members:
   :undoc-members:
   :show-inheritance:
