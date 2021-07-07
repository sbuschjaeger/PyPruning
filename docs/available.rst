Pruning an ensemble
===================

.. toctree::
   :maxdepth: 3
   :hidden:

   greedy
   rank
   cluster
   MIQP
   prox
   random

In general, there are four categories of pruning methods currently available

- Ranking based pruning
- Greedy pruning
- Clustering based pruning
- Optimization based pruning

Optimization based approaches can further be decomposed into different approaches for optimization:

- Mixed Quadratic Integer Programming
- Gradient descent and related approaches

Last, we found that a random selection also serves as a decent baseline leading to a total of 6 different pruning methods:


- :class:`~PyPruning.RandomPruningClassifier`: Selects a random subset of classifiers. This is mainly used as a baseline.
- :class:`~PyPruning.RankPruningClassifier`: Rank each classifier according to a given metric and then select the best K classifier.
- :class:`~PyPruning.ClusterPruningClassifier`: Cluster the classifiers according to a clustering method and then select a representative from each cluster to from the sub-ensemble.
- :class:`~PyPruning.GreedyPruningClassifier`: Proceeds in rounds and selects the best classifier in each round given the already selected sub-ensemble. 
- :class:`~PyPruning.MIQPPruningClassifier`: Constructs a mixed-integer quadratic problem and optimizes this to compute the best sub ensemble. 
- :class:`~PyPruning.ProxPruningClassifier`: Minimize a (regularized) loss function via (stochastic) proximal gradient descent over the ensembles weights. 


There is a decent amount of different pruning methods available in literature which mostly differs by the scoring / loss functions used to score the performance of sub-ensembles. Thus, we tried to make our implementation as flexible as possible. Most methods support custom metrics so that you can implement your pruning methods. For more information please check out :doc:`extending`. 

Every pruning method offers a ``prune``, ``predict_proba`` and ``predict`` method. Central for pruning is the ``prune(self, X, y, estimators)`` method, where 

- ``X`` are the pruning examples, 
- ``y`` are the corresponding pruning targets 
- ``estimators`` is the list of estimators to be pruned. 
- ``classes`` a list of classes this classifier was trained on which corresponding to the order of `predict_proba`. If this is `None` we try to infer this from the base estimators
- ``n_classes`` the total number of classes. If this is ``None`` we try to infer this from the base estimators

We assume that each estimator in ``estimators`` supports ``copy.deepcopy()`` and offers the following functions / fields: 

- ``predict(X)``: Returns the class predictions for each example in X. Result should be ``(X.shape[0], )``
- ``predict_proba(X)``: Returns the class probabilities for each example in X. Result should be ``(X.shape[0], n_classes_)`` where ``n_classes_`` is the number of classes the classifier was trained on.


**Note**: These requirements are usually met by scikit-learn and scikit-learn compatible estimators, but our code is currently does not implement the scikit-learn interface for BaseEstimators. Differently put: You should be able to prune any scikit-learn estimators, but we do not guarantee that the pruned ensembles fully integrates into scikit-learn. We try to enhance support for other libraries including scikit-learn in the future.   

A complete example might look like this:

.. code-block:: python

   # Load some data
   data, target = load_digits(return_X_y = True)

   # Perform a test / train / prune split
   XTP, Xtest, ytp, ytest = train_test_split(data, target, test_size=0.25, random_state=42)
   Xtrain, Xprune, ytrain, yprune = train_test_split(XTP, ytp, test_size=0.25, random_state=42)

   n_base = 128
   n_prune = 8

   # Train a "large" initial random forest
   model = RandomForestClassifier(n_estimators=n_base)
   model.fit(XTP, ytp)
   pred = model.predict(Xtest)

   print("Accuracy of RF trained on XTrain + XPrune with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

   # Train a "small" initial random forest for reference
   model = RandomForestClassifier(n_estimators=n_base)
   model.fit(Xtrain, ytrain)
   pred = model.predict(Xtest)

   print("Accuracy of RF trained on XTrain only with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

   # Use different pruning methods to prune the large forest 
   pruned_model = RandomPruningClassifier(n_estimators = n_prune)
   pruned_model.prune(Xprune, yprune, model.estimators_)
   pred = pruned_model.predict(Xtest)
   print("Accuracy of RandomPruningClassifier with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))

   pruned_model = GreedyPruningClassifier(n_prune, single_metric = error)
   pruned_model.prune(Xtrain, ytrain, model.estimators_)
   pred = pruned_model.predict(Xtest)
   print("GreedyPruningClassifier with {} estimators and {} metric is {} %".format(n_prune, m.__name__, 100.0 * accuracy_score(ytest, pred)))

   pruned_model = MIQPPruningClassifier(n_prune, single_metric = error)
   pruned_model.prune(Xtrain, ytrain, model.estimators_)
   pred = pruned_model.predict(Xtest)
   print("MIQPPruningClassifier with {} estimators and {} metric is {} %".format(n_prune, m.__name__, 100.0 * accuracy_score(ytest, pred)))
