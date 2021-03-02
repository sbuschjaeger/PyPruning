# PyPruning

Ensemble Algorithms implemented in Python. You can install this package via pip

    pip install git+https://github.com/sbuschjaeger/PyPruning.git

This package provides implementations for some common ensemble pruning algorithms. Pruning algorithms aim to select the best subset of an trained ensemble to minimize memory consumption and maximize accuracy. Currently, four types of pruning algorithms are implemented:

- `RandomPruningClassifier`: Selects a random subset of classifiers. This is mainly used as a reference.
- `GreedyPruningClassifier`: Computes scores for each classifier and picks the top K classifiers among all scores. This is sometimes also referred to as ordering based pruning. These methods usually yield good results in a decent runtime.
- `MIQPPruningClassifier`: Constructs a mixed-integer quadratic problem and optimizes this to compute the best sub ensemble. This usually yields a slightly better overall performance, but can have a much higher runtime for larger ensembles.
- `ProxPruningClassifier`: This pruning method performs proximal gradient descent on the ensembles weights. It is much faster compared to `MIQPPruningClassifier` with similar results. However, there is only a limited support for scoring functions at this moment (see below).

For details on each method please have a look at the documentation. (todo)

# Some notes on the MIQPPruningClassifier 

Installing this package should already give you all necessary dependencies. If something is missing there is a conda environment for development. Try building and using it via

    conda env create -f environment.yml
    conda activate pypruning

For implementing the `MIQPPruningClassifier` we use `cvxpy` which does _not_ come with a MIQP solver. If you want to use this algorithm you have to manually install a solver, e.g.

    pip install cvopt

for a free solver or if you want to use a commercial solver and use Anaconda you can also install gurobi (with a free license)

    conda install -c gurobi gurobi

For more information on setting the solver for `MIQPPruningClassifier` have a look [here](https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options).

# Pruning an ensemble

A complete example might look like this. See below for more details and `run/tests.py` for a complete example:

```Python

data, target = load_digits(return_X_y = True)

XTP, Xtest, ytp, ytest = train_test_split(data, target, test_size=0.25, random_state=42)
Xtrain, Xprune, ytrain, yprune = train_test_split(XTP, ytp, test_size=0.25, random_state=42)

n_base = 128
n_prune = 8
model = RandomForestClassifier(n_estimators=n_base)
model.fit(XTP, ytp)
pred = model.predict(Xtest)

print("Accuracy of RF trained on XTrain + XPrune with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

model = RandomForestClassifier(n_estimators=n_base)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)

print("Accuracy of RF trained on XTrain only with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

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
```

Each pruning method has the following common interface:

```Python
def __init__(self, n_estimators = 5, base_estimator = None, n_jobs = 8):
    self.n_estimators = n_estimators
    self.base_estimator = base_estimator
    self.n_jobs = n_jobs
```

where 
- `n_estimators` is the size of the _pruned_ ensemble
- `base_estimator` is an object to train a new ensemble (see below) 
- `n_jobs` is the number of jobs used for computing all the scores. Note that `n_jobs` currently does not have any impact on the number of cores used by a the MIQP solver. 

To prune a set of estimators just call
```Python
def prune(self, X, y, estimators):
```

where 

- `X` are the pruning examples, 
- `y` are the corresponding pruning targets 
- `estimators` is the list of estimators to be pruned. 

We assume that each estimator in `estimators` has the following functions / fields: 

- `predict(X)`: Returns the class predictions for each example in X. Result should be `(X.shape[0], )`
- `predict_proba(X)`: Returns the class probabilities for each example in X. Result should be `(X.shape[0], n_classes_)` where `n_classes_` is the number of classes the classifier was trained on.
- `n_classes_`: Each classifier should have a field `n_classes_` which stores the number of classes the classifier was trained on

Moreover, each classifier should support `copy.deepcopy()`. If you want to directly train and prune a classifier you can supply a `base_estimator` in the constructor and call `fit(X,y)` on the pruning classifier. In this case, it is assumed that the `base_estimator` also offers a `fit(X,y)` function:

```Python
def fit(self, X, y):
    self.n_classes_ = len(set(y))
    model = self.base_estimator.fit(X,y)
    self.prune_(X, model.estimators_)

    return self
```
Please note however, that we currently only support scikit-learns `RandomForestClassifier` and `ExtraTreesClassifier` for this mode.


# Reproducing results from literature

There is a decent amount of pruning methods available in literature which mostly differs by the scoring functions used to score the performance of sub-ensembles. The `GreedyPruningClassifier` and `MIQPPruningClassifier` support the use of two pruning metrics:

- `single_metric`: Computes a score for each classifier
- `pairwise_metric`: Computes a score for each pair of classifiers 
- `l_reg`: Models the trade-off between both scoring functions where `l_reg = 0` only uses `single_metric` and `l_reg = 1` only uses the `pairwise_metric`

A list of available metrics can be found in `Metrics.py` with some comments from which paper they are taken. For convenience you can also "query" well-known methods form literature via `create_pruner(optimizer = "Greedy", paper = "margineantu1997", **kwargs)` where

- `optimizer = {{Greedy, MIQP}}`: The optimizer to be used
- `paper = {{"margineantu1997", "lazarevic2001", "lu2010", "guo2018", "cavalcanti2016", "zhang2006"}}`: The pruning method from the named paper
- `**kwargs`: Any additional parameters such as `n_estimators` etc. passed to the corresponding pruner.

For a complete example also have a look at`run/tests.py`.

# Extending PyPruning

If you want to implement your own pruning method then there are two ways:

## Implementing a custom metric

You can implement your own metric as a function that should be _minimized_. It should follow the following interface

```Python
def my_metric(iproba, jproba, target):
```

where 

- `iproba` is the predicted probability of the i-th estimator (`estimator[i].predict_proba(Xprune)`) evaluated on the pruning set 
- `jproba` is the predicted probability of the j-th estimator (`estimator[j].predict_proba(Xprune)`) evaluated on the pruning set 
- `target` are the corresponding targets in the pruning set. 

If you do not compute pairwise statistics just ignore the second argument `jproba`. An example could be:

```Python
def my_metric(iproba, jproba, target):
    return (iproba.argmax(axis=1) != target).mean()
```


## Implementing a custom pruner

You can implement your own pruner as a well. In this case you just have to implement the `PruningClassifier` class. To do so, you just need to implement the `prune_(self, proba, target)` function which receives a list of all predictions of all classifiers as well as the corresponding targets. The function is supposed to return a list of indices corresponding to the chosen estimators as well as the corresponding weights. If you need access to the estimators as well (and not just their predictions) you can access `self.estimators_` which already contains a copy of each classier. For more details have a look at the `PruningClassifier.py` interface. An example implementation could be:


```Python
class RandomPruningClassifier(PruningClassifier):

    def __init__(self, n_estimators = 5, base_estimator = None, n_jobs = 8):
        
        super().__init__(n_estimators, base_estimator, n_jobs)

    def prune_(self, proba, target):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        else:
            return np.random.choice(range(0, n_received),size=self.n_estimators), [1.0 / self.n_estimators for _ in range(self.n_estimators)]
```

# Acknowledgements 

Special thanks goes to Henri Petuker (henri.petuker@tu-dortmund.de) who implemented the initial version of many of these algorithms during his bachelor thesis.