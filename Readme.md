# PyPruning

Ensemble Algorithms implemented in Python. You can install this package via pip

    pip install git+https://github.com/sbuschjaeger/PyPruning.git

This package provides implementations for some common ensemble pruning algorithms. Pruning algorithms aim to select the best subset of an trained ensemble to minimize memory consumption and maximize accuracy. Currently, four types of pruning algorithms are implemented:

- `RandomPruningClassifier`: Selects a random subset of classifiers. This is mainly used as a reference.
- `GreedyPruningClassifier`: Proceeds in rounds and selects the best classifier in each round given the already selected sub-ensemble. These methods usually yield good results in a decent runtime.
- `MIQPPruningClassifier`: Constructs a mixed-integer quadratic problem and optimizes this to compute the best sub ensemble. This usually yields a slightly better overall performance, but can have a much higher runtime for larger ensembles. Additionally, for larger ensembles numerical instabilities sometimes trouble the solver which then  might not find a valid solution.
- `ProxPruningClassifier`: This pruning method performs proximal gradient descent on the ensembles weights. It is much faster compared to `MIQPPruningClassifier` with similar results. We have shown that this method statistically beats the other methods. In addition, this method allows you to regularize the selected ensemble to e.g. focus on smaller trees. 

For details on each method please have a look at the documentation. 

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

To prune a set of estimators just call
```Python
def prune(self, X, y, estimators):
```
of one of the pruning classes, where 

- `X` are the pruning examples, 
- `y` are the corresponding pruning targets 
- `estimators` is the list of estimators to be pruned. 
- `classes` a list of classes this classifier was trained on which corresponding to the order of `predict_proba`. If this is `None` we try to infer this from the base estimators
- `n_classes` the total number of classes. If this is `None` we try to infer this from the base estimators

We assume that each estimator in `estimators` has the following functions / fields: 

- `predict(X)`: Returns the class predictions for each example in X. Result should be `(X.shape[0], )`
- `predict_proba(X)`: Returns the class probabilities for each example in X. Result should be `(X.shape[0], n_classes_)` where `n_classes_` is the number of classes the classifier was trained on.

Moreover, each classifier should support `copy.deepcopy()`. 


# Reproducing results from literature

There is a decent amount of pruning methods available in literature which mostly differs by the scoring functions used to score the performance of sub-ensembles. The `GreedyPruningClassifier`, `MIQPPruningClassifier` and the `RankPruningClassifier` all support the use of different metrics. Please have a look at the specific class files to see which metrics are already implemented. If you cannot find you metric of choice feel free to implement it (details below). Last, `Papers.py` also contains a helper function `create_pruner` which lets you select a pruning method based on common names in literature. Currently supported are

- `individual_margin_diversity` (Guo, H., Liu, H., Li, R., Wu, C., Guo, Y., & Xu, M. (2018). Margin & diversity based ordering ensemble pruning. Neurocomputing, 275, 237–246. https://doi.org/10.1016/j.neucom.2017.06.052)
- `individual_contribution` (Lu, Z., Wu, X., Zhu, X., & Bongard, J. (2010). Ensemble pruning via individual contribution ordering. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 871–880. https://doi.org/10.1145/1835804.1835914)
- `individual_error`
- `individual_kappa_statistic` (Margineantu, D., & Dietterich, T. G. (1997). Pruning Adaptive Boosting. Proceedings of the Fourteenth International Conference on Machine Learning, 211–218. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.7017&rep=rep1&type=pdf)
- `reduced_error` (Margineantu, D., & Dietterich, T. G. (1997). Pruning Adaptive Boosting. Proceedings of the Fourteenth International Conference on Machine Learning, 211–218. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.7017&rep=rep1&type=pdf)
- `complementariness` (Martínez-Muñoz, G., & Suárez, A. (2004). Aggregation ordering in bagging. Proceedings of the IASTED International Conference. Applied Informatics, 258–263. Retrieved from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2035&rep=rep1&type=pdf)
- `drep` (Li, N., Yu, Y., & Zhou, Z.-H. (2012). Diversity Regularized Ensemble Pruning. In P. A. Flach, T. De Bie, & N. Cristianini (Eds.), Machine Learning and Knowledge Discovery in Databases (pp. 330–345). Berlin, Heidelberg: Springer Berlin Heidelberg. https://link.springer.com/content/pdf/10.1007%2F978-3-642-33460-3.pdf)
- `margin_distance` (Martínez-Muñoz, G., & Suárez, A. (2004). Aggregation ordering in bagging. Proceedings of the IASTED International Conference. Applied Informatics, 258–263. Retrieved from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2035&rep=rep1&type=pdf)
- `combined` (Cavalcanti, G. D. C., Oliveira, L. S., Moura, T. J. M., & Carvalho, G. V. (2016). Combining diversity measures for ensemble pruning. Pattern Recognition Letters, 74, 38–45. https://doi.org/10.1016/j.patrec.2016.01.029)
- `combined_error` (Zhang, Y., Burer, S., & Street, W. N. (2006). Ensemble Pruning Via Semi-definite Programming. Journal of Machine Learning Research, 7, 1315–1338. https://doi.org/10.1016/j.jasms.2006.06.007)
- `reference_vector` (Hernández-Lobato, D., Martínez-Muñoz, G., & Suárez, A. (2006). Pruning in Ordered Bagging Ensembles. International Conference on Machine Learning, 1266–1273. https://doi.org/10.1109/ijcnn.2006.246837)

For a complete example also have a look at `run/tests.py`.

# Extending PyPruning

If you want to implement your own pruning method then there are two ways:

## Implementing a custom metric

You can implement your own metric for `GreedyPruningClassifier`, `MIQPPruningClassifier` or a `RankPruningClassifier` you simply have to implement a python function that should be _minimized_. The specific interface required by each method slightly differs so please check out the specific documentation for the method of your choice. In all cases, each method expects functions with at-least three parameters

    - `i` (int): The classifier which should be rated
    - `ensemble_proba` (A (M, N, C) matrix ): All N predictions of all M classifier in the entire ensemble for all C classes
    - `target` (list / array): A list / array of class targets.

Note that `ensemble_proba` contains all class probabilities predicted by all members in the ensemble. So in order to get individual class predictions for the i-th classifier you can access it via `ensemble_proba[i,:,:]`. A complete example which simply computes the error of each method would be

```Python
def individual_error(i, ensemble_proba, target):
    iproba = ensemble_proba[i,:,:]
    return (iproba.argmax(axis=1) != target).mean()
```

## Implementing a custom pruner

You can implement your own pruner as a well. In this case you just have to implement the `PruningClassifier` class. To do so, you just need to implement the `prune_(self, proba, target)` function which receives a list of all predictions of all classifiers as well as the corresponding data and targets. The function is supposed to return a list of indices corresponding to the chosen estimators as well as the corresponding weights. If you need access to the estimators as well (and not just their predictions) you can access `self.estimators_` which already contains a copy of each classier. For more details have a look at the `PruningClassifier.py` interface. An example implementation could be:


```Python
class RandomPruningClassifier(PruningClassifier):

    def __init__(self):
        super().__init__()

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        else:
            return np.random.choice(range(0, n_received),size=self.n_estimators), [1.0 / self.n_estimators for _ in range(self.n_estimators)]
```
For more details check out the abstract class `PruningClassifier`

# Acknowledgements 

Special thanks goes to Henri Petuker (henri.petuker@tu-dortmund.de) who implemented the initial version of many of these algorithms during his bachelor thesis.