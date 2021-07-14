<img src="docs/pruning-logo.png" width="200" height="200"> <h1>PyPruning</h1>

[![Building docs](https://github.com/sbuschjaeger/PyPruning/actions/workflows/docs.yml/badge.svg)](https://github.com/sbuschjaeger/PyPruning/actions/workflows/docs.yml)
[![tests](https://github.com/sbuschjaeger/PyPruning/actions/workflows/tests.yml/badge.svg)](https://github.com/sbuschjaeger/PyPruning/actions/workflows/tests.yml)

This package provides implementations for some common ensemble pruning algorithms. Pruning algorithms aim to select the best subset of an trained ensemble to minimize memory consumption and maximize accuracy. Currently, six types of pruning algorithms are implemented:

- `RandomPruningClassifier`: Selects a random subset of classifiers. This is mainly used as a baseline.
- `RankPruningClassifier`: Rank each classifier according to a given metric and then select the best K classifier.
- `ClusterPruningClassifier`: Cluster the classifiers according to a clustering method and then select a representative from each cluster to from the sub-ensemble.
- `GreedyPruningClassifier`: Proceeds in rounds and selects the best classifier in each round given the already selected sub-ensemble. 
- `MIQPPruningClassifier`: Constructs a mixed-integer quadratic problem and optimizes this to compute the best sub ensemble. 
- `ProxPruningClassifier`: Minimize a (regularized) loss function via (stochastic) proximal gradient descent over the ensembles weights. 

For more details please have a look at the [documentation](https://sbuschjaeger.github.io/PyPruning/html/index.html). 

# How to install

You can install this package via directly via pip from git 

    pip install git+https://github.com/sbuschjaeger/PyPruning.git


If you have trouble with dependencies you can try setting up a conda environment which I use for development:

```bash
   git clone git@github.com:sbuschjaeger/PyPruning.git
   cd PyPruning
   conda env create -f environment.yml
   conda activate pypruning
```

### Some notes on the MIQPPruningClassifier

For implementing the `MIQPPruningClassifier` we use [cvxpy](https://www.cvxpy.org/) which does **not** come with a MIQP solver. If you want to use this algorithm you have to manually install a solver, e.g.

    pip install cvopt

for a free solver or if you want to use a commercial solver and use Anaconda you can also install gurobi (with a free license)

    conda install -c gurobi gurobi

For more information on setting the solver for `MIQPPruningClassifier` have a look [here](https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options).

# Pruning an ensemble

A complete example might look like this. For more details please have a look at the [documentation](https://sbuschjaeger.github.io/PyPruning/html/index.html) or at the files under `run/tests.py`:

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

# Reproducing results from literature

There is a decent amount of pruning methods available in literature which mostly differ by the scoring function as well as the optimizer used to score the performance of sub-ensembles. All pruning methods accept different forms of metrics and/or algorithm to determine the specific behavior. Please have a look at the documentation to see which metrics are already implemented and how to add your own. Currently supported are

- `individual_margin_diversity` (Guo, H., Liu, H., Li, R., Wu, C., Guo, Y., & Xu, M. (2018). Margin & diversity based ordering ensemble pruning. Neurocomputing, 275, 237–246. https://doi.org/10.1016/j.neucom.2017.06.052)
- `individual_contribution` (Lu, Z., Wu, X., Zhu, X., & Bongard, J. (2010). Ensemble pruning via individual contribution ordering. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 871–880. https://doi.org/10.1145/1835804.1835914)
- `individual_error` (Jiang, Z., Liu, H., Fu, B., & Wu, Z. (2017). Generalized ambiguity decompositions for classification with applications in active learning and unsupervised ensemble pruning. 31st AAAI Conference on Artificial Intelligence, AAAI 2017, 2073–2079.)
- `error_ambiguity` (Jiang, Z., Liu, H., Fu, B., & Wu, Z. (2017). Generalized ambiguity decompositions for classification with applications in active learning and unsupervised ensemble pruning. 31st AAAI Conference on Artificial Intelligence, AAAI 2017, 2073–2079.)
- `individual_kappa_statistic` (Margineantu, D., & Dietterich, T. G. (1997). Pruning Adaptive Boosting. Proceedings of the Fourteenth International Conference on Machine Learning, 211–218. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.7017&rep=rep1&type=pdf)
- `reduced_error` (Margineantu, D., & Dietterich, T. G. (1997). Pruning Adaptive Boosting. Proceedings of the Fourteenth International Conference on Machine Learning, 211–218. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.7017&rep=rep1&type=pdf)
- `complementariness` (Martínez-Muñoz, G., & Suárez, A. (2004). Aggregation ordering in bagging. Proceedings of the IASTED International Conference. Applied Informatics, 258–263. Retrieved from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2035&rep=rep1&type=pdf)
- `drep` (Li, N., Yu, Y., & Zhou, Z.-H. (2012). Diversity Regularized Ensemble Pruning. In P. A. Flach, T. De Bie, & N. Cristianini (Eds.), Machine Learning and Knowledge Discovery in Databases (pp. 330–345). Berlin, Heidelberg: Springer Berlin Heidelberg. https://link.springer.com/content/pdf/10.1007%2F978-3-642-33460-3.pdf)
- `margin_distance` (Martínez-Muñoz, G., & Suárez, A. (2004). Aggregation ordering in bagging. Proceedings of the IASTED International Conference. Applied Informatics, 258–263. Retrieved from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2035&rep=rep1&type=pdf)
- `combined` (Cavalcanti, G. D. C., Oliveira, L. S., Moura, T. J. M., & Carvalho, G. V. (2016). Combining diversity measures for ensemble pruning. Pattern Recognition Letters, 74, 38–45. https://doi.org/10.1016/j.patrec.2016.01.029)
- `combined_error` (Zhang, Y., Burer, S., & Street, W. N. (2006). Ensemble Pruning Via Semi-definite Programming. Journal of Machine Learning Research, 7, 1315–1338. https://doi.org/10.1016/j.jasms.2006.06.007)
- `reference_vector` (Hernández-Lobato, D., Martínez-Muñoz, G., & Suárez, A. (2006). Pruning in Ordered Bagging Ensembles. International Conference on Machine Learning, 1266–1273. https://doi.org/10.1109/ijcnn.2006.246837)
- `largest_mean_distance` (Giacinto, G., Roli, F., & Fumera, G. (n.d.). Design of effective multiple classifier systems by clustering of classifiers. Proceedings 15th International Conference on Pattern Recognition. ICPR-2000. doi:10.1109/icpr.2000.906039)
- `cluster_accuracy` (Lazarevic, A., & Obradovic, Z. (2001). Effective pruning of neural network classifier ensembles. Proceedings of the International Joint Conference on Neural Networks, 2(January), 796–801. https://doi.org/10.1109/ijcnn.2001.939461)
- `cluster_centroids` (Bakker, Bart, and Tom Heskes. "Clustering ensembles of neural network models." Neural networks 16.2 (2003): 261-269.)


# Acknowledgements 

The software is written and maintained by [Sebastian Buschjäger](https://sbuschjaeger.github.io/) as part of his work at the [Chair for Artificial Intelligence](https://www-ai.cs.tu-dortmund.de) at the TU Dortmund University and the [Collaborative Research Center 876](https://sfb876.tu-dortmund.de). If you have any question feel free to contact me under sebastian.buschjaeger@tu-dortmund.de 

Special thanks goes to [Henri Petuker](mailto:henri.petuker@tu-dortmund.de) who provided parts of this implementation during his bachelor thesis and and [David Clemens](mailto:david.clemens@tu-dortmund.de) who made the logo. 
