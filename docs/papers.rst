Reproducing results from literature
-----------------------------------

There is a decent amount of pruning methods available in literature which mostly differs by the scoring functions used to score the performance of sub-ensembles. All pruning methods accept different forms of metrics and/or algorithm to determine the specific behavior. Please have a look at the specific class files to see which metrics are already implemented. If you cannot find you metric of choice feel free to implement it. Currently supported are

- :func:`~PyPruning.RankPruningClassifier.individual_margin_diversity` (Guo, H., Liu, H., Li, R., Wu, C., Guo, Y., & Xu, M. (2018). Margin & diversity based ordering ensemble pruning. Neurocomputing, 275, 237–246. https://doi.org/10.1016/j.neucom.2017.06.052)
- :func:`~PyPruning.RankPruningClassifier.individual_contribution` (Lu, Z., Wu, X., Zhu, X., & Bongard, J. (2010). Ensemble pruning via individual contribution ordering. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 871–880. https://doi.org/10.1145/1835804.1835914)
- :func:`~PyPruning.RankPruningClassifier.individual_error`
- :func:`~PyPruning.RankPruningClassifier.individual_kappa_statistic` (Margineantu, D., & Dietterich, T. G. (1997). Pruning Adaptive Boosting. Proceedings of the Fourteenth International Conference on Machine Learning, 211–218. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.7017&rep=rep1&type=pdf)
- :func:`~PyPruning.GreedyPruningClassifier.reduced_error` (Margineantu, D., & Dietterich, T. G. (1997). Pruning Adaptive Boosting. Proceedings of the Fourteenth International Conference on Machine Learning, 211–218. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.7017&rep=rep1&type=pdf)
- :func:`~PyPruning.GreedyPruningClassifier.complementariness` (Martínez-Muñoz, G., & Suárez, A. (2004). Aggregation ordering in bagging. Proceedings of the IASTED International Conference. Applied Informatics, 258–263. Retrieved from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2035&rep=rep1&type=pdf)
- :func:`~PyPruning.GreedyPruningClassifier.drep` (Li, N., Yu, Y., & Zhou, Z.-H. (2012). Diversity Regularized Ensemble Pruning. In P. A. Flach, T. De Bie, & N. Cristianini (Eds.), Machine Learning and Knowledge Discovery in Databases (pp. 330–345). Berlin, Heidelberg: Springer Berlin Heidelberg. https://link.springer.com/content/pdf/10.1007%2F978-3-642-33460-3.pdf)
- :func:`~PyPruning.GreedyPruningClassifier.margin_distance` (Martínez-Muñoz, G., & Suárez, A. (2004). Aggregation ordering in bagging. Proceedings of the IASTED International Conference. Applied Informatics, 258–263. Retrieved from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2035&rep=rep1&type=pdf)
- :func:`~PyPruning.MIQPPruningClassifier.combined` (Cavalcanti, G. D. C., Oliveira, L. S., Moura, T. J. M., & Carvalho, G. V. (2016). Combining diversity measures for ensemble pruning. Pattern Recognition Letters, 74, 38–45. https://doi.org/10.1016/j.patrec.2016.01.029)
- :func:`~PyPruning.MIQPPruningClassifier.combined_error` (Zhang, Y., Burer, S., & Street, W. N. (2006). Ensemble Pruning Via Semi-definite Programming. Journal of Machine Learning Research, 7, 1315–1338. https://doi.org/10.1016/j.jasms.2006.06.007)
- :func:`~PyPruning.RankPruningClassifier.reference_vector` (Hernández-Lobato, D., Martínez-Muñoz, G., & Suárez, A. (2006). Pruning in Ordered Bagging Ensembles. International Conference on Machine Learning, 1266–1273. https://doi.org/10.1109/ijcnn.2006.246837)

For convenience you can access these pruning methods via the `create_pruner` function:

.. code-block:: python

   from PyPruning.Papers import create_pruner
   md_pruner = create_pruner("margin_distance", n_estimators=10)

.. automodule:: PyPruning.Papers
   :members:
   :undoc-members:
   :show-inheritance: