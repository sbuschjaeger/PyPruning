from functools import partial

from sklearn.metrics import pairwise 

from .MIQPPruningClassifier import MIQPPruningClassifier, combined, combined_error
from .GreedyPruningClassifier import GreedyPruningClassifier, error, complementariness, margin_distance, drep
from .RankPruningClassifier import RankPruningClassifier, individual_margin_diversity, individual_contribution, individual_error, individual_kappa_statistic

#from .Metrics import Metrics

def create_pruner(method = "reduced_error", **kwargs):

    if method == "individual_margin_diversity":
        metric = partial(individual_margin_diversity, alpha = 0.2)
        return RankPruningClassifier(metric=metric,  **kwargs)
    elif method == "individual_contribution":
        return RankPruningClassifier(metric=individual_contribution,  **kwargs)
    elif method == "individual_error":
        return RankPruningClassifier(metric=individual_error,  **kwargs)
    elif method == "individual_kappa_statistic":
        return RankPruningClassifier(metric=individual_kappa_statistic,  **kwargs)
    elif method == "reduced_error":
        return GreedyPruningClassifier(metric=error,  **kwargs)
    elif method == "complementariness":
        return GreedyPruningClassifier(metric=complementariness,  **kwargs)
    elif method == "drep":
        return GreedyPruningClassifier(metric=drep,  **kwargs)
    elif method == "margin_distance":
        return GreedyPruningClassifier(metric=margin_distance,  **kwargs)
    elif method == "combined":
        return MIQPPruningClassifier(single_metric=None, pairwise_metric=combined, alpha = 1.0)
    elif method == "combined_error":
        return MIQPPruningClassifier(single_metric=None, pairwise_metric=combined_error, alpha = 1)
    # elif method == "disagreement":
    #     return MIQPPruningClassifier(single_metric=None, pairwise_metric=disagreement, alpha = 1.0)

    return None
    