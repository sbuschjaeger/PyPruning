from functools import partial

from sklearn.metrics import pairwise 

from .MIQPPruningClassifier import MIQPPruningClassifier, combined, combined_error
from .GreedyPruningClassifier import GreedyPruningClassifier, reduced_error, complementariness, margin_distance, drep
from .RankPruningClassifier import RankPruningClassifier, individual_margin_diversity, individual_contribution, individual_error, individual_kappa_statistic, reference_vector, error_ambiguity
from .ClusterPruningClassifier import ClusterPruningClassifier, largest_mean_distance, cluster_accuracy, centroid_selector, agglomerative, kmeans

def create_pruner(method = "reduced_error", **kwargs):
    """ This helper function creates a pruner with the given name.  

    Parameters
    ----------
    method : string, default is "reduced_error"
        The name of the method for which a pruner should be created. Currently supported are: ``{"individual_margin_diversity", "individual_contribution", "individual_error", "individual_kappa_statistic", "reduced_error", "complementariness", "drep", "margin_distance", "combined", "reference_vector", "combined_error", "error_ambiguity", "largest_mean_distance", "cluster_accuracy", "cluster_centroids"}``
    kwargs : 
        All additional kwargs parameters are directly passed to the creating method. Use this to e.g. set `n_estimators` etc.
    """
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
        return GreedyPruningClassifier(metric=reduced_error,  **kwargs)
    elif method == "complementariness":
        return GreedyPruningClassifier(metric=complementariness,  **kwargs)
    elif method == "drep":
        return GreedyPruningClassifier(metric=drep,  **kwargs)
    elif method == "margin_distance":
        return GreedyPruningClassifier(metric=margin_distance,  **kwargs)
    elif method == "combined":
        return MIQPPruningClassifier(single_metric=None, pairwise_metric=combined, alpha = 1.0)
    elif method == "reference_vector":
        return RankPruningClassifier(metric=reference_vector,  **kwargs)
    elif method == "combined_error":
        return MIQPPruningClassifier(single_metric=None, pairwise_metric=combined_error, alpha = 1.0)
    elif method == "error_ambiguity":
        return RankPruningClassifier(metric=error_ambiguity,  **kwargs)
    elif method == "largest_mean_distance":
        return ClusterPruningClassifier(cluster_estimators=agglomerative, select_estimators=largest_mean_distance, cluster_mode = "accuracy", **kwargs)
    elif method == "cluster_accuracy":
        return ClusterPruningClassifier(cluster_estimators=kmeans, select_estimators=cluster_accuracy, **kwargs)
    elif method == "cluster_centroids":
        # Original publication uses simulated annealing. We stick to kmeans  
        return ClusterPruningClassifier(cluster_estimators=kmeans, select_estimators=centroid_selector, **kwargs)
    # elif method == "disagreement":
    #     return MIQPPruningClassifier(single_metric=None, pairwise_metric=disagreement, alpha = 1.0)

    return None
    