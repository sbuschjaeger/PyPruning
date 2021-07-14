from functools import partial
import numpy as np
from sklearn import cluster
from sklearn.metrics import roc_auc_score

from joblib import Parallel,delayed
from sklearn.utils import axis0_safe_slice
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from .PruningClassifier import PruningClassifier

def kmeans(X, n_estimators, **kwargs):
    """
    Perform kmeans clustering on given data `X`. The original publication (see below) considers the predictions of each estimator which can be achieved by setting `cluster_mode = "predictions"` in the ClusterPruningClassifier. Second, the original publication discusses binary classification problems. In this multi-class implementation the proba/predictions for each class are flattened before clustering.

    Reference:
        Lazarevic, A., & Obradovic, Z. (2001). Effective pruning of neural network classifier ensembles. Proceedings of the International Joint Conference on Neural Networks, 2(January), 796–801. https://doi.org/10.1109/ijcnn.2001.939461pdf
    """
    kmeans = KMeans(n_clusters = n_estimators, **kwargs)
    assignments = kmeans.fit_predict(X)
    return assignments

def agglomerative(X, n_estimators, **kwargs):
    """
    Perform agglomerative clustering on the given data `X`. The original publication (see below) considers the accuracy / error of each estimator which can be achieved by setting `cluster_mode = "accuracy"` in the ClusterPruningClassifier. 

    Reference:
        Giacinto, G., Roli, F., & Fumera, G. (n.d.). Design of effective multiple classifier systems by clustering of classifiers. Proceedings 15th International Conference on Pattern Recognition. ICPR-2000. doi:10.1109/icpr.2000.906039 
    """
    agg = AgglomerativeClustering(n_clusters = n_estimators, distance_threshold = None, **kwargs)
    assignments = agg.fit_predict(X)
    return assignments

def centroid_selector(X, assignments, target):
    """
    Returns the centroid of each cluster. Bakker and Heske propose this approach, although there are subtle differences. Originally they propose to use annealing via an EM algorithm, whereas we use kmeans / agglomerative clustering. 

    Reference
        Bakker, Bart, and Tom Heskes. "Clustering ensembles of neural network models." Neural networks 16.2 (2003): 261-269.
    """
    # TODO centroids are already known if kmeans was used, but agglomerative does not know / care about centroids
    clf = NearestCentroid()
    clf.fit(X, assignments)
    centroids = clf.centroids_

    centroid_idx,_ = pairwise_distances_argmin_min(centroids, X)

    return centroid_idx

def cluster_accuracy(X, assignments, target, n_classes = None):
    """
    Select the most accurate model from each cluster. Lazarevic and Obradovic propose this approach although there are subtle differences. In the original paper they remove the least-accurate classifier as long as the performance of the sub-ensemble does not decrease. In this implementation we simply select the best / most accurate classifier from each cluster.

    Reference
        Lazarevic, A., & Obradovic, Z. (2001). Effective pruning of neural network classifier ensembles. Proceedings of the International Joint Conference on Neural Networks, 2(January), 796–801. https://doi.org/10.1109/ijcnn.2001.939461
    """
    idx_per_centroid = {}
    for i, a in enumerate(assignments):
        if a not in idx_per_centroid:
            idx_per_centroid[a] = []
        idx_per_centroid[a].append(i)
    
    selected_idx = []

    if n_classes is None:
        n_classes = len(set(target))
    
    preds = X.reshape(X.shape[0], int(X.shape[1]/n_classes), n_classes)
    for c, idx in idx_per_centroid.items():
        accs = [ (preds[i,:].argmax(axis=1) == target).mean() for i in idx ]
        selected_idx.append(np.argmax(accs))

    return selected_idx

def largest_mean_distance(X, assignments, target, metric = 'euclidean', n_jobs = None):
    """
    Select the most distant classifier to all other clusters. 

    Reference:
        Giacinto, G., Roli, F., & Fumera, G. (n.d.). Design of effective multiple classifier systems by clustering of classifiers. Proceedings 15th International Conference on Pattern Recognition. ICPR-2000. doi:10.1109/icpr.2000.906039 
    """
    idx_per_centroid = {}
    for i, a in enumerate(assignments):
        if a not in idx_per_centroid:
            idx_per_centroid[a] = []
        idx_per_centroid[a].append(i)
    
    all_distances = pairwise_distances(X, metric = metric, n_jobs = n_jobs)
    
    selected_indxed = []
    for c, idx in idx_per_centroid.items():
        mask = np.ones(all_distances.shape[0], np.bool)
        mask[idx] = 0
        distances = []
        for i in idx:
            avg_distance = np.mean(all_distances[i,:][mask])
            distances.append(avg_distance)
        imax = np.argmax(distances)
        selected_indxed.append(idx[imax])

    return selected_indxed

def random_selector(X, assignments, target):
    """
    Randomly select a classifier from each cluster.
    """
    idx_per_centroid = {}
    for i, a in enumerate(assignments):
        if a not in idx_per_centroid:
            idx_per_centroid[a] = []
        idx_per_centroid[a].append(i)

    selected_idx = []
    for c, idx in idx_per_centroid.items():
        selected_idx.append(np.random.choice(idx))

    return selected_idx

class ClusterPruningClassifier(PruningClassifier):
    """ Clustering-based pruning. 
    
    Clustering-based methods follow a two-step procedure. In first step, they cluster the estimators in the ensemble according to some clustering algorithm. Then, in the second, a representative form each cluster is selected to form the pruned ensemble. More formally, clustering-based pruning uses the following optimization problem:

    In this implementation, you must provide two functions

    - `cluster_estimators`: A function which clusters the estimators given their representation X (see `cluster_mode` for details) and return the cluster assignment for each estimator. An example of kmeans clustering would be:

    .. code-block:: python

        def kmeans(X, n_estimators, **kwargs):
            kmeans = KMeans(n_clusters = n_estimators, **kwargs)
            assignments = kmeans.fit_predict(X)
            return assignments 
    
    - `select_estimators`: A function which selects the estimators from the clustering and returns the selected indices. An example of which selects the centroids would be:
    
    .. code-block:: python

        def centroid_selector(x, assignments, target, **kwargs):

            clf = NearestCentroid()
            clf.fit(x, assignments)
            centroids = clf.centroids_

            centroid_idx,_ = pairwise_distances_argmin_min(centroids, x)

            return centroid_idx 

    If you want to pass additional parameter to `cluster_estimators` or `select_estimators` you can do so via the `cluster_options` and `selector_options` respectively. These parameters are passed via **kwargs to the functions so please make sure that they are either `None` or valid Python dictionaries. 

    Attributes
    ----------
    n_estimators : int, default is 5
        The number of estimators which should be selected.
    cluster_estimators : function, default is kmeans
        A function that clusters the classifier.
    select_estimators : function, default is random_selector
        A function that selects representatives from each cluster
    cluster_mode: str, default is probabilities"
        The representation of each estimator used for clustering. Must be one of {"probabilities", "predictions", "accuracy"}:
            - "probabilities": Uses the raw probability output of each estimator for clustering. For multi-class problems the vector is "flattened" to a N * C vector where N is the number pf data points in the pruning set and C is the number of classes  
            - "predictions": Same as "probabilities", but uses the predictions instead of the probabilities.
            - "accuracy": Computes the accuracy of each estimator on each datapoint and uses the corresponding vector for clustering. 
    cluster_options : dict, default is None
        Additional options passed to `cluster_estimators`
    selector_options : dict, default is None
        Additional options passed to `select_estimators`
    """
    def __init__(self, n_estimators = 5, cluster_estimators = kmeans, select_estimators = random_selector, cluster_mode = "probabilities", cluster_options = None, selector_options = None):
        super().__init__()

        assert cluster_mode in ["probabilities", "predictions", "accuracy"]

        self.n_estimators = n_estimators
        self.cluster_estimators = cluster_estimators
        self.select_estimators = select_estimators
        self.cluster_mode = cluster_mode
        if cluster_options is None:
            self.cluster_options = {}
        else:
            self.cluster_options = cluster_options
        
        if selector_options is None:
            self.selector_options = {}
        else:
            self.selector_options = selector_options

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]

        if self.cluster_mode == "predictions":
            # See https://stackoverflow.com/questions/44143438/numpy-indexing-set-1-to-max-value-and-zeros-to-all-others
            tmp = np.zeros_like(proba)
            d1 = np.arange(proba.shape[0])[:,None]
            d2 = np.arange(proba.shape[1])
            idx = proba.argmax(axis=2)
            tmp[d1,d2,idx] = 1
            fc_proba = tmp
            fc_proba = fc_proba.reshape(fc_proba.shape[0],fc_proba.shape[1]*fc_proba.shape[2])
        elif self.cluster_mode == "accuracy":
            preds = proba.argmax(axis=2)
            fc_proba = (preds == target).astype(int)
        else:
            fc_proba = proba.reshape(proba.shape[0],proba.shape[1]*proba.shape[2])

        assignments = self.cluster_estimators(fc_proba, self.n_estimators, **self.cluster_options)
        
        if self.select_estimators == cluster_accuracy and "n_classes" not in self.selector_options:
            self.selector_options["n_classes"] = proba.shape[2]

        selected_models = self.select_estimators(fc_proba, assignments, target, **self.selector_options)

        return selected_models, [1.0 / len(selected_models) for _ in selected_models]