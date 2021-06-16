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

def kmeans(proba, n_estimators, **kwargs):
    proba = proba.reshape(proba.shape[0],proba.shape[1]*proba.shape[2])
    kmeans = KMeans(n_clusters = n_estimators, **kwargs)
    assignments = kmeans.fit_predict(proba)
    return assignments

def agglomerative(proba, n_estimators, **kwargs):
    proba = proba.reshape(proba.shape[0],proba.shape[1]*proba.shape[2])
    agg = AgglomerativeClustering(n_clusters = n_estimators, distance_threshold = None, **kwargs)
    assignments = agg.fit_predict(proba)
    return assignments

def centroid_selector(proba, assignments, target):
    proba = proba.reshape(proba.shape[0],proba.shape[1]*proba.shape[2])

    # First compute the centroid given the assignments.
    # TODO centroids are already known if kmeans was used, but agglomerative does not know / care about centroids
    clf = NearestCentroid()
    clf.fit(proba, assignments)
    centroids = clf.centroids_

    centroid_idx,_ = pairwise_distances_argmin_min(centroids, proba)

    return centroid_idx

def accuracy(proba, assignments, target):
    idx_per_centroid = {}
    for i, a in enumerate(assignments):
        if a not in idx_per_centroid:
            idx_per_centroid[a] = []
        idx_per_centroid[a].append(i)
    
    selected_idx = []
    
    for c, idx in idx_per_centroid.items():
        accs = [ (proba[i,:].argmax(axis=1) == target).mean() for i in idx ]
        selected_idx.append(np.argmax(accs))

    return selected_idx

def largest_mean_distance(proba, assignments, target, metric = 'euclidean', n_jobs = None):
    idx_per_centroid = {}
    for i, a in enumerate(assignments):
        if a not in idx_per_centroid:
            idx_per_centroid[a] = []
        idx_per_centroid[a].append(i)
    
    proba = proba.reshape(proba.shape[0],proba.shape[1]*proba.shape[2])
    all_distances = pairwise_distances(proba, metric = metric, n_jobs = n_jobs)
    
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

def random_selector(proba, assignments, target):
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
    ''' Clustering-based pruning. 
    
    Clustering-based methods follow a two-step procedure. In first step, they cluster the estimators in the ensemble according to some clustering algorithm. Then, in the second step some representatives form each cluster are selected. 

    $$
    \begin{align}
        &\arg\min_{w\in \{0,1\}^M} \sum_{(x,y) \in \mathcal S} d(\mu_{c(i)}(x) - h_i(x)) + w_i \ell(\mu_{c(i)}, h(x), y) \\ \nonumber
        &\text{~st~} c(i) = \arg\min \left \{\sum_{(x,y) \in \mathcal S} d(\mu_{j}(x) - h_i(x)) | j=1,\dots,M \right \} \\ \nonumber
        &\phantom{\text{~st~}} \lVert w \rVert_0 = K, \forall w_i,w_j \not= 0 \wedge i \not= j: c(i) \not= c(j)
    \end{align}
    $$
    where $\{\mu_1,\dots,\mu_K\} \subseteq H$ are the cluster centers and $d\colon \mathbb R^C \times \mathbb R^C \to \mathbb R$ is a distance metric.

    order the estimators in the ensemble according to their performance. In contrast to ranking-based pruning however they also consider the already selected sub-ensemble for selecting the next classifier. They start with the empty ensemble and then greedily select in each round that classifier which minimizes a loss function the most:

    $$
    \\arg\\min_{h} L\\left(\\frac{K-1}{K} f(x) + \\frac{1}{K} \\cdot h(x), y\\right)
    $$

    where f is the already selected ensemble with K-1 members and h is the newly selected member. In this sense, this selection re-order the classifiers in the ensemble and hence sometimes the name ordering-based pruning is used. In this implementation a loss function receives 4 parameters:

    - `i` (int): The classifier which should be rated
    - `ensemble_proba` (A (M, N, C) matrix ): All N predictions of all M classifier in the entire ensemble for all C classes
    - `selected_models` (list of ints): All models which are selected so far
    - `target` (list / array): A list / array of class targets.

    A simple loss function which minimizes the overall sub-ensembles error would be

    ```Python
        def error(i, ensemble_proba, selected_models, target):
            iproba = ensemble_proba[i,:,:]
            sub_proba = ensemble_proba[selected_models, :, :]
            pred = 1.0 / (1 + len(sub_proba)) * (sub_proba.sum(axis=0) + iproba)
            return (pred.argmax(axis=1) != target).mean() 
    ```

    Attributes
    ----------
    n_estimators : int, default is 5
        The number of estimators which should be selected.
    metric : function, default is error
        A function that assigns a score (smaller is better) to each classifier which is then used for selecting the next classifier in each round
    n_jobs : int, default is 8
        The number of threads used for computing the individual metrics for each classifier.
    '''
    def __init__(self, cluster_estimators = kmeans, select_estimators = random_selector, cluster_options = None, selector_options = None, n_estimators = 5):
        super().__init__()

        self.n_estimators = n_estimators
        self.cluster_estimators = cluster_estimators
        self.select_estimators = select_estimators
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

        assignments = self.cluster_estimators(proba, self.n_estimators, **self.cluster_options)
        
        selected_models = self.select_estimators(proba, assignments, target, **self.selector_options)

        return selected_models, [1.0 / len(selected_models) for _ in selected_models]