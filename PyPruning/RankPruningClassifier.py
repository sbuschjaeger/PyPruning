from functools import partial
import numpy as np

from sklearn.metrics import roc_auc_score, cohen_kappa_score

from scipy import spatial

from joblib import Parallel,delayed

from .PruningClassifier import PruningClassifier

def individual_margin_diversity(i, ensemble_proba, target, alpha = 0.2):
    '''
    Computes the individual diversity of the classifier wrt. to the ensemble and its contribution to the margin. alpha controls the trade-off between both values.

    Note: The paper uses alpha = 0.2 in all experiments and reports that it worked well. Thus, it is also the default value here. If you want to change this value you can use `partial` to set it to a different value (e.g. 0.5) before creating a new RankPruningClassifier:

    ```Python
        from functools import partial
        m_function = partial(individual_margin_diversity, alpha = 0.5)
        pruner = RankPruningClassifier(n_estimators = 10, metric = m_function, n_jobs = 8)
    ```

    Reference:
        Guo, H., Liu, H., Li, R., Wu, C., Guo, Y., & Xu, M. (2018). Margin & diversity based ordering ensemble pruning. Neurocomputing, 275, 237–246. https://doi.org/10.1016/j.neucom.2017.06.052
    '''
    iproba = ensemble_proba[i,:,:]
    n = iproba.shape[0]

    predictions = iproba.argmax(axis=1)
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
    V = V.sum(axis=0)

    #V = jproba #all_proba.sum(axis=0)#.argmax(axis=1)
    MDM = 0
    
    for j in range(n):
        if (predictions[j] == target[j]):
            
            # special case for margin: prediction for label with majority of votes
            if(predictions[j] == np.argmax(V[j,:])):
                # calculate margin with second highest number of votes
                sortedArray = np.sort(np.copy(V[j,:]))
                
                # check whether 1. and 2. max vot counts are equal! (margin = 0)
                if(sortedArray[-2] == np.max(V[j,:])):
                    margin = (  V[j, target[j]]  - (sortedArray[-2] -1)   ) / n
                else:
                    margin = (  V[j, target[j]]  - sortedArray[-2]   ) / n
                   
            else:
                # usual case for margin: prediction not label with majority of votes
                margin = (  V[j, target[j]]  - np.max(V[j,:])   ) / n
            
            
            # somehow theres still a rare case for margin == 0
            if(margin == 0):
                margin = 0.01
            
            fm = np.log(abs(margin))
            fd = np.log(V[j, target[j]] / n)
            MDM = MDM + (alpha*fm) + ((1-alpha)*fd)
    return - 1.0 * MDM

def individual_contribution(i, ensemble_proba, target):
    '''
    Compute the individual contributions of each classifier wrt. the entire ensemble. Return the negative contribution due to the minimization.

    Reference:
        Lu, Z., Wu, X., Zhu, X., & Bongard, J. (2010). Ensemble pruning via individual contribution ordering. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 871–880. https://doi.org/10.1145/1835804.1835914
    '''
    iproba = ensemble_proba[i,:,:]
    n = iproba.shape[0]

    predictions = iproba.argmax(axis=1)
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
    V = V.sum(axis=0)

    IC = 0
    #V = all_proba.argmax(axis=2)
    #predictions = iproba.argmax(axis=1)
    #V = all_proba.sum(axis=0)#.argmax(axis=1)

    for j in range(n):
        if (predictions[j] == target[j]):
            
            # case 1 (minority group)
            # label with majority votes on datapoint  = np.argmax(V[j, :]) 
            if(predictions[j] != np.argmax(V[j,:])):
                IC = IC + (2*(np.max(V[j,:])) - V[j, predictions[j]])
                
            else: # case 2 (majority group)
                # calculate second largest nr of votes on datapoint i
                sortedArray = np.sort(np.copy(V[j,:]))
                IC = IC + (sortedArray[-2])
                
        else:
            # case 3 (wrong prediction)
            IC = IC + (V[j, target[j]]  -  V[j, predictions[j]] - np.max(V[j,:]) )
    return - 1.0 * IC

def individual_error(i, ensemble_proba, target):
    ''' 
    Compute the error for the individual classifier. If I read it correctly, then the following paper proposed this method. Although the paper is not super clear on this.

    Reference:
        Jiang, Z., Liu, H., Fu, B., & Wu, Z. (2017). Generalized ambiguity decompositions for classification with applications in active learning and unsupervised ensemble pruning. 31st AAAI Conference on Artificial Intelligence, AAAI 2017, 2073–2079.
    '''
    iproba = ensemble_proba[i,:,:]
    return (iproba.argmax(axis=1) != target).mean()

def error_ambiguity(i, ensemble_proba, target):
    '''
    Compute the error for the individual classifier according to the ambiguity decomposition. I am fairly sure that this implementation is correct, however, the paper is not super clear on what they do from an algorithmic point of view. From what I can tell is, that the authors compute the ambiguity scores for each classifier only once and then "greedily" pick the best K models. 

    Note: The paper only considers binary classification problems and specifically focuses on the logistic loss function. Luckily, Hastie et al. proposed a multi-class boosting algorithm which uses a multi class variation of the (binary) logistic loss. Both loss functions are equal for 2 classes and thus we implement the multi-class version here. For more details see the reference.

    Reference:
        Jiang, Z., Liu, H., Fu, B., & Wu, Z. (2017). Generalized ambiguity decompositions for classification with applications in active learning and unsupervised ensemble pruning. 31st AAAI Conference on Artificial Intelligence, AAAI 2017, 2073–2079.
        
        Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class AdaBoost. Statistics and Its Interface, 2(3), 349–360. https://doi.org/10.4310/sii.2009.v2.n3.a8
    '''
    iproba = ensemble_proba[i,:,:]
    all_proba = ensemble_proba.mean(axis=0)
    sqdiff = (iproba - all_proba)**2

    C = iproba.shape[1]

    A = 1.0 / C**2 * np.exp(- 1.0 / C * iproba)
    B = 1.0 / C**2 * (1.0 / (C-1))**2 * np.exp(1.0 / C * 1.0 / (C-1) * iproba)

    bitmask = np.zeros(A.shape)
    # bitmask[:,target] = 1.0
    np.put_along_axis(bitmask, target[:,None], 1.0, 1)
    return (bitmask * A + (1.0 - bitmask) * B).sum() + sqdiff.sum()

    # for j in range(iproba.shape[0]):
    #     for c in range(C):
    #         if target[j] == c:
    #             A += 1.0 / C**2 * np.exp(- 1.0 / C * iproba[j,c])
    #         else:
    #             A += 1.0 / C**2 * (1.0 / (C-1))**2 * np.exp(1.0 / C * 1.0 / (C-1) * iproba[j,c])
    
    # return A + sqdiff.sum()

def individual_neg_auc(i, ensemble_proba, target):
    ''' 
    Compute the roc auc score for the individual classifier, but return its negative value for minimization.
    '''
    iproba = ensemble_proba[i,:,:]
    if(iproba.shape[1] == 2):
        iproba = iproba.argmax(axis=1)
        return - 1.0 * roc_auc_score(target, iproba)
    else:
        return - 1.0 * roc_auc_score(target, iproba, multi_class="ovr")

def individual_kappa_statistic(i, ensemble_proba, target):
    ''' 
    Compute the Cohen-Kappa statistic for the individual classifier with respect to the entire ensemble.

    Reference:
        Margineantu, D., & Dietterich, T. G. (1997). Pruning Adaptive Boosting. Proceedings of the Fourteenth International Conference on Machine Learning, 211–218. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.38.7017&rep=rep1&type=pdf
    '''
    scores = []
    iproba = ensemble_proba[i,:,:].argmax(axis=1)

    for j, jproba in enumerate(ensemble_proba):
        if j != i:
            # See https://github.com/scikit-learn/scikit-learn/issues/14256
            # and https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
            with np.errstate(divide='ignore',invalid='ignore'):
                score = cohen_kappa_score(iproba, jproba.argmax(axis=1))
                if np.isnan(score):
                    scores.append(0.0)
                else:
                    scores.append(score)
    return min(scores)

def reference_vector(i, ensemble_proba, target):
    '''
    Compare how close the individual predictions is to the entire ensemble's prediction by using the cosine similary

    Note: The paper describes a slightly different distance metric compared to what is implemented here. The paper uses a projection to a reference vector, but - unfortunately - does not explain the specific implementation in detail. However, the authors also note two things:
    
    - (1) They use all classifier with an angle <= pi/2 which can lead to more than n_estimator classifier. This implementation selects at most n_estimators and thus we need to present an ordering based on the angles and pick the first n_estimator.
    - (2) "The classifiers are ordered by increasing values of the angle between the signature vectors of the individual classifiers and the reference vector". 
    
    `ref` and `ipred` (see source code) follow the exact definitions as presented in the paper (eq. 3) and cosine is the most direct implementation of "the angle between signature and reference vector" 


    Reference:
        Hernández-Lobato, D., Martínez-Muñoz, G., & Suárez, A. (2006). Pruning in Ordered Bagging Ensembles. International Conference on Machine Learning, 1266–1273. https://doi.org/10.1109/ijcnn.2006.246837
    '''
    ref = 2 * (ensemble_proba.mean(axis=0).argmax(axis=1) == target) - 1.0
    ipred = 2 * (ensemble_proba[i,:].argmax(axis=1) == target) - 1.0
    return 1.0 - spatial.distance.cosine(ref, ipred)
    # ref /= np.linalg.norm(ref)
    # ipred /= np.linalg.norm(ipred)
    #return np.dot(ref, ipred)

class RankPruningClassifier(PruningClassifier):
    ''' Rank pruning. 
    
    Ranking methods assign a rank to each classifier in the ensemble and then select the best n_estimators according to this ranking. To rate each classifier a metric must be given. A metric is a function with receives three parameters:
        
    - `i` (int): The classifier which should be rated
    - `ensemble_proba` (A (M, N, C) matrix ): All N predictions of all M classifier in the entire ensemble for all C classes
    - `target` (list / array): A list / array of class targets.

    A simple example for this function would be the individual error of each method:
    
    ```Python
        def individual_error(i, ensemble_proba, target):
            iproba = ensemble_proba[i,:,:]
            return (iproba.argmax(axis=1) != target).mean()
    ```

    **Important** The classifiers are sorted in ascending order and the first n_estimators are selected. Differently put, the metric is always minimized.

    Attributes
    ----------
    n_estimators : int, default is 5
        The number of estimators which should be selected.
    metric : function, default is individual_error 
        A function that assigns a score to each classifier which is then used for sorting
    n_jobs : int, default is 8
        The number of threads used for computing the individual metrics for each classifier.
    '''
    def __init__(self, n_estimators = 5, metric = individual_error, n_jobs = 8, **kwargs):
        """
        Creates a new RankPruningClassifier.

        Parameters
        ----------

        n_estimators : int, default is 5
            The number of estimators which should be selected.
        metric : function, default is individual_error 
            A function that assigns a score to each classifier which is then used for sorting
        n_jobs : int, default is 8
            The number of threads used for computing the individual metrics for each classifier.
        kwargs : 
            Any additional kwargs are directly supplied to the metric function via a partial
        """
        super().__init__()

        assert metric is not None, "You must provide a valid metric!"
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        if len(kwargs) > 0:
            self.metric = partial(metric, **kwargs)
        else:
            self.metric = metric

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]
        
        single_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self.metric) (i, proba, target) for i in range(n_received)
        )
        single_scores = np.array(single_scores)

        return np.argpartition(single_scores, self.n_estimators)[:self.n_estimators], [1.0 / self.n_estimators for _ in range(self.n_estimators)]
        