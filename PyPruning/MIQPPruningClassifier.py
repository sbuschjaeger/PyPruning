from functools import partial
import numpy as np
from sklearn import metrics
import cvxpy as cp
from cvxpy import atoms
from joblib import Parallel,delayed
from sklearn.metrics import pairwise

from .PruningClassifier import PruningClassifier

from .RankPruningClassifier import *

def combined(i, j, ensemble_proba, target, weights = [1.0 / 5.0 for _ in range(5)]):
    """
    Computes a (weighted) combination of 5 different measures for a pair of classifiers. The original paper also optimizes the weights of this combination using an evolutionary approach and cross-validation. Per default, we use equal weights here. You can supply a different ``weights`` via the ``metric_options`` parameter of :class:`~PyPruning.MIQPPruningClassifier`.

    Reference:
        Cavalcanti, G. D. C., Oliveira, L. S., Moura, T. J. M., & Carvalho, G. V. (2016). Combining diversity measures for ensemble pruning. Pattern Recognition Letters, 74, 38–45. https://doi.org/10.1016/j.patrec.2016.01.029
    """
    ipred = ensemble_proba[i,:,:].argmax(axis=1)
    jpred = ensemble_proba[j,:,:].argmax(axis=1)

    a = 0.0   
    b = 0.0   
    c = 0.0   #h1 incorrect, h2 correct
    d = 0.0   #h1 and h2 incorrect
    
    icorr = (ipred == target)
    jcorr = (jpred == target)
    m = len(target)

    #hi and hj correct
    a = np.logical_and(icorr, jcorr).sum()
    
    #hi correct, hj incorrect
    b = np.logical_and(icorr, np.invert(jcorr)).sum()
    
    #hi incorrect, hj correct
    c = np.logical_and(np.invert(icorr), jcorr).sum()

    #hi incorrect, hj incorrect
    d = np.logical_and(np.invert(icorr), np.invert(jcorr)).sum()
    
    # calculate the 5 different metrics
    # 1) disagreement measure 
    dis = (b+c) / m
    
    # 2) qstatistic
    # rare case: divison by zero
    if((a*d) + (b*c) == 0):
        Q = ((a*d) - (b*c)) / 1.0
    else:
        Q = ((a*d) - (b*c)) / ((a*d) + (b*c))
        
    # 3) correlation measure
    # rare case: division by zero
    if((a+b)*(a+c)*(c+d)*(b+d) == 0):
        rho = ((a*d) - (b*c)) / 0.001
    else:
        rho = ((a*d) - (b*c)) / np.sqrt( (a+b)*(a+c)*(c+d)*(b+d) )
        
    # 4) kappa statistic
    kappa1 = (a+d) / m
    kappa2 = ( ((a+b)*(a+c)) + ((c+d)*(b+d)) ) / (m*m)
    
    # rare case: kappa2 == 1
    if(kappa2 == 1):
        kappa2 = kappa2 - 0.001
    kappa = (kappa1 - kappa2) / (1 - kappa2)
    
    # 5) doublefault measure
    df = d / m
    
    # all weighted; disagreement times (-1) so that all metrics are minimized
    return weights[0] * (dis * -1.0) + weights[1] * Q + weights[2] * rho + weights[3] * kappa + weights[4] * df

# # Paper:   Effective pruning of neural network classifier ensembles
# # Authors: Lazarevic et al. 2001
# #
# def disagreement(i, j, ensemble_proba, target):
#     iproba = ensemble_proba[i,:,:].argmax(axis=1)
#     jproba = ensemble_proba[j,:,:].argmax(axis=1)

#     kappa = cohen_kappa_score(iproba, jproba)
    
#     #calculate correlation coefficient
#     m = len(iproba)
#     a = 0   #h1 and h2 correct
#     b = 0   #h1 correct, h2 incorrect
#     c = 0   #h1 incorrect, h2 correct
#     d = 0   #h1 and h2 incorrect
    
#     for j in range(m):
#         if(iproba[j] == target[j] and jproba[j] == target[j]):
#             a = a + 1
#         elif(iproba[j] == target[j] and jproba[j] != target[j]):
#             b = b + 1
#         elif(iproba[j] != target[j] and jproba[j] == target[j]):
#             c = c + 1
#         else:
#             d = d + 1
    
#     # rare case: division by 0
#     if((a+b)*(a+c)*(c+d)*(b+d) == 0):
#         correlation = (a*d) - (b*c) / 0.001
#     else:
#         correlation = ((a*d) - (b*c)) / np.sqrt( (a+b)*(a+c)*(c+d)*(b+d) )
    
#     # equally weighted (no further explanations)
#     return (kappa + correlation) / 2

def combined_error(i, j, ensemble_proba, target):
    """
    Computes the pairwise errors of the two classifiers i and j.

    Reference:
        Zhang, Y., Burer, S., & Street, W. N. (2006). Ensemble Pruning Via Semi-definite Programming. Journal of Machine Learning Research, 7, 1315–1338. https://doi.org/10.1016/j.jasms.2006.06.007
    """
    iproba = ensemble_proba[i,:,:].argmax(axis=1)
    jproba = ensemble_proba[j,:,:].argmax(axis=1)

    ierr = (iproba != target).astype(np.int32)
    jerr = (jproba != target).astype(np.int32)
    if i == j:
        return (ierr * jerr).mean()
    else:
        Gi = (ierr * ierr).sum()
        Gj = (jerr * jerr).sum()
        combined = ierr * jerr
        return 0.5 * (combined / Gi + combined / Gj).sum()

    # if i == j:
    #     return (ierr*jerr).mean()
    # else:

    # count_h1h2 = 0.0
    # count_h1 = 0.0
    # count_h2 = 0.0
    
    # for j in range(len(iproba)):
    #     if(iproba[j] != target[j]):
    #         count_h1 = count_h1 + 1
    #     if(jproba[j] != target[j]):
    #         count_h2 = count_h2 + 1
    #     if (iproba[j] != target[j] and jproba[j] != target[j]):
    #         count_h1h2 = count_h1h2 + 1
    
    # # avoid rare error: division by 0
    # # if one of these cases occur, count_h1h2 will be 0 aswell
    # # thus count_h1/h2 = 1 wont matter
    # if(count_h1 == 0):
    #     count_h1 = 1.0
    # if(count_h2 == 0):
    #     count_h2 = 1.0
    
    # return ( (count_h1h2 / count_h1 ) + (count_h1h2 / count_h2) ) / 2.0

class MIQPPruningClassifier(PruningClassifier):
    """ Mixed Integer Quadratic Programming (MIQP) Pruning.

    This pruning method constructs a MIQP so that its solution is the pruned ensemble. Formally, it uses the problem
    
    .. math::

        \\arg\\min_w (1 - \\alpha ) q^T w + \\alpha w^T P w

    where :math:`\\alpha \\in [0,1]` is the trade-off between the first and the second term. The first vector q contains the individual metrics for each classifier similar to what a RankPruningClassifier would compute, whereas P contains pairwise metrics for each classifier pair in the ensemble. To compute :math:`q` and :math:`P` there are two metrics required:

    **Single_metric**
    
    This metric assigns a value to each individual classifier in the ensemble without considering pairs of classifier. A single_metric function should accept the following parameters:

    - ``i`` (int): The classifier which should be rated
    - ``ensemble_proba`` (A (M, N, C) matrix ): All N predictions of all M classifier in the entire ensemble for all C classes
    - ``target`` (list / array): A list / array of class targets.
    
    The single_metric is compatible with the metrics for a :class:`~PyPruning.RankPruningClassifier`. You can use any metric from the :class:`~PyPruning.RankPruningClassifier` here and vice-versa

    **Pairwise_metric**

    This metric assigns a value to each pair of classifiers in the ensemble. A pairwise_metric function should accept the following parameters:

    - ``i`` (int): The first classifier in the pair
    - ``j`` (int): The second classifier in the pair 
    - ``ensemble_proba`` (A (M, N, C) matrix ): All N predictions of all M classifier in the entire ensemble for all C classes
    - ``target`` (list / array): A list / array of class targets.
    
    If you set ``alpha = 0`` or choose the pairwise metric that simply returns 0 a MIQPPruningClassifier should produce the same solution as a RankPruningClassifier does. 

    **Important:** All metrics are **minimized**. If you implement your own metric make sure that it assigns smaller values to better classifiers.
    
    This code uses ``cvxpy`` to access a wide variety of MQIP solver. For more information on how to configure your solver and interpret its output in case of failures please have a look at the cvxpy documentation https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options.

    Attributes
    ----------
    n_estimators : int, default is 5
        The number of estimators which should be selected.
    single_metric : function, default is None
        A function that assigns a value to each classifier which forms the q vector
    pairwise_metric : function, default is combined_error
        A function that assigns a value to each pair of classifiers which forms the P matrix
    alpha : float, must be in [0,1]
        The trade-off between the single and pairwise metric. alpha = 0 only considers the single_metric, whereas alpha = 1 only considers the pairwise metric 
    eps : float, default 1e-2
        Sometimes and especially for larger P matrices there can be numerical inaccuries. In this case, the resulting problem might become non-convex so that the MQIP solver cannot solve the problem anymore. For a better numerical stability the eps value can be added to the diagonal of the P matrix. 
    verbose : boolean, default is False
        If true, more information from the MQIP solver is printed. 
    n_jobs : int, default is 8
        The number of threads used for computing the metrics. This does not have any effect on the number of threads used by the MQIP solver.
    """

    def __init__(self, n_estimators = 5, single_metric = None, pairwise_metric = combined_error, alpha = 1, eps = 1e-2, verbose = False, n_jobs = 8, single_metric_options = None, pairwise_metric_options = None):
        """ 
        Creates a new MIQPPruningClassifier.

        Parameters
        ----------

        n_estimators : int, default is 5
            The number of estimators which should be selected.
        single_metric : function, default is None
            A function that assigns a value to each classifier which forms the q vector
        pairwise_metric : function, default is combined_error
            A function that assigns a value to each pair of classifiers which forms the P matrix
        alpha : float, must be in [0,1]
            The trade-off between the single and pairwise metric. alpha = 0 only considers the single_metric, whereas alpha = 1 only considers the pairwise metric 
        eps : float, default 1e-2
            Sometimes and especially for larger P matrices there can be numerical inaccuries. In this case, the resulting problem might become non-convex so that the MQIP solver cannot solve the problem anymore. For a better numerical stability the eps value can be added to the diagonal of the P matrix. 
        verbose : boolean, default is False
            If true, more information from the MQIP solver is printed. 
        n_jobs : int, default is 8
            The number of threads used for computing the metrics. This does not have any effect on the number of threads used by the MQIP solver.
        kwargs : 
            Any additional kwargs are directly supplied to single_metric function and pairwise_metric function via partials
        """
        super().__init__()
        
        assert 0 <= alpha <= 1, "l_reg should be from [0,1], but you supplied {}".format(alpha)
        assert eps >= 0, "Eps should be >= 0, but you supplied".format(eps)

        assert pairwise_metric is not None or single_metric is not None, "You did not provide a single_metric or pairwise_metric. Please provide at-least one of them"

        if single_metric is None and alpha < 1:
            print("Warning: You did not provide a single_metric, but set l_reg < 1. This does not make sense. Setting l_reg = 1 for you.")
            self.alpha = 1

        if pairwise_metric is None and alpha > 0:
            print("Warning: You did not provide a pairwise_metric, but set l_reg > 0. This does not make sense. Setting l_reg = 0 for you.")
            self.alpha = 0

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        if single_metric_options is None:
            self.single_metric_options = {}
        else:
            self.single_metric_options = single_metric_options
        
        if pairwise_metric_options is None:
            self.pairwise_metric_options = {}
        else:
            self.pairwise_metric_options = pairwise_metric_options
        
        self.alpha = alpha
        self.verbose = verbose
        self.eps = eps

    # I assume that Parallel keeps the order of evaluations regardless of its backend (see eg. https://stackoverflow.com/questions/56659294/does-joblib-parallel-keep-the-original-order-of-data-passed)
    # But for safety measures we also return the index of the current model
    def _single_metric(self, i, proba, target, additional_options):
        return (i, self.single_metric(i, proba, target, **additional_options))

    def _pairwise_metric(self, i, j, proba, target, additional_options):
        return (i, self.pairwise_metric(i, proba, target, **additional_options))

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]

        if self.alpha < 1:
            single_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._single_metric) (i, proba, target, self.single_metric_options) for i in range(n_received)
            )
            # TODO MAKE SURE SORTING IS CORRECT
            # best_model, _ = min(scores, key = lambda e: e[1])
            q = np.array(single_scores)
        else:
            q = np.zeros((n_received,1))

        if self.alpha > 0:
            pairwise_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._pairwise_metric) (i, j, proba, target, self.pairwise_metric_options) for i in range(n_received) for j in range(i, n_received)
            )
            # TODO MAKE SURE SORTING IS CORRECT
            # best_model, _ = min(scores, key = lambda e: e[1])

            # TODO This is probably easier and quicker with some fancy numpy operations
            P = np.zeros((n_received,n_received))
            s = 0
            for i in range(n_received):
                for j in range(i, n_received):
                    if i == j:
                        P[i,j] = pairwise_scores[s]
                    else:
                        P[i,j] = pairwise_scores[s]
                        P[j,i] = pairwise_scores[s]
                    s += 1
            P += self.eps * np.eye(n_received)

        else:
            P = np.zeros((n_received,n_received))

        w = cp.Variable(n_received, boolean=True)
        
        if self.alpha == 1:
            objective = cp.quad_form(w, P) 
        elif self.alpha == 0:
            objective = q.T @ w
        else:
            objective = cp.pos((1.0 - self.alpha)) * q.T @ w + cp.pos(self.alpha) * cp.quad_form(w, P)

        prob = cp.Problem(cp.Minimize(objective), [
            atoms.affine.sum.sum(w) == self.n_estimators,
        ]) 
        prob.solve(verbose=self.verbose)
        selected = [i for i in range(n_received) if w.value[i]]
        weights = [1.0/len(selected) for _ in selected]

        return selected, weights
