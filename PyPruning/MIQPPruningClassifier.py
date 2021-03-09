import numpy as np
from sklearn import metrics
import cvxpy as cp
from cvxpy import atoms
from joblib import Parallel,delayed
from sklearn.metrics import pairwise

from .PruningClassifier import PruningClassifier

from .RankPruningClassifier import *

# Paper  : Combining diversity measures for ensemble pruning
# Authors: Cavalcanti et al. 2016
#
def combined(i, j, ensemble_proba, target, weights = [1.0 / 5.0 for _ in range(5)]):
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

# Paper  : Ensemble Pruning Via Semi-definite Programming 
# Authors: Zhang et al 2006 
#
def combined_error(i, j, ensemble_proba, target):
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

    def __init__(self, 
        n_estimators = 5, 
        single_metric = None,
        pairwise_metric = combined_error, 
        alpha = 1,
        verbose = False,
        n_jobs = 8):
        
        super().__init__()
        
        assert 0 <= alpha <= 1, "l_reg should be from [0,1], but you supplied {}".format(alpha)
        assert pairwise_metric is not None or single_metric is not None, "You did not provide a single_metric or pairwise_metric. Please provide at-least one of them"

        if single_metric is None and alpha < 1:
            print("Warning: You did not provide a single_metric, but set l_reg < 1. This does not make sense. Setting l_reg = 1 for you.")
            self.alpha = 1

        if pairwise_metric is None and alpha > 0:
            print("Warning: You did not provide a pairwise_metric, but set l_reg > 0. This does not make sense. Setting l_reg = 0 for you.")
            self.alpha = 0

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.single_metric = single_metric
        self.pairwise_metric = pairwise_metric
        self.alpha = alpha
        self.verbose = verbose

    def prune_(self, proba, target, data = None):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]

        if self.alpha < 1:
            single_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.single_metric) (i, proba, target) for i in range(n_received)
            )
            q = np.array(single_scores)
        else:
            q = np.zeros((n_received,1))

        if self.alpha > 0:
            pairwise_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.pairwise_metric) (i, j, proba, target) for i in range(n_received) for j in range(i, n_received)
            )

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
        else:
            P = np.zeros((n_received,n_received))

        w = cp.Variable(n_received, boolean=True)
        
        if self.alpha == 1:
            objective = cp.quad_form(w, P) 
        elif self.alpha == 0:
            objective = q.T @ w
        else:
            objective = (1.0 - self.alpha) * q.T @ w + cp.quad_form(w, self.alpha * P) 

        prob = cp.Problem(cp.Minimize(objective), [
            atoms.affine.sum.sum(w) == self.n_estimators,
        ]) 
        prob.solve(verbose=self.verbose)
        selected = [i for i in range(n_received) if w.value[i]]
        weights = [1.0/len(selected) for _ in selected]

        return selected, weights
