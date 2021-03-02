import numpy as np
from sklearn import metrics
import cvxpy as cp
from cvxpy import atoms
from joblib import Parallel,delayed

from PruningClassifier import PruningClassifier

from Metrics import error


# transform matrix Q according to transformation 4.3
# def transform_q_43(Q):
#     Q_tilde = np.copy(Q)
    
#     for i in range(len(Q[0])):
#         for j in range(len(Q[0])):
#             Q_tilde[i][j] = (Q[i][j] + Q[j][i]) / 2
#     return Q_tilde

# transform Matrix Q according to transformation 4.4
def transform_q_44(Q):
    currentlyMax = 0
    current = 0
    for i in range(len(Q)):
        current = np.sum(Q[i])
        if current > currentlyMax:
            currentlyMax = current

    k_1 = currentlyMax + 1
    
    for i in range(len(Q)):
        Q[i][i] = k_1 
    return Q

class ILPPruningClassifier(PruningClassifier):

    def __init__(self, 
        n_estimators = 5, 
        base_estimator = None, 
        single_metric = error,
        pairwise_metric = None, 
        l_reg = 0,
        verbose = False,
        #backend = PULP_CBC_CMD(msg=False), 
        n_jobs = 8):
        
        super().__init__(n_estimators, base_estimator, n_jobs)
        self.n_jobs = n_jobs
        self.single_metric = single_metric
        self.pairwise_metric = pairwise_metric
        self.l_reg = l_reg
        self.verbose = verbose
        #self.backend = backend
        
        assert 0 <= l_reg <= 1, "l_reg should be from [0,1], but you supplied {}".format(l_reg)
        assert pairwise_metric is not None or single_metric is not None, "You did not provide a single_metric or pairwise_metric. Please provide at-least one of them"

        if single_metric is None and l_reg < 1:
            print("Warning: You did not provide a single_metrics, but set l_reg < 1. This does not make sense. Setting l_reg = 1 for you.")

        if pairwise_metric is None and l_reg > 0:
            print("Warning: You did not provide a pairwise_metric, but set l_reg > 0. This does not make sense. Setting l_reg = 0 for you.")

    def prune_(self, proba, target):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received), [1.0 / n_received for _ in range(n_received)]

        if self.l_reg < 1:
            single_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.single_metric) (proba[i,:], proba.sum(axis=0), target) for i in range(n_received)
            )
            q = np.array(single_scores)
        else:
            q = np.zeros((n_received,1))

        if self.l_reg > 0:
            P = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.pairwise_metric) (ip, jp, target) for ip in proba for jp in proba
            )
            P = self.l_reg * np.reshape(P, (n_received,n_received))
            #P = np.reshape(P, (n_received,n_received))
        else:
            P = np.zeros((n_received,n_received))

        # TODO WHEN DO WE NEED THIS?
        if self.l_reg < 1:
            for i in range(n_received):
                #P[i][i] = 0
                P[i][i] += (1.0 - self.l_reg) * single_scores[i]

        # TODO THIS IS NEVER SET??
        # convex = False
        # if (convex):
        #     P = transform_q_44(P)
        
        
        # model = LpProblem("quadratic", LpMaximize)
        # w = LpVariable.dicts("w", range(n_received), cat=LpBinary)

        # if single_scores is not None:
        #     q = single_scores
        #     #model += (1.0 - self.l_reg) * lpSum([ wi * si for wi, si in zip(w.values(), single_scores) ] ) 
        # else:
        #     q = [0.0 for _ in range(n_received)]

        # if Q is not None:
        #     model += self.l_reg * lpSum([ w[i]*w[j]*Q[i][j] for i in range(n_received) for j in range(n_received) ]) 
            #model += self.l_reg * lpSum([ wi*wj*Q[i][j] for i in range(n_received) for j in range(n_received) ]) 

        w = cp.Variable(n_received, boolean=True)
        
        # The default GLPK_MI solver is a bit bitchy about the edge cases where l_reg is \in {0,1}. So handle that separatly
        # if self.l_reg == 0:
        #     objective = q.T @ w 
        # elif self.l_reg == 1:
        #     objective = cp.quad_form(w, P)
        # else:
        
        #objective = cp.quad_form(w, self.l_reg * P) + (1.0 - self.l_reg) * q.T @ w
        objective = cp.quad_form(w, P) 

        prob = cp.Problem(cp.Minimize(objective), [
            atoms.affine.sum.sum(w) == self.n_estimators,
            #0*w <= 0 # The default GLPK_MI solver will not work due to a bug which requires inequality constraints (https://github.com/cvxgrp/cvxpy/issues/1112)
        ]) 
        prob.solve(verbose=self.verbose)
        selected = [i for i in range(n_received) if w.value[i]]
        weights = [1.0/len(selected) for _ in selected]

        return selected, weights

        #p = matrix([1.0, 1.0])
        # G = matrix(np.zeros((n_received,n_received)))
        # h = matrix([0.0 for _ in range(n_received)]) 
        # A = matrix(np.eye(n_received))
        # b = matrix([self.n_estimators] + [0 for _ in range(n_received - 1)])
        # sol = solvers.qp(P, q, G, h, A, b)

        # model += lpSum(w[i] for i in range(n_received)) == self.n_estimators
        # status = model.solve(self.backend)

        # Q = self.l_reg * Q
        # c = (1 - self.l_reg) * single_scores
        
        # model = Model('quadratic')
        # model.modelSense = GRB.MINIMIZE
        # w = model.addVars(n_received, vtype=GRB.BINARY)
        # model.setObjective(quicksum(w[i]*w[j]*Q[i][j] for i in range(n_received) for j in range(n_received)) 
        #                 + quicksum(w[i]*c[i] for i in range(n_received)) ) 
        #                 #- (k*Q[0][0]))
        # model.addConstr(quicksum(w[i] for i in range(n_received)) == self.n_estimators)
        # model.optimize()
        # selected = [key for key,val in w.items() if value(val) > 0]
        # weights = [1.0/len(selected) for _ in selected]
        
        # return selected, weights
        # vals = [(key,value(val)) for key,val in w.items()]
        #print("scores ", single_scores)
        #print("selected ", selected)
        # print("val ", vals)
        # return [key for key,val in w.items() if value(val) > 0], [1.0/len(selected) for _ in selected]
        #return [i for i in range(n_received) if w[i].x >= 0.99]