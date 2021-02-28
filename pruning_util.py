"""

Implementation of the Ensemble Pruning Framework presented in "Anwendung von Ensemble-
Modellen unter Ressourcenbeschränkungen: Ein Framework für Ensemble Pruning Verfahren".

This module contains functions that are used to calculate pruning metrics and to 
solve the ensemble pruning problem. Please see the bachelor thesis for further explanations 
regarding the use of this module, signatures of functions etc.

"""

import numpy as np
import math
import random
from joblib import Parallel, delayed
from gurobipy import Model, GRB, quicksum
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score



# ----------------
# Internal Methods
# 
# 
# ----------------
  
# functions to calculate the metric vector c / matrix Q
def fill_c_parallel(predictions, y, V, metric, jobs):
    n_est = len(predictions)
    c = Parallel(n_jobs=jobs, backend="threading")(
            delayed(metric) (predictions[i], y, n_est, V) for i in range(n_est)
        )
    return np.array(c)

def fill_q_parallel(predictions, y, metric, jobs):
    n_est = len(predictions)
    Q = Parallel(n_jobs=jobs, backend="threading")(
                 delayed(metric) (predictions[i], predictions[j], y) for i in range(n_est) 
                 for j in range(n_est)
        )
    Q = np.reshape(Q, (n_est,n_est))
    for i in range(n_est):
        Q[i][i] = 0
    return Q


# function for calculating the number of votes for each label on each data point
# shape of allVotes: 1. dimension number of data entries, 2. dimension number of labels
def get_initial_votes(predictions, labelNumbers):
    allVotes = []
    for i in range(len(predictions[0])):
        votesOnXi = []
        for j in range(labelNumbers):
            votesOnXi.append(np.count_nonzero(predictions[:, i] == j))
        allVotes.append(votesOnXi)
    
    return np.array(allVotes)


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
    

# transform matrix Q according to transformation 4.3
def transform_q_43(Q):
    Q_tilde = np.copy(Q)
    
    for i in range(len(Q[0])):
        for j in range(len(Q[0])):
            Q_tilde[i][j] = (Q[i][j] + Q[j][i]) / 2
    return Q_tilde


# function for calculating subensemble predictions using majority voting
# only considers models (indices) in array selected
def aggregate_votes(preds, selected):
    selectedClassifiers = []
    for i in range(len(selected)):
        selectedClassifiers.append(preds[selected[i]]) 
    selectedClassifiers = np.array(selectedClassifiers)
    
    chosenLabels = []
    for j in range(len(selectedClassifiers[0])):
        counts = np.bincount(selectedClassifiers[:,j])
        chosenLabels.append(np.argmax(counts))
        
    return np.array(chosenLabels)
        

    
# -------
# Solving Methods
#
# These functions try to approximate the solution of the ensemble pruning
# problem in regards to the chosen pruning metric.
#
# Mathematical programming methods MINIMIZE the metric values while 
# the greedy algorithms MAXIMIZE or MINIMIZE metric values!
# 
#
# Three variants are included and it is important to choose the correct
# one in regards to the type of pruning metric:
# (1) for pruning metrics that evaluate single models
# (2) for pruning metrics that evaluate paarwise combinations of models
# (3) for a combination of the above
# -------


def solve_miqp_1(c, p):
    n = len(c)
    k = int(p * n)
    
    model = Model('linear')
    model.modelSense = GRB.MINIMIZE
    w = model.addVars(n, vtype=GRB.BINARY)
    model.setObjective(quicksum(w[i]*c[i] for i in range(n)))
    model.addConstr(quicksum(w[i] for i in range(n)) == k)
    model.optimize()
    
    return [i for i in range(n) if w[i].x >= 0.99]


# note: no transformations on Q (increases computation time of gurobi significantly!)
def solve_miqp_2(Q, p):
    convex = False
    if(convex):
        Q = transform_q_44(Q)
    n = len(Q[0])
    k = int(p * n)
    
    model = Model('quadratic')
    model.modelSense = GRB.MINIMIZE
    w = model.addVars(n, vtype=GRB.BINARY)
    model.setObjective(quicksum(w[i]*w[j]*Q[i][j] for i in range(n) for j in range(n))) 
                       #- (k*Q[0][0]))
    model.addConstr(quicksum(w[i] for i in range(n)) == k)
    model.optimize()
    
    return [i for i in range(n) if w[i].x >= 0.99]


# expand: choose parameter alpha
def solve_miqp_3(c, Q, p):
    alpha = 0.5
    convex = False
    if(convex):
        Q = transform_q_44(Q)
    Q = alpha * Q
    c = (1 - alpha) * c
    n = len(c)
    k = int(p * n)
    
    model = Model('quadratic')
    model.modelSense = GRB.MINIMIZE
    w = model.addVars(n, vtype=GRB.BINARY)
    model.setObjective(quicksum(w[i]*w[j]*Q[i][j] for i in range(n) for j in range(n)) 
                       + quicksum(w[i]*c[i] for i in range(n)) ) 
                       #- (k*Q[0][0]))
    model.addConstr(quicksum(w[i] for i in range(n)) == k)
    model.optimize()
    
    return [i for i in range(n) if w[i].x >= 0.99]


# only for maximizing pruning metric
def solve_greedy_1(c, p):
    # sorts from low to high values
    
    # zip and sort
    indicies = np.arange(len(c))
    zipped = zip(indicies, c)
    zipped = sorted(zipped, key = lambda t: t[1])
    indicies, values = zip(*zipped)
    
    #select top k classifiers
    sol = []
    for i in range(int(len(c)*p)):
        sol.append(indicies[len(indicies)-(i+1)])
    return sol


def solve_greedy_2(preds, metric, y, p, ordering):
    # step 0: initialize
    k = int(len(preds) * p)
    H = set(range(len(preds)))  # models to select from
    S = set() # current subensemble
    
    # step 1: choose model with highest accuracy on pruning data
    # (default option for greedy 2)
    accuracies = []
    for i in range(len(preds)):
        accuracies.append(accuracy_score(y, preds[i]))
    chosen = np.argmax(accuracies)
    S.add(chosen)
    H.remove(chosen)
    
    #step 2: iteratively add (k-1) models to S
    for i in range(k-1):
        # get subensemble S prediction
        predictionS = aggregate_votes(preds, list(S))
        
        # find model with highest/lowest score
        order = []
        scores = []
        copyH = H.copy()
        for j in range(len(H)):
            current = copyH.pop()
            order.append(current)
            scores.append(metric(preds[current], predictionS, y))    
        zipped = zip(order, scores)
        zipped = sorted(zipped, key = lambda t: t[1])
        indicies, values = zip(*zipped)
        
        if(ordering == "maximize"):
            S.add(indicies[len(H)-1])
            H.remove(indicies[len(H)-1])
        elif(ordering == "minimize"):
            S.add(indicies[0])
            H.remove(indicies[0])
    
    return list(S)


# expand: choose parameter alpha
def solve_greedy_3(preds, c, metric2, y, p, ordering):
    alpha = 0.5
    
    # step 0: initialize
    k = int(len(preds) * p)
    H = set(range(len(preds)))  # models to select from
    S = set() # current subensemble
    
    # step 1: choose model with highest/lowest c on pruning data
    if(ordering == "maximize"):
        chosen = np.argmax(c)
    elif(ordering == "minimize"):
        chosen = np.argmin(c)
    
    S.add(chosen)
    H.remove(chosen)
    
    #step 2: iteratively add (k-1) models to S
    for i in range(k-1):
        # get subensemble S prediction
        predictionS = aggregate_votes(preds, list(S))
        
        # find model with highest/lowest score
        order = []
        scores = []
        copyH = H.copy()
        for j in range(len(H)):
            current = copyH.pop()
            order.append(current)
            scores.append( (alpha * metric2(preds[current], predictionS, y)) 
                          + ((1-alpha) * c[current]))    
        zipped = zip(order, scores)
        zipped = sorted(zipped, key = lambda t: t[1])
        indicies, values = zip(*zipped)
        
        if(ordering == "maximize"):
            S.add(indicies[len(H)-1])
            H.remove(indicies[len(H)-1])
        elif(ordering == "minimize"):
            S.add(indicies[0])
            H.remove(indicies[0])
    
    return list(S)


def solve_random(n, p):
    k = int(n * p)
    indices = np.arange(n)
    return random.choices(indices, k=k)



# -------
# Pruning Metrics
#
# Custom metrics that are used for pruning ensembles. Please see the comments or
# the bachelor thesis for references to the original papers.
# -------



# Paper:   Ensemble Pruning via individual contribution
# Authors: Lu et al. 2010
#
def c_lu10(predictions, y, n, V):
    IC = 0
    
    for j in range(len(predictions)):
        if (predictions[j] == y[j]):
            
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
            IC = IC + (V[j, y[j]]  -  V[j, predictions[j]] - np.max(V[j,:]) )
    return IC



# Paper:   Margin & Diversity based ordering ensemble pruning
# Authors: Guo et al. 2018
#
def c_guo18(predictions, y, n, V):
    MDM = 0
    alpha = 0.2
    
    for j in range(len(predictions)):
        if (predictions[j] == y[j]):
            
            # special case for margin: prediction for label with majority of votes
            if(predictions[j] == np.argmax(V[j,:])):
                # calculate margin with second highest number of votes
                sortedArray = np.sort(np.copy(V[j,:]))
                
                # check whether 1. and 2. max vot counts are equal! (margin = 0)
                if(sortedArray[-2] == np.max(V[j,:])):
                    margin = (  V[j, y[j]]  - (sortedArray[-2] -1)   ) / n
                else:
                    margin = (  V[j, y[j]]  - sortedArray[-2]   ) / n
                   
            else:
                # usual case for margin: prediction not label with majority of votes
                margin = (  V[j, y[j]]  - np.max(V[j,:])   ) / n
            
            
            # somehow theres still a rare case for margin == 0
            if(margin == 0):
                margin = 0.01
            
            fm = math.log(abs(margin))
            fd = math.log(V[j, y[j]] / n)
            MDM = MDM + (alpha*fm) + ((1-alpha)*fd)
    return MDM



# Paper  : Ensemble Pruning Via Semi-definite Programming 
# Authors: Zhang et al 2006 
#
def c_zhang06(predictions, y, n, V):
    count = 0
    
    for j in range(len(predictions)):
        if (predictions[j] != y[j]):
            count = count + 1
    return (count / len(predictions))

def q_zhang06(predictions_h1, predictions_h2, y):
    count_h1h2 = 0
    count_h1 = 0
    count_h2 = 0
    
    for j in range(len(predictions_h1)):
        if(predictions_h1[j] != y[j]):
            count_h1 = count_h1 + 1
        if(predictions_h2[j] != y[j]):
            count_h2 = count_h2 + 1
        if (predictions_h1[j] != y[j] and predictions_h2[j] != y[j]):
            count_h1h2 = count_h1h2 + 1
    
    # avoid rare error: division by 0
    # if one of these cases occur, count_h1h2 will be 0 aswell
    # thus count_h1/h2 = 1 wont matter
    if(count_h1 == 0):
        count_h1 = 1
    if(count_h2 == 0):
        count_h2 = 1
    
    return ( (count_h1h2 / count_h1 ) + (count_h1h2 / count_h2) ) / 2




# Paper  : Combining diversity measures for ensemble pruning
# Authors: Cavalcanti et al. 2016
#
def q_cavalcanti16(predictions_h1, predictions_h2, y):
    m = len(predictions_h1)
    a = 0   #h1 and h2 correct
    b = 0   #h1 correct, h2 incorrect
    c = 0   #h1 incorrect, h2 correct
    d = 0   #h1 and h2 incorrect
    
    for j in range(m):
        if(predictions_h1[j] == y[j] and predictions_h2[j] == y[j]):
            a = a + 1
        elif(predictions_h1[j] == y[j] and predictions_h2[j] != y[j]):
            b = b + 1
        elif(predictions_h1[j] != y[j] and predictions_h2[j] == y[j]):
            c = c + 1
        else:
            d = d + 1
    
    # calculate the 5 different metrics
    # 1) disagreement measure 
    disagree = (b+c) / m
    
    # 2) qstatistic
    # rare case: divison by zero
    if((a*d) + (b*c) == 0):
        qstatistic = ((a*d) - (b*c)) / 1 
    else:
        qstatistic = ((a*d) - (b*c)) / ((a*d) + (b*c))
        
    # 3) correlation measure
    # rare case: division by zero
    if((a+b)*(a+c)*(c+d)*(b+d) == 0):
        correlation = ((a*d) - (b*c)) / 0.001
    else:
        correlation = ((a*d) - (b*c)) / math.sqrt( (a+b)*(a+c)*(c+d)*(b+d) )
        
    # 4) kappa statistic
    kappa1 = (a+d) / m
    kappa2 = ( ((a+b)*(a+c)) + ((c+d)*(b+d)) ) / (m*m)
    
    # rare case: kappa2 == 1
    if(kappa2 == 1):
        kappa2 = kappa2 - 0.001
    kappa = (kappa1 - kappa2) / (1 - kappa2)
    
    # 5) doublefault measure
    doublefault = d / m
    
    # all equally weighted; disagreement times (-1) so that all metrics are minimized
    return ((disagree * -1) + qstatistic + correlation + kappa + doublefault) / 5
        

    
# Paper  : Pruning adaptive boosting
# Authors: Margineantu et Dietterich 1997 
# Uses the sklearn-implementation of the "Kappa-Statistic" measure
def q_margineantu97(predictions_h1, predictions_h2, y):
    return cohen_kappa_score(predictions_h1, predictions_h2)
    
    
    
    
# Paper:   Effective pruning of neural network classifier ensembles
# Authors: Lazarevic et al. 2001
#
def q_lazarevic01(predictions_h1, predictions_h2, y):
    kappa = cohen_kappa_score(predictions_h1, predictions_h2)
    
    #calculate correlation coefficient
    m = len(predictions_h1)
    a = 0   #h1 and h2 correct
    b = 0   #h1 correct, h2 incorrect
    c = 0   #h1 incorrect, h2 correct
    d = 0   #h1 and h2 incorrect
    
    for j in range(m):
        if(predictions_h1[j] == y[j] and predictions_h2[j] == y[j]):
            a = a + 1
        elif(predictions_h1[j] == y[j] and predictions_h2[j] != y[j]):
            b = b + 1
        elif(predictions_h1[j] != y[j] and predictions_h2[j] == y[j]):
            c = c + 1
        else:
            d = d + 1
    
    # rare case: division by 0
    if((a+b)*(a+c)*(c+d)*(b+d) == 0):
        correlation = (a*d) - (b*c) / 0.001
    else:
        correlation = ((a*d) - (b*c)) / math.sqrt( (a+b)*(a+c)*(c+d)*(b+d) )
    
    # equally weighted (no further explanations)
    return (kappa + correlation) / 2
        

    
# accuracy for ensemble pruning
# Uses the sklearn-implementation
def c_accuracy(predictions, y, n, V):
    return accuracy_score(y, predictions)


# Area under the ROC-Curve (AUC) metric for ensemble pruning
# Uses the sklearn-implementation
def c_auc(predictions_proba, y, n, V):
    if(len(predictions_proba[0]) == 2):
        return roc_auc_score(y, predictions_proba[:, 1])
    else:
        return roc_auc_score(y, predictions_proba, multi_class="ovr")



# -------
# Specific pruning metrics for models from Machine Learning Libraries 
#
# 
# -------


# for forest classifiers from scikit-learn library
def prune_sklearn(sklearn_model, X, y, labelcount, metric, optimizer, p):
    
    # check whether correct model, fitted etc.
    forest = sklearn_model
    
    # calculate predictions
    preds = []
    for h in forest.estimators_:
        preds.append(h.predict(X))
    preds = np.array(preds, dtype=int)
    
    # calculate votes for each label on all data entries
    votes = get_initial_votes(preds, labelcount)
    
    # pass prediction probabilities if metric == auc 
    # (or other metrics that need proba(X))
    if(metric == "auc"):
        preds_proba = []
        for h in forest.estimators_:
            preds_proba.append(h.predict_proba(X))
            preds = np.array(preds_proba)
    
    
    # calculate selected estimators
    sol = prune(preds, y, votes, metric, optimizer, p)
    
    # replace list of estimators
    # print(sol)
    chosen = []
    for i in range(len(sol)):
        chosen.append(forest.estimators_[sol[i]])
    forest.estimators_ = chosen
    
    
    
# -------
# Pruning Method
#
# This function selects the correct pruning metric and approximation method  
# depending on the pruning configuration. It requires the predictions of the
# models in preds and votes.
# -------    
    
def prune(preds, y, votes, metric, optimizer, p):
    # change depending on system
    jobs = 8
    
    if(optimizer == "greedy"):
        if(metric == "lu2010"):
            c = fill_c_parallel(preds, y, votes, c_lu10, jobs)
            return solve_greedy_1(c, p)
        elif(metric == "guo2018"):
            c = fill_c_parallel(preds, y, votes, c_guo18, jobs)
            return solve_greedy_1(c, p)
        elif(metric == "margineantu1997"):
            return solve_greedy_2(preds, q_margineantu97, y, p, "minimize")
        elif(metric == "cavalcanti2016"):
            return solve_greedy_2(preds, q_cavalcanti16, y, p, "minimize")
        elif(metric == "lazarevic2001"):
            return solve_greedy_2(preds, q_lazarevic01, y, p, "minimize")
        elif(metric == "zhang2006"):
            c = fill_c_parallel(preds, y, votes, c_zhang06, jobs)
            return solve_greedy_3(preds, c, q_zhang06, y, p, "minimize")
        elif(metric == "kappaplusaccuracy"):
            c = fill_c_parallel(preds, y, votes, c_accuracy, jobs)
            c = (-1) * c
            return solve_greedy_3(preds, c, q_margineantu97, y, p, "minimize")
        elif(metric == "accuracy"):
            c = fill_c_parallel(preds, y, votes, c_accuracy, jobs)
            return solve_greedy_1(c, p)
        elif(metric == "auc"):
            c = fill_c_parallel(preds, y, votes, c_auc, jobs)
            return solve_greedy_1(c, p)
        else:
            print("error pruning metric")
            return
        
    elif(optimizer == "miqp"):
        if(metric == "cavalcanti2016"):
            Q = fill_q_parallel(preds, y, q_cavalcanti16, jobs)
            return solve_miqp_2(Q, p)
        elif(metric == "zhang2006"):
            c = fill_c_parallel(preds, y, votes, c_zhang06, jobs)
            Q = fill_q_parallel(preds, y, q_zhang06, jobs)
            return solve_miqp_3(c, Q, p)
        elif(metric == "guo2018"):
            c = fill_c_parallel(preds, y, votes, c_guo18, jobs)
            c = (-1) * c
            return solve_miqp_1(c, p)
        elif(metric == "lu2010"):
            c = fill_c_parallel(preds, y, votes, c_lu10, jobs)
            c = (-1) * c
            return solve_miqp_1(c, p)
        elif(metric == "margineantu1997"):
            Q = fill_q_parallel(preds, y, q_margineantu97, jobs)
            return solve_miqp_2(Q, p)
        elif(metric == "lazarevic2001"):
            Q = fill_q_parallel(preds, y, q_lazarevic01, jobs)
            return solve_miqp_2(Q, p)
        elif(metric == "accuracy"):
            c = fill_c_parallel(preds, y, votes, c_accuracy, jobs)
            c = (-1) * c
            return solve_miqp_1(c, p)
        elif(metric == "auc"):
            c = fill_c_parallel(preds, y, votes, c_auc, jobs)
            c = (-1) * c
            return solve_miqp_1(c, p)
        elif(metric == "kappaplusaccuracy"):
            Q = fill_q_parallel(preds, y, q_margineantu97, jobs)
            c = fill_c_parallel(preds, y, votes, c_accuracy, jobs)
            c = (-1) * c
            return solve_miqp_3(c, Q, p)
        else:
            print("error pruning metric")
            return
            
    elif(optimizer == "rand"):
        return solve_random(len(preds), p)
    else:
        print("error approximation method")
        return
    