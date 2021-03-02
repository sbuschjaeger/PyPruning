import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, cohen_kappa_score

# TODO RENAME ACCURACY AND AUC

# accuracy for ensemble pruning
# Uses the sklearn-implementation
# def accuracy(iproba, jproba, target):
#     return - 1.0 * accuracy_score(target, iproba.argmax(axis=1))

def error(iproba, jproba, target):
    return (iproba.argmax(axis=1) != target).mean()

# Area under the ROC-Curve (AUC) metric for ensemble pruning
# Uses the sklearn-implementation
def neg_auc(iproba, jproba, target):
    if(iproba.shape[1] == 2):
        return - 1.0 * roc_auc_score(target, iproba)
    else:
        return - 1.0 * roc_auc_score(target, iproba, multi_class="ovr")

# Paper:   Ensemble Pruning via individual contribution
# Authors: Lu et al. 2010
#
def individual_contribution(iproba, jproba, target):
    IC = 0
    #V = all_proba.argmax(axis=2)
    #predictions = iproba.argmax(axis=1)
    #V = all_proba.sum(axis=0)#.argmax(axis=1)
    V = jproba
    predictions = iproba.argmax(axis=1)

    for j in range(len(predictions)):
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
    return IC

# Paper:   Margin & Diversity based ordering ensemble pruning
# Authors: Guo et al. 2018
#
def margin_diversity(iproba, jproba, target, alpha = 0.2):
    predictions = iproba.argmax(axis=1)
    V = jproba #all_proba.sum(axis=0)#.argmax(axis=1)
    MDM = 0
    #V = all_proba.argmax(axis=1)
    #predictions = iproba.argmax(axis=1)
    n = len(jproba)
    
    for j in range(len(predictions)):
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


# Paper  : Pruning adaptive boosting
# Authors: Margineantu et Dietterich 1997 
# Uses the sklearn-implementation of the "Kappa-Statistic" measure
def kappa_statistic(iproba, jproba, target):
    iproba = iproba.argmax(axis=1)
    jproba = jproba.argmax(axis=1)
    #np.argmax(all_proba, axis=2)
    return cohen_kappa_score(iproba, jproba)

# Paper  : Combining diversity measures for ensemble pruning
# Authors: Cavalcanti et al. 2016
#
def combined(iproba, jproba, target):
    #iproba = np.argmax(iproba, axis=1)
    #all_proba = np.argmax(all_proba, axis=1)
    iproba = iproba.argmax(axis=1)
    jproba = jproba.argmax(axis=1)

    m = len(iproba)
    a = 0   #h1 and h2 correct
    b = 0   #h1 correct, h2 incorrect
    c = 0   #h1 incorrect, h2 correct
    d = 0   #h1 and h2 incorrect
    
    for j in range(m):
        if(iproba[j] == target[j] and jproba[j] == target[j]):
            a = a + 1
        elif(iproba[j] == target[j] and jproba[j] != target[j]):
            b = b + 1
        elif(iproba[j] != target[j] and jproba[j] == target[j]):
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
        correlation = ((a*d) - (b*c)) / np.sqrt( (a+b)*(a+c)*(c+d)*(b+d) )
        
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

# Paper:   Effective pruning of neural network classifier ensembles
# Authors: Lazarevic et al. 2001
#
def disagreement(iproba, jproba, target):
    iproba = iproba.argmax(axis=1)
    jproba = jproba.argmax(axis=1)

    kappa = cohen_kappa_score(iproba, jproba)
    
    #calculate correlation coefficient
    m = len(iproba)
    a = 0   #h1 and h2 correct
    b = 0   #h1 correct, h2 incorrect
    c = 0   #h1 incorrect, h2 correct
    d = 0   #h1 and h2 incorrect
    
    for j in range(m):
        if(iproba[j] == target[j] and jproba[j] == target[j]):
            a = a + 1
        elif(iproba[j] == target[j] and jproba[j] != target[j]):
            b = b + 1
        elif(iproba[j] != target[j] and jproba[j] == target[j]):
            c = c + 1
        else:
            d = d + 1
    
    # rare case: division by 0
    if((a+b)*(a+c)*(c+d)*(b+d) == 0):
        correlation = (a*d) - (b*c) / 0.001
    else:
        correlation = ((a*d) - (b*c)) / np.sqrt( (a+b)*(a+c)*(c+d)*(b+d) )
    
    # equally weighted (no further explanations)
    return (kappa + correlation) / 2

# Paper  : Ensemble Pruning Via Semi-definite Programming 
# Authors: Zhang et al 2006 
#
# def error(iproba, all_proba, target):
#     iproba = np.argmax(iproba, axis=1)
#     return (iproba != target).mean()

def q_zhang06(iproba, jproba, target):
    iproba = iproba.argmax(axis=1)
    jproba = jproba.argmax(axis=1)

    count_h1h2 = 0.0
    count_h1 = 0.0
    count_h2 = 0.0
    
    for j in range(len(iproba)):
        if(iproba[j] != target[j]):
            count_h1 = count_h1 + 1
        if(jproba[j] != target[j]):
            count_h2 = count_h2 + 1
        if (iproba[j] != target[j] and jproba[j] != target[j]):
            count_h1h2 = count_h1h2 + 1
    
    # avoid rare error: division by 0
    # if one of these cases occur, count_h1h2 will be 0 aswell
    # thus count_h1/h2 = 1 wont matter
    if(count_h1 == 0):
        count_h1 = 1.0
    if(count_h2 == 0):
        count_h2 = 1.0
    
    return ( (count_h1h2 / count_h1 ) + (count_h1h2 / count_h2) ) / 2.0