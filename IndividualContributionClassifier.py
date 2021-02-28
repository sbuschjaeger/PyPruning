import numpy as np

from PruningClassifier import PruningClassifier

class IndividualContributionClassifier(PruningClassifier):

    def __init__(self, n_estimators = 5, base_estimator = None):
        super().__init__(n_estimators, base_estimator)

    def fill_c_parallel(predictions, y, V, metric, jobs):
        n_est = len(predictions)
        c = Parallel(n_jobs=jobs, backend="threading")(
                delayed(metric) (predictions[i], y, n_est, V) for i in range(n_est)
            )
        return np.array(c)

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



    def prune_(self, X, estimators):
