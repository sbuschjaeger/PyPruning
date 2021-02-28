import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score

from joblib import Parallel,delayed

from PruningClassifier import PruningClassifier

from Metrics import accuracy

class GreedyPruningClassifier(PruningClassifier):

    def __init__(self, 
        n_estimators = 5, 
        base_estimator = None, 
        single_metric = accuracy,
        pairwise_metric = None, 
        l_reg = 0, 
        n_jobs = 8):
        
        super().__init__(n_estimators, base_estimator, n_jobs)
        self.n_jobs = n_jobs
        self.single_metric = single_metric
        self.pairwise_metric = pairwise_metric
        self.l_reg = l_reg
        
        assert 0 <= l_reg <= 1, "l_reg should be from [0,1], but you supplied {}".format(l_reg)
        assert pairwise_metric is not None or single_metric is not None, "You did not provide a single_metric or pairwise_metric. Please provide at-least one of them"

        if single_metric is None and l_reg < 1:
            print("Warning: You did not provide a single_metrics, but set l_reg < 1. This does not make sense. Setting l_reg = 1 for you.")

        if pairwise_metric is None and l_reg > 0:
            print("Warning: You did not provide a pairwise_metric, but set l_reg > 0. This does not make sense. Setting l_reg = 0 for you.")

    def prune_(self, proba, target):
        n_received = len(proba)
        if self.n_estimators >= n_received:
            return range(0, n_received)

        if self.l_reg < 1:
            single_scores = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.single_metric) (proba[i,:], proba, target) for i in range(n_received)
            )
            single_scores = np.array(single_scores)

        if self.l_reg > 0:
            best_model = np.argmax(single_scores)
            not_seleced_models = range(n_received)
            not_seleced_models.remove(best_model)
            selected_models = [ best_model ]

            for _ in range(self.n_estimators - 1):
                selected_proba = proba[selected_models,:]
                scores = []
                for i in not_seleced_models:
                    pairwise_score = self.l_reg * self.pairwise_metric(proba[i, :], selected_proba, target)
                    if self.l_reg < 1:
                        pairwise_score += (1-self.l_reg) * single_scores[i]
                    
                    scores.append( pairwise_score ) 
                best_model = np.argmax(scores)
                not_seleced_models.remove(best_model)
                selected_models.append(best_model)
            
            return selected_models
        else:
            # this should work?
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array#comment42801155_23734295
            return np.argpartition(single_scores, -self.n_estimators)[-self.n_estimators:]
        
        # zip and sort
        # indicies = np.arange(len(c))
        # zipped = zip(indicies, c)
        # zipped = sorted(zipped, key = lambda t: t[1])
        # indicies, values = zip(*zipped)
        
        # #select top k classifiers
        # sol = []

        # for i in range(int(len(c)*p)):
        #     sol.append(indicies[len(indicies)-(i+1)])
        # return sol