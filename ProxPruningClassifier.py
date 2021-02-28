import numpy as np
from joblib import Parallel, delayed
import copy

from scipy.special import softmax

from OnlineLearner import OnlineLearner

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

# See https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf for details
def to_prob_simplex(x):
    sorted_x = np.sort(x)
    x_sum = sorted_x[0]
    l = 1.0 - sorted_x[0]
    for i in range(1,len(sorted_x)):
        x_sum += sorted_x[i]
        tmp = 1.0 / (i + 1.0) * (1.0 - x_sum)
        if (sorted_x[i] + tmp) > 0:
            l = tmp 
    
    return [max(xi + l, 0.0) for xi in x]

'''
'''
class ProxPruningClassifier(OnlineLearner):
    """
    The ProximalPruningClassifier. This classifier expects a list of classifiers (=ensemble members) and prunes these to find the smallest and most effective ensemble. The list of classifiers is assumed to be objects which offer a `predict_proba(X)` method, where X are the test examples. 

    Each member is assigned a weight and proximal stochastic gradient descent is performed on these weights. To ensure an effective ensemble, a `loss` function is minimized. To ensure small ensembles, two regularizer are used:
         - `ensemble_regularizer`: This regularizer tries to remove as many members as possible from the ensemble
         - `tree_regularizer`: This regularizer tries to choose smaller trees with fewer nodes over larger ones
    Moreover, to further combat overfitting the weights of the ensemble can be constraint to sum to 1 via `normalize_weights`. 
    For faster training `fast_fit` can be enabled, which stores all predictions of all estimators on all training examples and only references those. This increases the memory usage but only calls `predict_proba` of each base model once, instead for each new batch during SGD.  

    Attributes
    ----------
    step_size : float
        The step_size used for stochastic gradient descent for opt 
    loss : str
        The loss function for training. Should be one of {{"mse", "cross-entropy", "hinge2"}}
    normalize_weights : bool
        True if nonzero weights should be projected onto the probability simplex, that is they should sum to 1. 
    init_weight : float
        Initial value for the weights.
    ensemble_regularizer : str
        The ensemble_regularizer. Should be one of {{None, "L0", "L1", "hard-L1"}}
    l_ensemble_reg : float
        The ensemble_regularizer regularization strength. 
    tree_regularizer : str
        The tree_regularizer. Should be one of {{None,"node"}}
    l_tree_reg : float
        The tree_regularizer regularization strength. 
    n_jobs : int
        The number of parallel jobs used for calling predict_proba of each indiviual member. 
    fast_fit : bool
        True, if fast_fit method should be used. This method computes all predictions of all estimators on all training examples once and stores them. 
        This increases memory consumption, but circumenvents multiple calls of predict_proba for the same example.
    estimators_ : list of objects
        The list of estimators which are used to built the ensemble. Each estimator must offer a predict_proba method.
    estimator_weights_ : np.array of floats
        The list of weights corresponding to their respective estimator in self.estimators_. 

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes

    """

    def __init__(self,
                estimators,
                #base_estimator,
                loss = "cross-entropy",
                step_size = 1e-1,
                ensemble_regularizer = None,
                l_ensemble_reg = 0,  
                tree_regularizer = None,
                l_tree_reg = 0,
                normalize_weights = False,
                init_weight = 0,
                n_jobs = 1,
                fast_fit = True,
                *args, **kwargs
                ):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert ensemble_regularizer is None or ensemble_regularizer in ["none","L0", "L1", "hard-L1"], "Currently only {{none,L0, L1, hard-L1}} the ensemble regularizer is supported"
        assert isinstance(estimators, (np.ndarray, list)), "Estimators should be a list of estimators from which the ensemble should be selected. Each object in this list must provide a `predict_proba` method. "
        #assert base_estimator is not None, "base_estimator must be a valid base model to be fitted"
        #assert isinstance(base_estimator, (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)), "Only the following base_estimators are currently supported {{RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier}}"
        assert l_tree_reg >= 0, "l_tree_reg must be greate or equal to 0"
        assert tree_regularizer is None or tree_regularizer in ["node"], "Currently only {{none, node}} regularizer is supported for tree the regularizer."

        if ensemble_regularizer == "hard-L1" and l_ensemble_reg < 1:
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is hard-L1. In this mode, l_ensemble_reg should be an integer 1 <= l_ensemble_reg <= max_trees where max_trees is the number of estimators trained by base_estimator!".format(l_ensemble_reg))

        if (l_ensemble_reg > 0 and (ensemble_regularizer == "none" or ensemble_regularizer is None)):
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is None. Ignoring l_ensemble_reg!".format(l_ensemble_reg))
            l_ensemble_reg = 0
            
        if (l_ensemble_reg == 0 and (ensemble_regularizer != "none" and ensemble_regularizer is not None)):
            print("WARNING: You set l_ensemble_reg to 0, but choose regularizer {}.".format(ensemble_regularizer))

        super().__init__(*args, **kwargs)

        #self.base_estimator = copy.deepcopy(base_estimator)
        self.step_size = step_size
        self.loss = loss
        self.normalize_weights = normalize_weights
        self.init_weight = init_weight
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.tree_regularizer = tree_regularizer
        self.l_tree_reg = l_tree_reg
        self.n_jobs = n_jobs
        self.fast_fit = fast_fit
        self.estimators_ = estimators
        self.estimator_weights_ = np.ones((len(self.estimators_), )) * self.init_weight
        
    
    def _individual_proba(self, X):
        assert self.estimators_ is not None, "Call fit before calling predict_proba!"

        def single_predict_proba(h,X):
            return h.predict_proba(X)

        all_proba = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(single_predict_proba) (h,X) for h in self.estimators_
        )
        all_proba = np.array(all_proba)
        return all_proba

    def _combine_proba(self, all_proba):
        scaled_prob = np.array([w * p for p,w in zip(all_proba, self.estimator_weights_)])

        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict_proba(self, X):
        if (len(self.estimators_)) == 0:
            return np.zeros((X.shape[0], self.n_classes_))
        else:
            all_proba = self._individual_proba(X)
            return self._combine_proba(all_proba)

    def next(self, data, target, train = False, new_epoch = False):
        # Check training mode. fast_fit computes all predictions for all trees during fit() stores these only once. 
        # This is faster, but uses more memory. During fast_fit we use indices to access the correct prediction 
        # for each tree instead of the examples directly (see fit() for details).
        # The normal approach calls fit for each individual tree on each batch, which uses fewer resources but calls
        # fit more often.
        if self.fast_fit and train:
            train_idx = [i[0] for i in data]
            all_proba = self.train_preds[:,train_idx,:]
        else:
            all_proba = self._individual_proba(data)
        
        output = self._combine_proba(all_proba)

        # Compute the appropriate loss. 
        if self.loss == "mse":
            target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
            loss = (output - target_one_hot) * (output - target_one_hot)
            loss_deriv = 2 * (output - target_one_hot)
        elif self.loss == "cross-entropy":
            target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
            p = softmax(output, axis=1)
            loss = -target_one_hot*np.log(p + 1e-7)
            m = target.shape[0]
            loss_deriv = softmax(output, axis=1)
            loss_deriv[range(m),target_one_hot.argmax(axis=1)] -= 1
        elif self.loss == "hinge2":
            target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(self.n_classes_)] for y in target] )
            zeros = np.zeros_like(target_one_hot)
            loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
            loss_deriv = - 2 * target_one_hot * np.maximum(1.0 - target_one_hot * output, zeros) 
        else:
            raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.loss)
        
        # Compute the appropriate ensemble_regularizer
        if self.ensemble_regularizer == "L0":
            loss = np.mean(loss) + self.l_ensemble_reg * np.linalg.norm(self.estimator_weights_,0)
        elif self.ensemble_regularizer == "L1":
            loss = np.mean(loss) + self.l_ensemble_reg * np.linalg.norm(self.estimator_weights_,1)
        else:
            loss = np.mean(loss) 
        
        # Compute the appropriate tree_regularizer
        if self.tree_regularizer == "node":
            loss += self.l_tree_reg * np.sum( [ (w * est.tree_.node_count) for w, est in zip(self.estimator_weights_, self.estimators_)] )
        
        if train:
            # Compute the gradient for the loss
            directions = np.mean(all_proba*loss_deriv,axis=(1,2))

            # Compute the gradient for the tree regularizer
            if self.tree_regularizer:
                node_deriv = self.l_tree_reg * np.array([ est.tree_.node_count for est in self.estimators_])
            else:
                node_deriv = 0

            # Perform the gradient step. Note that L0 / L1 regularizer is performed via the prox operator 
            # and thus performed _after_ this update.
            tmp_w = self.estimator_weights_ - self.step_size*directions - self.step_size*node_deriv
            
            # Compute the prox step. 
            if self.ensemble_regularizer == "L0":
                tmp = np.sqrt(2 * self.l_ensemble_reg * self.step_size)
                tmp_w = np.array([0 if abs(w) < tmp else w for w in tmp_w])
                
                #nonzero_idx = np.nonzero(tmp_w)
                #self.estimator_weights_ = [0 if abs(w) < tmp else w for w in tmp_w]
            elif self.ensemble_regularizer == "L1":
                sign = np.sign(tmp_w)
                tmp_w = np.abs(tmp_w) - self.step_size*self.l_ensemble_reg
                tmp_w = sign*np.maximum(tmp_w,0)
                #self.estimator_weights_ = sign*np.maximum(tmp_w,0)
            elif self.ensemble_regularizer == "hard-L1":
                top_K = np.argsort(tmp_w)[-self.l_ensemble_reg:]
                tmp_w = np.array([w if i in top_K else 0 for i,w in enumerate(tmp_w)])

                # keep K (=l_reg) largest values, then project
                #top_K = np.argpartition(self.estimator_weights_, -self.l_reg)[-self.l_reg:]
                # top_w = tmp_w[top_K]
                # top_w = to_prob_simplex(top_w)
                # self.estimator_weights_ = np.zeros((len(self.estimator_weights_)))
                # for i,w in zip(top_K, top_w):
                #     self.estimator_weights_[i] = w
                #self.estimator_weights_ = [self.estimator_weights_[i] if i in top_K else 0 for i in range(len(self.estimator_weights_))]
            
            # If set, normalize weights. Note that we use the support of tmp_w for the projection onto the probability simplex
            # as described in http://proceedings.mlr.press/v28/kyrillidis13.pdf
            # Thus, we first need to extract the nonzero weights, project these and then copy them back into corresponding array
            if self.normalize_weights:
                nonzero_idx = np.nonzero(tmp_w)[0]
                nonzero_w = tmp_w[nonzero_idx]
                nonzero_w = to_prob_simplex(nonzero_w)
                self.estimator_weights_ = np.zeros((len(self.estimator_weights_)))
                for i,w in zip(nonzero_idx, nonzero_w):
                    self.estimator_weights_[i] = w
            else:
                self.estimator_weights_ = tmp_w
                
            n_updates = 1
        else:
            n_updates = 0
            
        return {"loss": loss, "num_trees": self.num_trees(), "num_parameters":self.num_parameters()}, output, n_updates

    def num_trees(self):
        return np.count_nonzero(self.estimator_weights_)

    def num_parameters(self):
        return sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(self.estimator_weights_, self.estimators_)] )

    def fit(self, X, y, sample_weight = None):
        #model = self.base_estimator.fit(X, y, sample_weight)

        #self.estimators_ = model.estimators_
        #self.estimator_weights_ = np.ones((len(self.estimators_), )) * self.init_weight
        
        # If fast_fit is enabled we will store all predictions of all trees on all training examples in self.train_preds
        # To access the correct predictions for the current batch in next() fit() with indices for each example and
        # use them to access self.train_preds appropriately.   
        if self.fast_fit:
            Xtmp = np.array([[i] for i in range(X.shape[0])])
            self.train_preds = self._individual_proba(X)
            super().fit(Xtmp, y, sample_weight)
        else:
            super().fit(X, y, sample_weight)
