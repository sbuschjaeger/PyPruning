import numpy as np
from joblib import Parallel, delayed
import numbers
import time
from sklearn import ensemble
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score

from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier

from .PruningClassifier import PruningClassifier

def create_mini_batches(inputs, targets, data, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Shamelessly copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
    data : array-like vector or matrix 
        The data to be iterated in mini batches
    batch_size : int
        The mini batch size
    shuffle : bool, default False
        If True shuffle the batches 
    """
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        start_idx += batch_size

        yield inputs[excerpt], targets[excerpt], data[excerpt]

def to_prob_simplex(x):
    """ Projects the given vector to the probability simplex so that :math:`\\sum_{i=1}^k x_i = 1, x_i \\in [0,1]`. 

    Reference
        Weiran Wang and Miguel A. Carreira-Perpinan (2013) Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application. https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf

    Parameters
    ----------
    x : array-like vector with k entries
        The vector to be projected.

    Returns
    -------
    u : array-like vector with k entries
        The projected vector.

    """
    if x is None or len(x) == 0:
        return x
    u = np.sort(x)[::-1]

    l = None
    u_sum = 0
    for i in range(0,len(u)):
        u_sum += u[i]
        tmp = 1.0 / (i + 1.0) * (1.0 - u_sum)
        if u[i] + tmp > 0:
            l = tmp
    
    projected_x = [max(xi + l, 0.0) for xi in x]
    return projected_x

class ProxPruningClassifier(PruningClassifier):
    """ (Heterogeneous) Pruning via Proximal Gradient Descent
    
    This pruning method directly minimizes a constrained loss function :math:`L` including a regularizer :math:`R_1` via (stochastic) proximal gradient descent. There are two sets of constraints available. When soft constraints are used, then the following function is minimized

    .. math::

        \\arg\\min_w L \\left(\sum_{i=1}^M w_i h_i(x), y\\right) + \\lambda_1 \\sum_{i=1}^K w_i R_1(h_i) + \\lambda_2 R_2(w)
    
    When hard constraints are used, then the following objective is minimized

    .. math::

        \\arg\\min_w L \\left(\sum_{i=1}^M w_i h_i(x), y\\right) + \\lambda_1 \\sum_{i=1}^K w_i R_1(h_i) \\text{ s.t. } R_2(w) \le \\lambda_2

    The regularizer :math:`R_1` is used to select smaller trees, whereas the regularizer :math:`R_2` is used to select fewer trees from the ensemble.

    ----------
    step_size : float, default is 0.1
        The step_size used for stochastic gradient descent for opt 
    loss : str, default is ``"mse"``
        The loss function for training. Should be one of ``{"mse", "cross-entropy", "hinge2"}``. 

        - ``"mse"``: :math:`L(f(x),y) = \\sum_{i=1}^C (f(x)_i - y_i)^2`
        - ``"cross-entropy"``: :math:`L(f(x),y) = \\sum_{i=1}^C y_i \\log(s(f(x))_i)`, where :math:`s` is the softmax function.
        - ``"hinge2"``: :math:`L(f(x),y) = \\sum_{i=1}^C \\max(0, 1 - y_i \\cdot f(x)_i )^2`
    normalize_weights : boolean, default is True
        True if nonzero weights should be projected onto the probability simplex, that is they should sum to 1. 
    ensemble_regularizer : str or None, default is ``"hard-L0"``
        The ensemble_regularizer :math:`R_2`. This regularizer is used to select fewer members from the ensembles. It should be one of ``{None, "L0", "L1", "hard-L0"}``

        - ``None``: No constraints are applied during ensemble selection.
        - ``"L0"``: Apply :math:`R_2(w) = || w ||_0` regularization (implemented via ``numpy.linalg.norm`` ). The regularization strength :math:`\lambda_2` scales the regularizer in this case.
        - ``"L1"``: Apply :math:`R_2(w) = || w ||_1` regularization (implemented via ``numpy.linalg.norm`` ). The regularization strength :math:`\lambda_2` scales the regularizer in this case.
        - ``"hard-L0"``: Apply :math:`R_2(w) = || w ||_0 \\le \\lambda_2` regularization. This is the "hard" version of the ``L0`` regularization. The regularization strength :math:`\\lambda_2` is used a an upper bound in this case.

    l_ensemble_reg : float, default is 0
        The ``ensemble_regularizer`` regularization strength :math:`\\lambda_2`. If ``"L0"`` or ``"L1"`` is selected, then ``l_ensemble_reg`` is the regularization strength which scales the regularizer. If ``"hard-L0"`` is selected, then ``l_ensemble_reg`` is the maximum number of members in pruned ensemble.
    tree_regularizer : str or ``None``, default is ``"node"``
        The tree_regularizer :math:`R_1`. This regularizer is used to select smaller trees. Should be one of ``{None,"node"}``

        - ``None``: No constraints are applied during ensemble selection.
        - ``"node"``: Apply :math:`R_1(h_i) = n_i` regularization where :math:`n_i` is the number of nodes in the tree.

    l_tree_reg : float, default is 0
        The ``tree_regularizer`` regularization strength :math:`\\lambda_1`. The ``tree_regularizer`` is scaled by this value. 
    batch_size: int, default is 256
        The batch sized used for PSGD
    epochs : int, default is 1
        The number of epochs PSGD is run.
    verbose : boolean, default is False
        If true, shows a progress bar via tqdm and some statistics
    update_leaves : boolean, default is False
        If true, then leave nodes of each tree are also updated via PSGD.
    out_path: str or None, default is None
        If not None, then statistics are stored in a file called ``$out_path/epoch_$i.npy`` for epoch $i.
    """

    def __init__(self,
        loss = "cross-entropy",
        step_size = 1e-1,
        ensemble_regularizer = "hard-L0",
        l_ensemble_reg = 0,  
        tree_regularizer = "node",
        l_tree_reg = 0,
        normalize_weights = True,
        batch_size = 256,
        epochs = 1,
        verbose = False, 
        update_leaves = False,
        out_path = None,
        eval_every_epochs = None):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert ensemble_regularizer is None or ensemble_regularizer in ["none","L0", "L1", "hard-L0"], "Currently only {{none,L0, L1, hard-L0}} the ensemble regularizer is supported"
        assert l_tree_reg >= 0, "l_reg must be greater or equal to 0"
        assert tree_regularizer is None or tree_regularizer in ["node"], "Currently only {{none, node}} regularizer is supported for tree the regularizer."
        assert batch_size >= 1, "batch_size must be at-least 1"
        assert epochs >= 1, "epochs must be at-least 1"

        if ensemble_regularizer == "hard-L0":
            assert l_ensemble_reg >= 1 or l_ensemble_reg == 0, "You chose ensemble_regularizer = hard-L0, but set 0 < l_ensemble_reg < 1 which does not really makes sense. If hard-L0 is set, then l_ensemble_reg is the maximum number of estimators in the pruned ensemble, thus likely an integer value >= 1."

        super().__init__()
        
        self.loss = loss
        self.step_size = step_size
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.tree_regularizer = tree_regularizer
        self.l_tree_reg = l_tree_reg
        self.normalize_weights = normalize_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.update_leaves = update_leaves
        self.out_path = out_path
        self.eval_every_epochs = eval_every_epochs

    def next(self, proba, target, data):
        # If we update the leaves, then proba also changes and we need to recompute them. Otherwise we can just use the pre-computed probas
        if self.update_leaves:
            proba = self._individual_proba(data)
        else:
            proba = np.swapaxes(proba, 0, 1)

        output = np.array([w * p for w,p in zip(proba, self.weights_)]).sum(axis=0)

        batch_size = output.shape[0]
        accuracy = (output.argmax(axis=1) == target) * 100.0
        n_trees = [self.num_trees() for _ in range(batch_size)]
        n_param = [self.num_parameters() for _ in range(batch_size)]
        
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
        
        loss = np.sum(np.mean(loss,axis=1))
        
        if self.ensemble_regularizer == "L0":
            loss += self.l_ensemble_reg * np.linalg.norm(self.weights_,0)
        elif self.ensemble_regularizer == "L1":
            loss += self.l_ensemble_reg * np.linalg.norm(self.weights_,1)

        # Compute the gradients for the loss
        directions = np.mean(proba*loss_deriv,axis=(1,2))

        # Compute the appropriate regularizer
        if self.tree_regularizer == "node" and self.l_tree_reg > 0:
            loss += self.l_tree_reg * np.sum( [ (w * est.tree_.node_count) for w, est in zip(self.weights_, self.estimators_)] )
            
            node_deriv = self.l_tree_reg * np.array([ est.tree_.node_count for est in self.estimators_])
        else:
            node_deriv = 0

        # Perform the gradient step + projection 
        tmp_w = self.weights_ - self.step_size*directions - self.step_size*node_deriv
        
        if self.update_leaves:
            # compute direction per tree
            # tree_deriv = proba*loss_deriv
            for i, h in enumerate(self.estimators_):
                tree_grad = (self.weights_[i] * loss_deriv)[:,np.newaxis,:]
                # find idx
                idx = h.apply(data)
                h.tree_.value[idx] = h.tree_.value[idx] - self.step_size * tree_grad[:,:,h.classes_.astype(int)]
                # update model
                #h.tree_.value[idx] = h.tree_.value[idx] - self.step_size*h.tree_.value[idx]*tree_deriv[i,:,np.newaxis]
                
                #step = self.step_size*tree_deriv[i,:,np.newaxis]
                #h.tree_.value[idx] = h.tree_.value[idx] - step[:,:,self.classes_.astype(int)]

        if self.ensemble_regularizer == "L0":
            tmp = np.sqrt(2 * self.l_ensemble_reg * self.step_size)
            tmp_w = np.array([0 if abs(w) < tmp else w for w in tmp_w])
        elif self.ensemble_regularizer == "L1":
            sign = np.sign(tmp_w)
            tmp_w = np.abs(tmp_w) - self.step_size*self.l_ensemble_reg
            tmp_w = sign*np.maximum(tmp_w,0)
        elif self.ensemble_regularizer == "hard-L0":
            top_K = np.argsort(tmp_w)[-self.l_ensemble_reg:]
            tmp_w = np.array([w if i in top_K else 0 for i,w in enumerate(tmp_w)])

        # If set, normalize the weights. Note that we use the support of tmp_w for the projection onto the probability simplex
        # as described in http://proceedings.mlr.press/v28/kyrillidis13.pdf
        # Thus, we first need to extract the nonzero weights, project these and then copy them back into corresponding array
        
        if self.normalize_weights and len(tmp_w) > 0:
            nonzero_idx = np.nonzero(tmp_w)[0]
            nonzero_w = tmp_w[nonzero_idx]
            nonzero_w = to_prob_simplex(nonzero_w)
            self.weights_ = np.zeros((len(tmp_w)))
            for i,w in zip(nonzero_idx, nonzero_w):
                self.weights_[i] = w
        else:
            self.weights_ = tmp_w
        
        return {"loss":loss, "accuracy": accuracy, "num_trees": n_trees, "num_parameters" : n_param}

    def num_trees(self):
        """ Returns the number of nonzero weights """
        return np.count_nonzero(self.weights_)

    def num_parameters(self):
        """ Returns the total number of decision nodes across all trees of the entire ensemble for all trees with nonzero weight. """
        return sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(self.weights_, self.estimators_)] )

    def prune_(self, proba, target, data):
        proba = np.swapaxes(proba, 0, 1)
        self.weights_ = np.array([1.0 / proba.shape[1] for _ in range(proba.shape[1])])

        if self.update_leaves:
            # SKlearn stores the raw counts instead of probabilities. For SGD its better to have the 
            # probabilities for numerical stability. 
            # tree.tree_.value is not writeable, but we can modify the values inplace. Thus we 
            # use [:] to copy the array into the normalized array. Also tree.tree_.value has a strange shape (batch_size, 1, n_classes)
            for tree in self.estimators_:
                tree.tree_.value[:] = tree.tree_.value / tree.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis]

        for epoch in range(self.epochs):

            mini_batches = create_mini_batches(proba, target, data, self.batch_size, True) 

            times = []
            total_time = 0
            metrics = {}
            example_cnt = 0

            with tqdm(total=proba.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for batch in mini_batches:
                    bproba, btarget, bdata = batch 

                    # Update Model                    
                    start_time = time.time()
                    batch_metrics = self.next(bproba, btarget, bdata)
                    batch_time = time.time() - start_time

                    # Extract statistics
                    for key,val in batch_metrics.items():
                        metrics[key] = np.concatenate( (metrics.get(key,[]), val), axis=None )
                        metrics[key + "_sum"] = metrics.get( key + "_sum",0) + np.sum(val)

                    example_cnt += bproba.shape[0]
                    pbar.update(bproba.shape[0])
                    
                    # TODO ADD times to metrics and write it to disk
                    times.append(batch_time)
                    total_time += batch_time

                    m_str = ""
                    for key,val in metrics.items():
                        if "_sum" in key:
                            m_str += "{} {:2.4f} ".format(key.split("_sum")[0], val / example_cnt)
                    
                    desc = '[{}/{}] {} time_item {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        m_str,
                        total_time / example_cnt
                    )
                    pbar.set_description(desc)
                
                if self.eval_every_epochs is not None and epoch % self.eval_every_epochs == 0 and self.out_path is not None:
                    np.save(os.path.join(self.out_path, "epoch_{}.npy".format(epoch)), metrics, allow_pickle=True)
    
        return [i for i in range(len(self.weights_)) if self.weights_[i] > 0], [w for w in self.weights_ if w > 0]
