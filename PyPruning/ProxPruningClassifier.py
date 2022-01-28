from asyncio import base_tasks
from pickletools import optimize
import numpy as np
from joblib import Parallel, delayed
import numbers
import time
from tqdm import tqdm
import os

from scipy.special import softmax

from .PruningClassifier import PruningClassifier

def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Shamelessly copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
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

        yield inputs[excerpt], targets[excerpt]

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

def node_regularizer(est):
    """ Extract the number of nodes in the given tree 

    Parameters
    ----------
    X : numpy matrix
        A (N, d) matrix with the datapoints used for pruning where N is the number of data points and d is the dimensionality
    
    Y : numpy array / list of ints
        A numpy array or list of N integers where each integer represents the class for each example. Classes should start with 0, so that for C classes the integer 0,1,...,C-1 are used

    est: object
        Estimator for which the regularizer is computed.

    Returns
    -------
    u : float / int scalar
        The computed regularizer
    """
    return est.tree_.node_count

def avg_path_len_regularizer(est):
    """ Extract the number of nodes in the given tree 

    Parameters
    ----------
    X : numpy matrix
        A (N, d) matrix with the datapoints used for pruning where N is the number of data points and d is the dimensionality
    
    Y : numpy array / list of ints
        A numpy array or list of N integers where each integer represents the class for each example. Classes should start with 0, so that for C classes the integer 0,1,...,C-1 are used

    est: object
        Estimator for which the regularizer is computed.

    Notes
    -----
    Thanks to Mojtaba Masoudinejad (mojtaba.masoudinejad@tu-dortmund.de) for the implementation

    Returns
    -------
    u : float / int scalar
        The computed regularizer
    """
    # %% Identify all child-parent relations
    # ----- read data from the tree
    n_nodes = est.tree_.node_count
    children_left = est.tree_.children_left
    children_right = est.tree_.children_right
    samples = est.tree_.n_node_samples
    weighted_n_node_samples = est.tree_.weighted_n_node_samples
    impurity = est.tree_.impurity
    total_sum_weights = weighted_n_node_samples[0]

    # ----- Initial empty variables
    is_leaves = np.zeros(shape = n_nodes, dtype = bool)
    node_parent = np.zeros(shape = n_nodes, dtype = np.int64) # Parent node ID
    r_node = np.zeros(shape = n_nodes, dtype = np.float64)

    # ----- Initialize the stack
    stack = [(0, -1)]  # [node id, parent id] root node has no parent => -1

    # ----- Go downward to identify child/parent and leaf status
    while len(stack) > 0:    
        # check each node only once, using pop
        node_id, parent  = stack.pop()
        node_parent[node_id] = parent
        
        # children of a node are different
        is_split_node = children_left[node_id] != children_right[node_id]
        
        # add data of split point to the stack to go through
        if is_split_node:
            child_l_id = children_left[node_id]
            child_r_id = children_right[node_id]
            stack.append((child_l_id, node_id))
            stack.append((child_r_id, node_id))
        else:
            is_leaves[node_id] = True

    # %% Identify all branches and probabilistic cost of each split node(branch)
    node_cost = np.zeros(shape = n_nodes, dtype = np.float64)

    for node_ind in range (n_nodes): # for all nodes
        r_node[node_ind] = impurity[node_ind]
        # r_node[node_ind] = (weighted_n_node_samples[node_ind] * impurity[node_ind] / total_sum_weights)
            
        if is_leaves[node_ind]:
            current_parent = node_parent[node_ind]
            upward_length = 1
            while current_parent != -1:
                node_cost[current_parent] = node_cost[current_parent] + upward_length * samples[node_ind]/samples[current_parent]
                current_parent = node_parent[current_parent] # get the next parent node id
                upward_length = upward_length + 1

    return node_cost[0]

def loss_and_deriv(loss_type, output, target):
    n_classes = output.shape[1]

    if loss_type == "mse":
        target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )
        loss = (output - target_one_hot) * (output - target_one_hot)
        loss_deriv = 2 * (output - target_one_hot)
    elif loss_type == "cross-entropy":
        target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )
        p = softmax(output, axis=1)
        loss = -target_one_hot*np.log(p + 1e-7)
        m = target.shape[0]
        loss_deriv = softmax(output, axis=1)
        loss_deriv[range(m),target_one_hot.argmax(axis=1)] -= 1
    elif loss_type == "hinge2":
        target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(n_classes)] for y in target] )
        zeros = np.zeros_like(target_one_hot)
        loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
        loss_deriv = - 2 * target_one_hot * np.maximum(1.0 - target_one_hot * output, zeros) 
    else:
        raise ValueError("Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(loss_type))
    
    return loss, loss_deriv

def prox(w, prox_type, normalize, l_reg, step_size):
    if prox_type == "L0":
        tmp = np.sqrt(2 * l_reg * step_size)
        tmp_w = np.array([0 if abs(wi) < tmp else wi for wi in w])
    elif prox_type == "L1":
        sign = np.sign(w)
        tmp_w = np.abs(w) - l_reg*step_size
        tmp_w = sign*np.maximum(tmp_w,0)
    elif prox_type == "hard-L0":
        top_K = np.argsort(w)[-l_reg:]
        tmp_w = np.array([wi if i in top_K else 0 for i,wi in enumerate(w)])

    # If set, normalize the weights. Note that we use the support of tmp_w for the projection onto the probability simplex
    # as described in http://proceedings.mlr.press/v28/kyrillidis13.pdf
    # Thus, we first need to extract the nonzero weights, project these and then copy them back into corresponding array
    if normalize and len(tmp_w) > 0:
        nonzero_idx = np.nonzero(tmp_w)[0]
        nonzero_w = tmp_w[nonzero_idx]
        nonzero_w = to_prob_simplex(nonzero_w)
        new_w = np.zeros((len(tmp_w)))
        for i,wi in zip(nonzero_idx, nonzero_w):
            new_w[i] = wi
        return new_w
    else:
        return tmp_w

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
    loss : str, default is ``"mse"``
        The loss function for training. Should be one of ``{"mse", "cross-entropy", "hinge2"}``. 

        - ``"mse"``: :math:`L(f(x),y) = \\sum_{i=1}^C (f(x)_i - y_i)^2`
        - ``"cross-entropy"``: :math:`L(f(x),y) = \\sum_{i=1}^C y_i \\log(s(f(x))_i)`, where :math:`s` is the softmax function.
        - ``"hinge2"``: :math:`L(f(x),y) = \\sum_{i=1}^C \\max(0, 1 - y_i \\cdot f(x)_i )^2`
    step_size : float, default is 0.1
        The step_size used for stochastic gradient descent for opt 
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
    tree_regularizer : function or ``None``, default is ``node_regularizer``
        The tree_regularizer :math:`R_1`. This regularizer is used to select smaller trees. This should be `None` or a function which returns the regularizer given a single tree. 
    l_tree_reg : float, default is 0
        The ``tree_regularizer`` regularization strength :math:`\\lambda_1`. The ``tree_regularizer`` is scaled by this value. 
    batch_size: int, default is 256
        The batch sized used for PSGD. Use 0 for the entire dataset per batch which leads to Prox Gradient Descent.
    epochs : int, default is 1
        The number of epochs PSGD is run.
    verbose : boolean, default is False
        If true, shows a progress bar via tqdm and some statistics
    out_path: str or None, default is None
        If not None, then statistics are stored in a file called ``$out_path/epoch_$i.npy`` for epoch $i.
    """

    def __init__(self,
        loss = "cross-entropy",
        step_size = 1e-1,
        ensemble_regularizer = "hard-L0",
        l_ensemble_reg = 0,  
        regularizer = node_regularizer,
        l_reg = 0,
        normalize_weights = True,
        batch_size = 256,
        epochs = 1,
        verbose = False, 
        optimizer = "adam"
        ):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported, but you gave {}".format(loss)
        assert ensemble_regularizer is None or ensemble_regularizer in ["none","L0", "L1", "hard-L0"], "Currently only {{none,L0, L1, hard-L0}} the ensemble regularizer is supported, but you gave {}".format(ensemble_regularizer)
        assert l_reg >= 0, "l_reg must be greater or equal to 0, but you gave {}".format(l_reg)
        assert batch_size >= 0, "batch_size must be >= 0, but you gave {}".format(batch_size)
        assert epochs >= 1, "epochs must be at-least 1, but you gave {}".format(epochs)
        assert optimizer in ["adam", "sgd"], "Optmizer must be on of {{adam, sgd}} but you gave {}".format(optimizer)

        if ensemble_regularizer == "hard-L0":
            assert l_ensemble_reg >= 1 or l_ensemble_reg == 0, "You chose ensemble_regularizer = hard-L0, but set 0 < l_ensemble_reg < 1 which does not really makes sense. If hard-L0 is set, then l_ensemble_reg is the maximum number of estimators in the pruned ensemble, thus likely an integer value >= 1."

        super().__init__()
        
        self.loss = loss
        self.lr = step_size
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.regularizer = regularizer
        self.l_reg = l_reg
        self.normalize_weights = normalize_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = optimizer

    def num_estimators(self):
        """ Returns the number of nonzero weights """
        return np.count_nonzero(self.weights_)

    # def num_parameters(self):
    #     """ Returns the total number of decision nodes across all estimators of the entire ensemble for all estimators with nonzero weight. """
    #     return sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(self.weights_, self.estimators_)] )

    def prune_(self, proba, target, data):
        proba = np.swapaxes(proba, 0, 1)
        self.weights_ = np.array([1.0 / proba.shape[1] for _ in range(proba.shape[1])])

        if self.regularizer is not None and self.l_reg > 0:
            self.regularizer_evals = np.array([ self.regularizer(est) for est in self.estimators_])
        
        if self.batch_size == 0:
            self.batch_size = proba.shape[0]

        if self.optimizer == "adam":
            m = np.zeros_like(self.weights_)
            v = np.zeros_like(self.weights_)
            t = 1

        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(proba, target, self.batch_size, True) 

            batch_cnt = 0
            time_sum = 0
            loss_sum = 0
            n_estimators_sum = 0
            accuracy_sum = 0

            with tqdm(total=proba.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for batch in mini_batches:
                    bproba, btarget = batch 

                    # Update Model                    
                    start_time = time.time()

                    bproba = np.swapaxes(bproba, 0, 1)
                    output = np.array([w * p for w,p in zip(bproba, self.weights_)]).sum(axis=0)

                    # Compute the appropriate loss and its derivative
                    loss, loss_deriv = loss_and_deriv(self.loss, output, btarget)
                    loss = loss.mean() #np.sum(np.mean(loss,axis=1))

                    # Compute the gradients for the loss
                    directions = np.mean(bproba*loss_deriv,axis=(1,2))

                    # Compute the appropriate regularizer and its derivative
                    if self.regularizer is not None and self.l_reg > 0:
                        loss += self.l_reg * np.sum( [ (w * tr) for w, tr in zip(self.weights_, self.regularizer_evals)] )
                        reg_deriv = self.l_reg * self.regularizer_evals
                    else:
                        reg_deriv = 0

                    # Perform the gradient step + projection 
                    grad = directions + reg_deriv
                    if self.optimizer == "sgd":
                        # sgd
                        tmp_w = self.weights_ - self.lr*grad 
                    else:
                        # adam
                        beta1 = 0.9
                        beta2 = 0.999
                        m = beta1 * m + (1-beta1) * grad 
                        v = beta2 * v + (1-beta2) * (grad ** 2) 
                        m_corrected = m / (1-beta1**t)
                        v_corrected = v / (1-beta2**t)
                        tmp_w = self.weights_ - self.lr * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                        t += 1

                    self.weights_ = prox(tmp_w, self.ensemble_regularizer, self.normalize_weights, self.l_ensemble_reg, self.lr)
                    #print(self.weights_)
                    if self.ensemble_regularizer == "L0":
                        loss += self.l_ensemble_reg * np.linalg.norm(self.weights_,0)
                    elif self.ensemble_regularizer == "L1":
                        loss += self.l_ensemble_reg * np.linalg.norm(self.weights_,1)

                    loss_sum += loss
                    accuracy_sum += (output.argmax(axis=1) == btarget).mean() * 100.0
                    n_estimators_sum += self.num_estimators()
                    batch_time = time.time() - start_time

                    batch_cnt += 1 
                    pbar.update(bproba.shape[1])
                    
                    time_sum += batch_time
                    desc = '[{}/{}] loss {:2.4f} accuracy {:2.4f} n_estimators {:2.4f} time_item {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        loss_sum / batch_cnt,
                        accuracy_sum / batch_cnt, 
                        n_estimators_sum / batch_cnt,
                        time_sum / batch_cnt
                    )
                    pbar.set_description(desc)
    
        return [i for i in range(len(self.weights_)) if self.weights_[i] > 0], [w for w in self.weights_ if w > 0]
