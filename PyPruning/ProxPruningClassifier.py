import numpy as np
from joblib import Parallel, delayed
import numbers
import time
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score

from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier

from .PruningClassifier import PruningClassifier

# Modified from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
def create_mini_batches(inputs, targets, batch_size, shuffle=False):
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
    if x is None or len(x) == 0:
        return x
    sorted_x = np.sort(x)
    x_sum = sorted_x[0]
    l = 1.0 - sorted_x[0]
    for i in range(1,len(sorted_x)):
        x_sum += sorted_x[i]
        tmp = 1.0 / (i + 1.0) * (1.0 - x_sum)
        if (sorted_x[i] + tmp) > 0:
            l = tmp 
    
    return [max(xi + l, 0.0) for xi in x]

class ProxPruningClassifier(PruningClassifier):
    def __init__(self,
        n_estimators,
        base_estimator = None,
        n_jobs = 1,
        loss = "cross-entropy",
        step_size = 1e-1,
        regularizer = "node",
        l_reg = 0,
        normalize_weights = True,
        batch_size = 256,
        epochs = 1,
        verbose = True, 
        out_path = None,
        eval_every_epochs = None):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert l_reg >= 0, "l_reg must be greate or equal to 0"
        assert regularizer is None or regularizer in ["node"], "Currently only {{none, node}} regularizer is supported for tree the regularizer."
        assert batch_size >= 1, "batch_size must be at-least 1"
        assert epochs >= 1, "epochs must be at-least 1"

        super().__init__(n_estimators, base_estimator, n_jobs)
        
        self.loss = loss
        self.step_size = step_size
        self.regularizer = regularizer
        self.l_reg = l_reg
        self.normalize_weights = normalize_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.out_path = out_path
        self.eval_every_epochs = eval_every_epochs

    def _combine_proba(self, all_proba):
        scaled_prob = np.array([w * p for w,p in zip(all_proba, self.weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def next(self, proba, target):
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
        
        loss = np.mean(loss) 
        
        # Compute the gradient for the loss
        directions = np.mean(proba*loss_deriv,axis=(1,2))

        # Compute the appropriate regularizer
        if self.regularizer == "node" and self.l_reg > 0:
            loss += self.l_reg * np.sum( [ (w * est.tree_.node_count) for w, est in zip(self.weights_, self.estimators_)] )
            
            node_deriv = self.l_reg * np.array([ est.tree_.node_count for est in self.estimators_])
        else:
            node_deriv = 0

        # Perform the gradient step + top-K projection 
        tmp_w = self.weights_ - self.step_size*directions - self.step_size*node_deriv
        # print(tmp_w[0:5])
        #print(self.weights_[0:5])
        
        top_K = np.argsort(tmp_w)[-self.n_estimators:]
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
        return np.count_nonzero(self.weights_)

    def num_parameters(self):
        return sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(self.weights_, self.estimators_)] )

    def prune_(self, proba, target):
        proba = np.swapaxes(proba, 0, 1)
        self.weights_ = np.array([1.0 / proba.shape[1] for _ in range(proba.shape[1])])

        for epoch in range(self.epochs):

            mini_batches = create_mini_batches(proba, target, self.batch_size, True) 

            times = []
            total_time = 0
            
            metrics = {}

            example_cnt = 0

            with tqdm(total=proba.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for batch in mini_batches:
                    bproba, btarget = batch 

                    # Update Model                    
                    start_time = time.time()
                    batch_metrics = self.next(bproba, btarget)
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
