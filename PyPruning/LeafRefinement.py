from PyPruning.ProxPruningClassifier import ProxPruningClassifier

class LeafRefinement(ProxPruningClassifier):
    def __init__(self,
        loss = "cross-entropy",
        step_size = 1e-1,
        batch_size = 256,
        epochs = 1,
        verbose = False, 
        out_path = None,
        eval_every_epochs = None,
    ):
        super().__init__(
            ensemble_regularizer = "L0",
            l_ensemble_reg = 0,
            l_tree_reg = 0,
            batch_size = batch_size,
            epochs =  epochs,
            step_size =  step_size, 
            verbose = verbose,
            loss = loss,
            update_leaves = True,
            normalize_weights = True,
            update_weights = False,
            out_path=out_path,
            eval_every_epochs=eval_every_epochs
        )