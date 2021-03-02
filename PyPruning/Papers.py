from functools import partial 

from .MIQPPruningClassifier import MIQPPruningClassifier
from .GreedyPruningClassifier import GreedyPruningClassifier
from .Metrics import Metrics

def create_pruner(optimizer = "Greedy", paper = "margineantu1997", **kwargs):
    optimizer = optimizer.lower()
    
    assert optimizer in ["greedy", "miqp"], "You provided {} as optimizer, but we only support {{Greedy, MIQP}}".format(optimizer)

    assert paper in ["margineantu1997", "lazarevic2001", "lu2010", "guo2018", "cavalcanti2016", "zhang2006"], "You provided {} as optimizer, but we only support {{margineantu1997, lazarevic2001, lu2010, guo2018, cavalcanti2016, zhang2006}}".format(paper)

    if paper == "lu2010":
        single_metric = Metrics.individual_contribution
        pairwise_metric = None
        l_reg = 0
    elif paper == "guo2018":
        single_metric = partial(Metrics.margin_diversity, alpha = 0.2)
        pairwise_metric = None
        l_reg = 0
    elif paper == "margineantu1997":
        if optimizer == "greedy":
            single_metric = Metrics.error
        else:
            single_metric = None
        pairwise_metric = Metrics.kappa_statistic
        l_reg = 1
    elif paper == "lazarevic2001":
        if optimizer == "greedy":
            single_metric = Metrics.error
        else:
            single_metric = None
        pairwise_metric = Metrics.disagreement
        l_reg = 1
    elif paper == "zhang2006":
        single_metric = Metrics.error
        pairwise_metric = Metrics.q_zhang06
        l_reg = 0.5
    elif paper == "cavalcanti2016":
        if optimizer == "greedy":
            single_metric = Metrics.error
        else:
            single_metric = None
        pairwise_metric = Metrics.combined
        l_reg = 1
    else:
        # should not happen
        pass

    if optimizer == "greedy":
        return GreedyPruningClassifier(single_metric=single_metric, pairwise_metric=pairwise_metric, l_reg = l_reg, **kwargs)
    elif optimizer == "miqp":
        return MIQPPruningClassifier(single_metric=single_metric, pairwise_metric=pairwise_metric, l_reg = l_reg, **kwargs)
    else:
        # should not happen
        return None