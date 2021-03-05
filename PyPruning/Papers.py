from functools import partial 

from .MIQPPruningClassifier import MIQPPruningClassifier
from .OrderPruningClassifier import OrderPruningClassifier
from .RankPruningClassifier import RankPruningClassifier

#from .Metrics import Metrics
from .Metrics import error, margin_diversity, individual_contribution, kappa_statistic, disagreement, q_zhang06, combined, neg_auc

def create_pruner(paper = "margineantu1997", **kwargs):
    assert paper in ["error", "neg_auc", "lu2010", "guo2018"], "You provided {} as optimizer, but we only support {{lu2010, guo2018, error, neg_auc}}".format(paper)

    if paper == "guo2018":
        metric = partial(margin_diversity, alpha = 0.2)
        return RankPruningClassifier(metric=metric,  **kwargs)
    elif paper == "lu2010":
        return RankPruningClassifier(metric=individual_contribution,  **kwargs)
    elif paper == "error":
        return RankPruningClassifier(metric=error,  **kwargs)
    elif paper == "neg_auc":
        return RankPruningClassifier(metric=neg_auc,  **kwargs)
    return None
    # elif paper == "margineantu1997":
    #     if optimizer == "greedy":
    #         single_metric = error
    #     else:
    #         single_metric = None
    #     pairwise_metric = kappa_statistic
    #     l_reg = 1
    # elif paper == "lazarevic2001":
    #     if optimizer == "greedy":
    #         single_metric = error
    #     else:
    #         single_metric = None
    #     pairwise_metric = disagreement
    #     l_reg = 1
    # elif paper == "zhang2006":
    #     single_metric = error
    #     pairwise_metric = q_zhang06
    #     l_reg = 0.5
    # elif paper == "cavalcanti2016":
    #     if optimizer == "greedy":
    #         single_metric = error
    #     else:
    #         single_metric = None
    #     pairwise_metric = combined
    #     l_reg = 1
    # else:
    #     # should not happen
    #     pass

    # if optimizer == "greedy":
    #     return GreedyPruningClassifier(single_metric=single_metric, pairwise_metric=pairwise_metric, l_reg = l_reg, **kwargs)
    # elif optimizer == "miqp":
    #     return MIQPPruningClassifier(single_metric=single_metric, pairwise_metric=pairwise_metric, l_reg = l_reg, **kwargs)
    # else:
    #     # should not happen
    #     return None