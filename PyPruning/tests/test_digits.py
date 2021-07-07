#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

from PyPruning.RandomPruningClassifier import RandomPruningClassifier
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier, error, complementariness, drep, neg_auc, margin_distance
from PyPruning.ProxPruningClassifier import ProxPruningClassifier
from PyPruning.RankPruningClassifier import RankPruningClassifier, reference_vector, individual_kappa_statistic, individual_neg_auc, error_ambiguity, individual_error, individual_contribution, individual_margin_diversity

from PyPruning.Papers import create_pruner
from PyPruning.ClusterPruningClassifier import ClusterPruningClassifier, accuracy, centroid_selector, kmeans, random_selector, agglomerative, largest_mean_distance

data, target = load_digits(return_X_y = True)

XTP, Xtest, ytp, ytest = train_test_split(data, target, test_size=0.25, random_state=42)
Xtrain, Xprune, ytrain, yprune = train_test_split(XTP, ytp, test_size=0.25, random_state=42)

n_base = 128
n_prune = 8
metrics = []

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--commit", help="Git commit hash", action="store", type=str)
parser.add_argument("-f", "--file", help="File to append results to",action="store", default="./accuracies.csv", type=str)
args = parser.parse_args()

model = RandomForestClassifier(n_estimators=n_prune)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)
metrics.append({
    "method":"RF-{} ".format(n_prune),
    "accuracy":100.0 * accuracy_score(ytest, pred),
    "commit":args.commit
})

model = RandomForestClassifier(n_estimators=n_base)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)
metrics.append({
    "method":"RF-{} ".format(n_base),
    "accuracy":100.0 * accuracy_score(ytest, pred),
    "commit":args.commit
})

for ce in [kmeans, agglomerative]:
    for se in [random_selector, centroid_selector, accuracy, largest_mean_distance]:
        pruned_model = ClusterPruningClassifier(cluster_estimators=ce, select_estimators=se, n_estimators=n_prune)
        pruned_model.prune(Xprune, yprune, model.estimators_)
        pred = pruned_model.predict(Xtest)
        metrics.append({
            "method":"ClusterPruningClassifier-{}, cd = {}, se = {} ".format(n_prune, ce.__name__, se.__name__),
            "accuracy":100.0 * accuracy_score(ytest, pred),
            "commit":args.commit
        })

# for m in [error, neg_auc, complementariness, margin_distance, drep]:
#     pruned_model = GreedyPruningClassifier(metric = m, n_estimators = n_prune)
#     pruned_model.prune(Xprune, yprune, model.estimators_)
#     pred = pruned_model.predict(Xtest)
#     metrics.append({
#         "method":"GreedyPruningClassifier-{}, m = {}".format(n_prune, m.__name__),
#         "accuracy":100.0 * accuracy_score(ytest, pred),
#         "commit":args.commit
#     })

# for m in [reference_vector, individual_kappa_statistic, individual_neg_auc, error_ambiguity, individual_error, individual_contribution, individual_margin_diversity]:
#     pruned_model = RankPruningClassifier(metric = m, n_estimators = n_prune)
#     pruned_model.prune(Xprune, yprune, model.estimators_)
#     pred = pruned_model.predict(Xtest)
#     metrics.append({
#         "method":"RankPruningClassifier-{}, m = {}".format(n_prune, m.__name__),
#         "accuracy":100.0 * accuracy_score(ytest, pred),
#         "commit":args.commit
#     })

# # TODO Vary other parameter?
# for l in ["mse", "cross-entropy"]:
#     for l_tree in [0, 1e-5]:
#         pruned_model = ProxPruningClassifier(l_ensemble_reg=n_prune, epochs=25, step_size=1e-3, ensemble_regularizer="hard-L0", batch_size=32, verbose=False, loss=l, normalize_weights=True, l_tree_reg=l_tree) 
#         pruned_model.prune(Xprune, yprune, model.estimators_)
#         pred = pruned_model.predict(Xtest)
#         metrics.append({
#             "method":"ProxPruningClassifier-{}, l = {}, l_tree = {}".format(n_prune, l, l_tree),
#             "accuracy":100.0 * accuracy_score(ytest, pred),
#             "commit":args.commit
#         })

# pruned_model = RandomPruningClassifier(n_estimators = n_prune)
# pruned_model.prune(Xprune, yprune, model.estimators_)
# pred = pruned_model.predict(Xtest)
# metrics.append({
#     "method":"PRandomPruningClassifier-{}".format(n_prune),
#     "accuracy":100.0 * accuracy_score(ytest, pred),
#     "commit":args.commit
# })
# TOOD ADD DATE
# df = pd.DataFrame(metrics)
# df.to_csv(args.file, mode='a', header= not os.path.exists(args.file), index=False)
