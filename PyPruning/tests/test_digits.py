#!/usr/bin/env python3

from PyPruning.MIQPPruningClassifier import MIQPPruningClassifier
import sys
import os
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

from PyPruning.MIQPPruningClassifier import combined, combined_error
from PyPruning.RandomPruningClassifier import RandomPruningClassifier
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier, reduced_error, complementariness, drep, neg_auc, margin_distance
from PyPruning.ProxPruningClassifier import ProxPruningClassifier
from PyPruning.RankPruningClassifier import RankPruningClassifier, reference_vector, individual_kappa_statistic, individual_neg_auc, error_ambiguity, individual_error, individual_contribution, individual_margin_diversity
from PyPruning.ClusterPruningClassifier import ClusterPruningClassifier, cluster_accuracy, centroid_selector, kmeans, random_selector, agglomerative, largest_mean_distance

from PyPruning.Papers import create_pruner

def test_model(model, Xprune, yprune, Xtest, ytest, estimators, target_accuracy = 0.9):
    model.prune(Xprune, yprune, estimators)
    pred = model.predict(Xtest)
    return accuracy_score(ytest, pred) > target_accuracy

data, target = load_digits(return_X_y = True)

XTP, Xtest, ytp, ytest = train_test_split(data, target, test_size=0.25, random_state=42)
Xtrain, Xprune, ytrain, yprune = train_test_split(XTP, ytp, test_size=0.25, random_state=42)

n_base = 128
n_prune = 8

target_accuracy = 0.9

model = RandomForestClassifier(n_estimators=n_prune)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)
if not accuracy_score(ytest, pred) > target_accuracy:
    sys.exit(1)
else:
    method = "RF-{} ".format(n_base)
    print("Test passed for ", method)

model = RandomForestClassifier(n_estimators=n_base)
model.fit(Xtrain, ytrain)
if not accuracy_score(ytest, pred) > target_accuracy:
    sys.exit(1)
else:
    method = "RF-{} ".format(n_base)
    print("Test passed for ", method)

for sm in [individual_error]:
    for pm in [combined_error, combined]:
        pruned_model = MIQPPruningClassifier(single_metric = sm, pairwise_metric = pm, alpha = 0.5, n_estimators = n_prune, verbose=False)
        if not test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_,target_accuracy):
            sys.exit(1)
        else:
            method = "MIQPPruningClassifier-{}, single = {}, pair = {}, alpha = {}".format(n_prune, sm.__name__, pm.__name__, 0.5)
            print("Test passed for ", method)

for ce in [kmeans, agglomerative]:
    for se in [random_selector, centroid_selector, cluster_accuracy, largest_mean_distance]:
        pruned_model = ClusterPruningClassifier(cluster_estimators=ce, select_estimators=se, n_estimators=n_prune)
        if not test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_,target_accuracy):
            sys.exit(1)
        else:
            method = "ClusterPruningClassifier-{}, cd = {}, se = {} ".format(n_prune, ce.__name__, se.__name__)
            print("Test passed for ", method)

for m in [reduced_error, neg_auc, complementariness, margin_distance, drep]:
    pruned_model = GreedyPruningClassifier(metric = m, n_estimators = n_prune)
    if not test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_,target_accuracy):
        sys.exit(1)
    else:
        method = "GreedyPruningClassifier-{}, m = {}".format(n_prune, m.__name__)
        print("Test passed for ", method)

for m in [reference_vector, individual_kappa_statistic, individual_neg_auc, error_ambiguity, individual_error, individual_contribution, individual_margin_diversity]:
    pruned_model = RankPruningClassifier(metric = m, n_estimators = n_prune)
    if not test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_,target_accuracy):
        sys.exit(1)
    else:
        method = "RankPruningClassifier-{}, m = {}".format(n_prune, m.__name__)
        print("Test passed for ", method)

# # TODO Vary other parameter?
for l in ["mse", "cross-entropy"]:
    for l_tree in [0, 1e-5]:
        pruned_model = ProxPruningClassifier(l_ensemble_reg=n_prune, epochs=25, step_size=1e-3, ensemble_regularizer="hard-L0", batch_size=32, verbose=False, loss=l, normalize_weights=True, l_tree_reg=l_tree) 
        if not test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_,target_accuracy):
            sys.exit(1)
        else:
            method = "ProxPruningClassifier-{}, l = {}, l_tree = {}".format(n_prune, l, l_tree)
            print("Test passed for ", method)

pruned_model = RandomPruningClassifier(n_estimators = n_prune)
if not test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_,target_accuracy):
    sys.exit(1)
else:
    method = "RandomPruningClassifier-{}".format(n_prune)
    print("Test passed for ", method)

for s in ["individual_margin_diversity", "individual_contribution", "individual_error", "individual_kappa_statistic", "reduced_error", "complementariness", "drep", "margin_distance", "combined", "reference_vector", "combined_error", "error_ambiguity", "largest_mean_distance", "cluster_accuracy", "cluster_centroids"]:
    pruned_model = create_pruner(s, n_estimators = n_prune) 
    if not test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_,target_accuracy):
        sys.exit(1)
    else:
        print("Test passed for ", s)

sys.exit(0)
