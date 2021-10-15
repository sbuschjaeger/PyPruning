#!/usr/bin/env python3

from PyPruning.MIQPPruningClassifier import MIQPPruningClassifier
import sys
import os
import numpy as np
import pandas as pd
import argparse

#SKLearn sometimes throws warnings due to n_jobs not being supported in the future for KMeans. Just ignore them for now
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

def test_model(model, Xprune, yprune, Xtest, ytest, estimators):
    model.prune(Xprune, yprune, estimators)
    pred = model.predict(Xtest)
    return accuracy_score(ytest, pred)

data, target = load_digits(return_X_y = True)

XTP, Xtest, ytp, ytest = train_test_split(data, target, test_size=0.25, random_state=42)
Xtrain, Xprune, ytrain, yprune = train_test_split(XTP, ytp, test_size=0.25, random_state=42)

n_base = 128
n_prune = 8

target_accuracy = 0.80

model = RandomForestClassifier(n_estimators=n_prune)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)
acc = accuracy_score(ytest, pred)
method = "RF-{} ".format(n_prune)
if not acc > target_accuracy:
    print("Test failed for {} with accuracy {}".format(method, acc*100.0))
    sys.exit(1)
else:
    print("Test passed for {} with accuracy {}".format(method, acc*100.0))

model = RandomForestClassifier(n_estimators=n_base)
model.fit(Xtrain, ytrain)
acc = accuracy_score(ytest, pred)
method = "RF-{} ".format(n_base)
if not acc > target_accuracy:
    print("Test failed for {} with accuracy {}".format(method, acc*100.0))
    sys.exit(1)
else:
    print("Test passed for {} with accuracy {}".format(method, acc*100.0))

for sm in [individual_error]:
    for pm in [combined_error, combined]:
        pruned_model = MIQPPruningClassifier(single_metric = sm, pairwise_metric = pm, alpha = 0.5, n_estimators = n_prune, verbose=False)
        method = "MIQPPruningClassifier-{}, single = {}, pair = {}, alpha = {}".format(n_prune, sm.__name__, pm.__name__, 0.5)
        acc = test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_)
        if not acc > target_accuracy:
            print("Test failed for {} with accuracy {}".format(method, acc*100.0))
            sys.exit(1)
        else:
            print("Test passed for {} with accuracy {}".format(method, acc*100.0))

for ce in [kmeans, agglomerative]:
    for se in [random_selector, centroid_selector, cluster_accuracy, largest_mean_distance]:
        pruned_model = ClusterPruningClassifier(cluster_estimators=ce, select_estimators=se, n_estimators=n_prune)
        method = "ClusterPruningClassifier-{}, cd = {}, se = {} ".format(n_prune, ce.__name__, se.__name__)
        acc = test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_)
        if not acc > target_accuracy:
            print("Test failed for {} with accuracy {}".format(method, acc*100.0))
            sys.exit(1)
        else:
            print("Test passed for {} with accuracy {}".format(method, acc*100.0))

for m in [reduced_error, neg_auc, complementariness, margin_distance, drep]:
    pruned_model = GreedyPruningClassifier(metric = m, n_estimators = n_prune)
    method = "GreedyPruningClassifier-{}, m = {}".format(n_prune, m.__name__)
    acc = test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_)
    if not acc > target_accuracy:
        print("Test failed for {} with accuracy {}".format(method, acc*100.0))
        sys.exit(1)
    else:
        print("Test passed for {} with accuracy {}".format(method, acc*100.0))

for m in [reference_vector, individual_kappa_statistic, individual_neg_auc, error_ambiguity, individual_error, individual_contribution, individual_margin_diversity]:
    pruned_model = RankPruningClassifier(metric = m, n_estimators = n_prune)
    method = "RankPruningClassifier-{}, m = {}".format(n_prune, m.__name__)
    acc = test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_)
    if not acc > target_accuracy:
        print("Test failed for {} with accuracy {}".format(method, acc*100.0))
        sys.exit(1)
    else:
        print("Test passed for {} with accuracy {}".format(method, acc*100.0))

# # TODO Vary other parameter?
for l in ["mse", "cross-entropy"]:
    for l_tree in [0, 1e-5]:
        pruned_model = ProxPruningClassifier(l_ensemble_reg=n_prune, epochs=25, step_size=1e-3, ensemble_regularizer="hard-L0", batch_size=32, verbose=False, loss=l, normalize_weights=True, l_tree_reg=l_tree) 
        method = "ProxPruningClassifier-{}, l = {}, l_tree = {}".format(n_prune, l, l_tree)
        acc = test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_)
        if not acc > target_accuracy:
            print("Test failed for {} with accuracy {}".format(method, acc*100.0))
            sys.exit(1)
        else:
            print("Test passed for {} with accuracy {}".format(method, acc*100.0))

pruned_model = RandomPruningClassifier(n_estimators = n_prune)
method = "RandomPruningClassifier-{}".format(n_prune)
acc = test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_)
if not acc > target_accuracy:
    print("Test failed for {} with accuracy {}".format(method, acc*100.0))
    sys.exit(1)
else:
    print("Test passed for {} with accuracy {}".format(method, acc*100.0))

for s in ["individual_margin_diversity", "individual_contribution", "individual_error", "individual_kappa_statistic", "reduced_error", "complementariness", "drep", "margin_distance", "combined", "reference_vector", "combined_error", "error_ambiguity", "largest_mean_distance", "cluster_accuracy", "cluster_centroids"]:
    pruned_model = create_pruner(s, n_estimators = n_prune) 
    acc = test_model(pruned_model, Xprune, yprune, Xtest, ytest, model.estimators_)
    if not acc > target_accuracy:
        print("Test failed for {} with accuracy {}".format(s, acc*100.0))
        sys.exit(1)
    else:
        print("Test passed for {} with accuracy {}".format(s, acc*100.0))
sys.exit(0)
