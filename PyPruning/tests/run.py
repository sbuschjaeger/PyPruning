#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

from PyPruning.Metrics import error, neg_auc, individual_contribution, margin_diversity, kappa_statistic, combined, disagreement, q_zhang06

from PyPruning.RankPruningClassifier import RankPruningClassifier
#from PyPruning.MIQPPruningClassifier import MIQPPruningClassifier
from PyPruning.RandomPruningClassifier import RandomPruningClassifier
from PyPruning.ProxPruningClassifier import ProxPruningClassifier

from PyPruning.Papers import create_pruner

data, target = load_digits(return_X_y = True)

XTP, Xtest, ytp, ytest = train_test_split(data, target, test_size=0.25, random_state=42)
Xtrain, Xprune, ytrain, yprune = train_test_split(XTP, ytp, test_size=0.25, random_state=42)

n_base = 128
n_prune = 8
model = RandomForestClassifier(n_estimators=n_base)
model.fit(XTP, ytp)
pred = model.predict(Xtest)

print("Accuracy of RF trained on XTrain + XPrune with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

model = RandomForestClassifier(n_estimators=n_base)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)

print("Accuracy of RF trained on XTrain only with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

for p in [ "guo2018", "lu2010", "error", "neg_auc"]:
    pruned_model = create_pruner(p, n_estimators = n_prune)
    pruned_model.prune(Xprune, yprune, model.estimators_)
    pred = pruned_model.predict(Xtest)
    print("Accuracy of {} via Greedy with {} estimators: {} %".format(p, n_prune, 100.0 * accuracy_score(ytest, pred)))

    # pruned_model = create_pruner(p, n_estimators = n_prune)
    # pruned_model.prune(Xtrain, ytrain, model.estimators_)
    # pred = pruned_model.predict(Xtest)
    # print("Accuracy of {} via MIQP with {} estimators: {} %".format(p, n_prune, 100.0 * accuracy_score(ytest, pred)))
    # print("")


pruned_model = ProxPruningClassifier(l_ensemble_reg=1e-3, epochs=25, step_size=1e-2, ensemble_regularizer="L1", batch_size=32, verbose=False, loss="mse", normalize_weights=True, l_tree_reg=1e-3) 
pruned_model.prune(Xprune, yprune, model.estimators_)
pred = pruned_model.predict(Xtest)
print("Accuracy of ProxPruningClassifier L1 with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))

pruned_model = ProxPruningClassifier(l_ensemble_reg=1e-3, epochs=25, step_size=1e-2, ensemble_regularizer="L0", batch_size=32, verbose=False, loss="mse",normalize_weights=True, l_tree_reg=1e-3) 
pruned_model.prune(Xprune, yprune, model.estimators_)
pred = pruned_model.predict(Xtest)
print("Accuracy of ProxPruningClassifier L0 with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))

pruned_model = ProxPruningClassifier(l_ensemble_reg=8, epochs=25, step_size=1e-2, ensemble_regularizer="hard-L1", batch_size=32, verbose=False, loss="mse",normalize_weights=True, l_tree_reg=1e-3) 
pruned_model.prune(Xprune, yprune, model.estimators_)
pred = pruned_model.predict(Xtest)
print("Accuracy of ProxPruningClassifier hard-L1 with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))

pruned_model = RandomPruningClassifier(n_estimators = n_prune)
pruned_model.prune(Xprune, yprune, model.estimators_)
pred = pruned_model.predict(Xtest)
print("Accuracy of RandomPruningClassifier with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))
print("")

# for m in [error, neg_auc, individual_contribution, margin_diversity, kappa_statistic, combined, disagreement, q_zhang06]:
#     pruned_model = GreedyPruningClassifier(n_prune, single_metric = m)
#     pruned_model.prune(Xtrain, ytrain, model.estimators_)
#     pred = pruned_model.predict(Xtest)
#     print("GreedyPruningClassifier with {} estimators and {} metric is {} %".format(n_prune, m.__name__, 100.0 * accuracy_score(ytest, pred)))

#     pruned_model = ILPPruningClassifier(n_prune, single_metric = m)
#     pruned_model.prune(Xtrain, ytrain, model.estimators_)
#     pred = pruned_model.predict(Xtest)
#     print("ILPPruningClassifier with {} estimators and {} metric is {} %".format(n_prune, m.__name__, 100.0 * accuracy_score(ytest, pred)))
#     print("")