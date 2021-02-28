#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

from Metrics import accuracy, auc, individual_contribution, margin_diversity, kappa_statistic, combined, disagreement, q_zhang06

from GreedyPruningClassifier import GreedyPruningClassifier

data, target = load_digits(return_X_y = True)

Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.33, random_state=42)

n_base = 128
n_prune = 16
model = RandomForestClassifier(n_estimators=n_base)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)

print("Accuracy of RF with {} estimators {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

for m in [accuracy, auc, individual_contribution, margin_diversity, kappa_statistic, combined, disagreement, q_zhang06]:

    pruned_model = GreedyPruningClassifier(n_prune, single_metric = m)
    pruned_model.prune(Xtrain, ytrain, model.estimators_)
    pred = pruned_model.predict(Xtest)

    print("Accuracy of pruned model with {} estimators and {} metric is {} %".format(n_prune, m.__name__, 100.0 * accuracy_score(ytest, pred)))
    print("")
