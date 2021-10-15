#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import random

from functools import partial

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from PyPruning.ProxPruningClassifier import ProxPruningClassifier
from PyPruning.Papers import create_pruner

df = pd.read_csv("wine.csv", header = 0, delimiter=";")
df = df.dropna()
label = df.pop("quality")
le = LabelEncoder()
Y = le.fit_transform(label)
X = df.values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
kf = KFold(n_splits=10, random_state=12345, shuffle=True)
idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X)], dtype=object)

accs = []
for (itrain, itest) in kf.split(X):
    dropidx = np.array( [i for i in itrain if Y[i] == 4] )
    itrain = np.setdiff1d(itrain, dropidx)
    XTest, YTest = X[itest], Y[itest]
    XTrain, XPrune, YTrain, YPrune = train_test_split(X[itrain], Y[itrain], test_size = 0.33) 

    print("Training initial RF on {} datapoints with classes {}".format(XTrain.shape, set(YTrain)))
    model = RandomForestClassifier(n_estimators=32, max_depth=None)
    model.fit(XTrain,YTrain)

    pruner = create_pruner("individual_contribution", n_estimators=8)
    print("Pruning on {} datapoints with classes {}".format(XPrune.shape, set(YPrune)))
    # pruner = ProxPruningClassifier(n_estimators=64, batch_size=32, epochs=50, loss="mse", l_reg = 0, step_size=1e-2, verbose=True, normalize_weights = True)
    pruner.prune(XPrune, YPrune,model.estimators_, model.classes_, len(set(Y)))

    accs.append(100.0 * accuracy_score(YTest, pruner.predict(XTest)))
    print("Testing on {} datapoints: {}".format(XTest.shape, accs[-1]))

print("Accuracy was: {}".format(np.mean(accs)))