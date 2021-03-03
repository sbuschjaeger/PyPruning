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

#df = pd.read_csv("PyPruning/tests/magic04.data")
df = pd.read_csv("magic04.data")
X = df.values[:,:-1].astype(np.float64)
Y = df.values[:,-1]
Y = np.array([0 if y == 'g' else 1 for y in Y])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
kf = KFold(n_splits=5, random_state=12345, shuffle=True)
idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X)], dtype=object)

accs = []
for (itrain, itest) in kf.split(X):
    XTest, YTest = X[itest], Y[itest]
    XTrain, XPrune, YTrain, YPrune = train_test_split(X[itrain], Y[itrain], test_size = 0.25) 

    print("Training initial RF on {} datapoints".format(XTrain.shape))
    model = RandomForestClassifier(n_estimators=128, max_depth=None)
    model.fit(XTrain,YTrain)

    print("Pruning on {} datapoints".format(XPrune.shape))
    pruner = ProxPruningClassifier(n_estimators=64, batch_size=32, epochs=50, loss="mse", l_reg = 0, step_size=1e-2, verbose=True, normalize_weights = True)
    pruner.prune(XPrune, YPrune,model.estimators_)

    accs.append(100.0 * accuracy_score(YTest, pruner.predict(XTest)))
    print("Testing on {} datapoints: {}".format(XTest.shape, accs[-1]))

print("Accuracy was: {}".format(np.mean(accs)))