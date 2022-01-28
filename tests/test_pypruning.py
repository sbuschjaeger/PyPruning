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

from PyPruning.RandomPruningClassifier import RandomPruningClassifier
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier, reduced_error, complementariness, drep, neg_auc, margin_distance
from PyPruning.ProxPruningClassifier import ProxPruningClassifier
from PyPruning.RankPruningClassifier import RankPruningClassifier, reference_vector, individual_kappa_statistic, individual_neg_auc, error_ambiguity, individual_error, individual_contribution, individual_margin_diversity
from PyPruning.ClusterPruningClassifier import ClusterPruningClassifier, cluster_accuracy, centroid_selector, kmeans, random_selector, agglomerative, largest_mean_distance

from PyPruning.Papers import create_pruner

import tempfile
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.io.arff import loadarff
import urllib

from io import BytesIO, TextIOWrapper
from zipfile import ZipFile

def download(url, filename, tmpdir = None):
    """Download the file under the given url and store it in the given tmpdir udner the given filename. If tmpdir is None, then `tempfile.gettmpdir()` will be used which is most likely /tmp on Linux systems.

    Args:
        url (str): The URL to the file which should be downloaded.
        filename (str): The name under which the downlaoded while should be stored.
        tmpdir (Str, optional): The directory in which the file should be stored. Defaults to None.

    Returns:
        str: Returns the full path under which the file is stored. 
    """
    if tmpdir is None:
        tmpdir = os.path.join(tempfile.gettempdir(), "data")

    os.makedirs(tmpdir, exist_ok=True)

    if not os.path.exists(os.path.join(tmpdir,filename)):
        print("{} not found. Downloading.".format(os.path.join(tmpdir,filename)))
        urllib.request.urlretrieve(url, os.path.join(tmpdir,filename))
    return os.path.join(tmpdir,filename)

def read_arff(path, class_name):
    """Loads the ARFF file under the given path and transforms it into a pandas dataframe. Each column which does not match class_name is copied into the pandas frame without changes. The column with the name `class_name` is renamed to `label` in the DataFrame. The behaviour of this method is undefined if the ARFF file already contains a `label` column and `class_name != 'label'`. 

    Args:
        path (str): The path to the ARFF file.
        class_name (str): The label column in the ARFF file

    Returns:
        pandas.DataFrame : A pandas dataframe containing the data from the ARFF file and an additional `label` column.
    """
    data, meta = loadarff(path)
    Xdict = {}
    for cname, ctype in zip(meta.names(), meta.types()):
        # Get the label attribute for the specific dataset:
        #   eeg: eyeDetection
        #   elec: class
        #   nomao: Class
        #   polish-bankruptcy: class
        if cname == class_name:
        #if cname in ["eyeDetection", "class",  "Class"]:
            enc = LabelEncoder()
            Xdict["label"] = enc.fit_transform(data[cname])
        else:
            Xdict[cname] = data[cname]
    return pd.DataFrame(Xdict)

def get_dataset(dataset, tmpdir = None, split = 0.3):
    """Returns XTrain, YTrain, XTest, YTest of the given dataset by name. If the dataset does not exist it will be automatically downloaded.

    Args:
        dataset (str): The name of the dataset to be returned (and downloaded if required.). Currently supports {magic, mnist, fashion, eeg}
        tmpdir (str, optional): The temporary folder to which the dataset is downloaded if it does not exist. If None then uses tempfile.gettempdir() to query for an appropriate temp folder. Defaults to None.
        split (float, optional): The applied train/test split. If the data-set comes with a pre-defined split (e.g. mnist) this value is ignored. Defaults to 0.3

    Raises:
        ValueError: Raises a ValueError if an unsupported dataset is passed as an argument

    Returns:
        XTrain, YTrain, XTest, YTest (2d np.array, np.array, 2d np.array, np.array): Returns the (N, d) train/test data and the (N, ) train/test labels where N is the number of data points and d is the number of features. 
    """

    if dataset == "magic":
        magic_path = download("http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", "magic.csv", tmpdir)
        df = pd.read_csv(magic_path)
        X = df.values[:,:-1].astype(np.float64)
        Y = df.values[:,-1]
        Y = np.array([0 if y == 'g' else 1 for y in Y])
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=split, random_state=42)
    elif dataset == "fashion" or dataset == "mnist":
        def load_mnist(path, kind='train'):
            # Taken from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
            import os
            import gzip
            import numpy as np

            """Load MNIST data from `path`"""
            labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
            images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

            return images, labels

        if dataset == "fashion":
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "fashion")
            else:
                out_path = os.path.join(tmpdir, "data", "fashion")

            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)
        else:
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "mnist")
            else:
                out_path = os.path.join(tmpdir, "data", "mnist")

            train_path = download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)

        XTrain, YTrain = load_mnist(out_path, kind='train')
        XTest, YTest = load_mnist(out_path, kind='t10k')
    elif dataset == "eeg":
        eeg_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff", "eeg.arff", tmpdir)
        
        df = read_arff(eeg_path, "eyeDetection")
        df = pd.get_dummies(df)
        df.dropna(axis=1, inplace=True)
        Y = df["label"].values.astype(np.int32)
        df = df.drop("label", axis=1)

        X = df.values.astype(np.float64)
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=split, random_state=42)
    else:
        raise ValueError("Unsupported dataset provided to get_dataset in test_utils.py: {}. Currently supported are {mnist, fashion eeg, magic}".format(dataset))
        # return None, None

    return XTrain, YTrain, XTest, YTest

def test_model(model, Xprune, yprune, Xtest, ytest, estimators, name):
    print("Testing {}".format(name), end="", flush=True)
    model.prune(Xprune, yprune, estimators)
    pred = model.predict(Xtest)
    acc = accuracy_score(ytest, pred)
    print(" with accuracy {}".format(acc))

def main():
    parser = argparse.ArgumentParser(description='Test and benchmark various pruning algorithms on the supplied dataset.')
    parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {magic, mnist, fashion, eeg}.')
    parser.add_argument('--test_split', required=False, default=0.2, type=float, help='Test/Train split.')
    parser.add_argument('--prune_split', required=False, default=0.2, type=float, help='Train/Prune split.')
    parser.add_argument('--n_estimators', required=False, type=int, default=128,help='Number of trees in a random forest.')
    parser.add_argument('--n_prune', required=False, type=int, default=32,help='Number of trees to prune forest down to.')
    parser.add_argument('--maxdepth', required=False, type=int, default=20,help='Maximum tree-depth for decision trees and random forest.')
    args = parser.parse_args()

    print("Loading {}".format(args.dataset))
    XTrain, YTrain, XTest, YTest = get_dataset(args.dataset,split = args.test_split)
    XTrain, XPrune, YTrain, YPrune = train_test_split(XTrain, YTrain, test_size=args.prune_split, random_state=42)

    model = RandomForestClassifier(n_estimators=args.n_prune)
    method = "RF-{} ".format(args.n_prune)
    print("Testing {}".format(method), end="", flush=True)
    model.fit(XTrain, YTrain)
    pred = model.predict(XTest)
    acc = accuracy_score(YTest, pred)
    print("with accuracy {}".format(acc))

    model = RandomForestClassifier(n_estimators=args.n_estimators)
    method = "RF-{} ".format(args.n_estimators)
    print("Testing {}".format(method), end="", flush=True)
    model.fit(XTrain, YTrain)
    acc = accuracy_score(YTest, pred)
    print("with accuracy {}".format(acc))

    for ce in [kmeans, agglomerative]:
        for se in [random_selector, centroid_selector, cluster_accuracy, largest_mean_distance]:
            pruned_model = ClusterPruningClassifier(cluster_estimators=ce, select_estimators=se, n_estimators=args.n_prune)
            name = "ClusterPruningClassifier-{}, cd = {}, se = {} ".format(args.n_prune, ce.__name__, se.__name__)
            test_model(pruned_model, XPrune, YPrune, XTest, YTest, model.estimators_, name)

    for m in [reduced_error, neg_auc, complementariness, margin_distance, drep]:
        pruned_model = GreedyPruningClassifier(metric = m, n_estimators = args.n_prune)
        name = "GreedyPruningClassifier-{}, m = {}".format(args.n_prune, m.__name__)
        test_model(pruned_model, XPrune, YPrune, XTest, YTest, model.estimators_, name)

    for m in [reference_vector, individual_kappa_statistic, individual_neg_auc, error_ambiguity, individual_error, individual_contribution, individual_margin_diversity]:
        pruned_model = RankPruningClassifier(metric = m, n_estimators = args.n_prune)
        name = "RankPruningClassifier-{}, m = {}".format(args.n_prune, m.__name__)
        test_model(pruned_model, XPrune, YPrune, XTest, YTest, model.estimators_, name)

    for l in ["mse", "cross-entropy"]:
        for o in ["adam", "sgd"]:
            for l_tree in [0]:
                pruned_model = ProxPruningClassifier(l_ensemble_reg=args.n_prune, epochs=25, step_size=2.5e-1, ensemble_regularizer="hard-L0", batch_size=0, verbose=False, loss=l, normalize_weights=True, l_reg=l_tree, optimizer = o) 

                name = "ProxPruningClassifier-{}, l = {}, o = {}, l_tree = {}".format(args.n_prune, l, o, l_tree)
                test_model(pruned_model, XPrune, YPrune, XTest, YTest, model.estimators_, name)

    name = "RandomPruningClassifier-{}".format(args.n_prune)
    acc = test_model(pruned_model, XPrune, YPrune, XTest, YTest, model.estimators_, name)

    for s in ["individual_margin_diversity", "individual_contribution", "individual_error", "individual_kappa_statistic", "reduced_error", "complementariness", "drep", "margin_distance", "reference_vector", "error_ambiguity", "largest_mean_distance", "cluster_accuracy", "cluster_centroids"]:
        pruned_model = create_pruner(s, n_estimators = args.n_prune) 
        name = "{}-{}".format(s, args.n_prune)
        acc = test_model(pruned_model, XPrune, YPrune, XTest, YTest, model.estimators_, name)

    sys.exit(0)

if __name__ == '__main__':
    main()