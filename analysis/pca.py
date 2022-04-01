"""
This script preprocesses the datasets using dataset_preprocessor.py and then removes categorical features.

Then the PCA function mimics the synthetic data generation in LIME by adding random noise to numerical features an adding them as new instances.

Finally it calculates the PCA values (2D) for both the real and synthetic data and plots them.
"""

from classifiers.adversarial_models import *
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.settings import params
from utils.dataset_preprocessor import *
import sys
from pathlib import Path
import os
file_runing_dir = os.path.dirname(os.path.abspath(__file__))
path_main = Path(file_runing_dir) / Path("..")
sys.path.append(str(path_main))


def prepare_compas():
    """ Preparing COMPAS dataset for PCA experiment.
    Preparation includes:
    1. applying dataset-specific preprocessing function (dataset_preprocessor.py)
    2. removing categorical features

    Parameters
    ----------
    None

    Returns 
    ----------
    return preprocessed numerical features of the dataset
    """

    X, y, cols = preprocess_compas(params)
    features = [c for c in X]

    X = X.values
    c_cols = [features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index(
        'two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")]

    X = np.delete(X, c_cols, axis=1)

    ss = StandardScaler().fit(X)
    X = ss.transform(X)

    return(X)


def prepare_boston():
    """ Preparing Boston housing dataset for PCA experiment.
    Preparation includes:
    1. applying dataset-specific preprocessing function (dataset_preprocessor.py)
    2. removing categorical features

    Parameters
    ----------
    None

    Returns 
    ----------
    return preprocessed numerical features of the dataset
    """

    X, y, cols = preprocess_boston_housing(params)
    X = X.values

    ss = StandardScaler().fit(X)
    X = ss.transform(X)

    return(X)


def prepare_cc():
    """ Preparing Communities and Crime dataset for PCA experiment.
    Preparation includes:
    1. applying dataset-specific preprocessing function (dataset_preprocessor.py)
    2. removing categorical features

    Parameters
    ----------
    None

    Returns 
    ----------
    return preprocessed numerical features of the dataset
    """

    X, y, cols = preprocess_cc(params)
    X = X.values

    ss = StandardScaler().fit(X)
    X = ss.transform(X)

    return(X)


def prepare_german():
    """ Preparing German Credit dataset for PCA experiment.
    Preparation includes:
    1. applying dataset-specific preprocessing function (dataset_preprocessor.py)
    2. removing categorical features

    Parameters
    ----------
    None

    Returns 
    ----------
    return preprocessed numerical features of the dataset
    """

    X, y, cols = preprocess_german(params)

    for i in cols:
        if len(X[i].unique()) <= 2:
            X.drop(i, axis=1, inplace=True)

    X = X.values

    ss = StandardScaler().fit(X)
    X = ss.transform(X)

    return(X)


def prepare_student():
    """ Preparing Students performance dataset for PCA experiment.
    Preparation includes:
    1. applying dataset-specific preprocessing function (dataset_preprocessor.py)
    2. removing categorical features

    Parameters
    ----------
    None

    Returns 
    ----------
    return preprocessed numerical features of the dataset
    """

    X, y, cols = preprocess_student_performance(params)

    for i in cols:
        if len(X[i].unique()) <= 2:
            X.drop(i, axis=1, inplace=True)

    X = X.values

    ss = StandardScaler().fit(X)
    X = ss.transform(X)

    return(X)


def pca_experiment(prepare_dataset_func, dataset_name):
    """ Applies random noise to the dataset and calculates PCA values. (2D)
    Finally it plots the PCA values for both the real and synthetic data.

    Parameters
    ----------
    prepare_dataset_func: function
        function that prepares the dataset for PCA experiment
    dataset_name: string
        name of the dataset

    Returns 
    ----------
    """

    X = prepare_dataset_func()
    r = []
    for _ in range(1):
        p = np.random.normal(0, 1, size=X.shape)
        X_p = X + p
        r.append(X_p)

    r = np.vstack(r)
    p = [1 for _ in range(len(r))]
    iid = [0 for _ in range(len(X))]

    all_x = np.vstack((r, X))
    all_y = np.array(p + iid)

    pca = PCA(n_components=2)
    results = pca.fit_transform(all_x)

    plt.title(
        'PCA comparison of real (orange) vs synthetic (blue) data of ' + dataset_name)
    if dataset_name == 'student':
        plt.scatter(results[:100, 0], results[:100, 1], alpha=.5)
        plt.scatter(results[-100:, 0], results[-100:, 1], alpha=.5)

    else:
        plt.scatter(results[:500, 0], results[:500, 1], alpha=.5)
        plt.scatter(results[-500:, 0], results[-500:, 1], alpha=.5)

    print(f"{dataset_name} is done!")

    path_fig = path_main / \
        Path(f'graphics/PCA/pca_experiment_{dataset_name}.png')
    plt.savefig(path_fig)

    plt.show()


if __name__ == "__main__":
    pca_experiment(prepare_compas, 'compas')
    pca_experiment(prepare_boston, 'boston')
    pca_experiment(prepare_cc, 'cc')
    pca_experiment(prepare_german, 'german')
    pca_experiment(prepare_student, 'student')
