import sys
from pathlib import Path
import os
file_runing_dir = os.path.dirname(os.path.abspath(__file__))
path_main = Path(file_runing_dir)/ Path("..")
sys.path.append(str(path_main))

from utils.settings import params
from utils.utils import one_hot_encode
import numpy as np
# to make the estimator as a classifier for scikitlearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
"""
     The models f and psi for datasets such as GERMAN, COMPASS, andBoston Housing.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
"""
"""
    If you want to overridde predict function then use the following exampe code into your task file

class racist_model_f_child(racist_model_f):
     def predict(self,X):
         return np.array([params.negative_outcome if x[get_par.biased_feature_idx] > 0 else params.positive_outcome for x in X])
"""
# the biased model 
class racist_model_f(BaseEstimator,ClassifierMixin):
    
    def __init__(self,ufi=None):
        self._biased_feature_idx = ufi
        self._positive_outcome = params.positive_outcome
        self._negative_outcome = params.negative_outcome
    @property
    def biased_feature_idx(self):
        """Getter for unrelated_feature_idx"""
        return self._biased_feature_idx

    @biased_feature_idx.setter
    def biased_feature_idx(self, value):
        """Setter for unrelated_feature_idx"""
        self._biased_feature_idx = value
    @property
    def positive_outcome(self):
        """Getter for positive outcome"""
        return self._positive_outcome
    @positive_outcome.setter
    def positive_outcome(self,value):
        """Setter for positive outcome"""
        self._positive_outcome = value
    @property
    def negative_outcome(self):
        """Getter for negative outcome"""
        return self._negative_outcome
    @negative_outcome.setter
    def negative_outcome(self,value):
        """Getter for negative outcome"""
        self._negative_outcome = value
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([self._negative_outcome if x[self._biased_feature_idx] > 0 else self._positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    # this is just added so that estimator can be treated as estimator for the scikitlearn use
    def fit(self,X,y):
        # check if they have correct shape
        X, y = check_X_y(X, y)
        # store classes
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self
    
# the display model with one unrelated feature
class innocuous_model_psi(BaseEstimator,ClassifierMixin):
    def __init__(self,ufi=None):
        self._unrelated_feature_idx = ufi
        self._positive_outcome = params.positive_outcome
        self._negative_outcome = params.negative_outcome
    @property
    def unrelated_feature_idx(self):
        """Getter for unrelated_feature_idx"""
        return self._unrelated_feature_idx

    @unrelated_feature_idx.setter
    def unrelated_feature_idx(self, value):
        """Setter for unrelated_feature_idx"""
        self._unrelated_feature_idx = value
    @property
    def positive_outcome(self):
        """Getter for positive outcome"""
        return self._positive_outcome
    @positive_outcome.setter
    def positive_outcome(self,value):
        """Setter for positive outcome"""
        self._positive_outcome = value
    @property
    def negative_outcome(self):
        """Getter for negative outcome"""
        return self._negative_outcome
    @negative_outcome.setter
    def negative_outcome(self,value):
        """Getter for negative outcome"""
        self._negative_outcome = value
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        return one_hot_encode(np.array([self._negative_outcome if x[self._unrelated_feature_idx] > 0 else self._positive_outcome for x in X]))

# the display model with two unrelated features
class innocuous_model_psi_two(BaseEstimator,ClassifierMixin):
    def __init__(self,ufi=None, ufi2 = None):
        self._unrelated_feature_idx = ufi
        self._positive_outcome = params.positive_outcome
        self._negative_outcome = params.negative_outcome
        self._unrelated_feature_idx_2 = ufi2
    @property
    def unrelated_feature_idx(self):
        """Getter for unrelated_feature_idx"""
        return self._unrelated_feature_idx

    @unrelated_feature_idx.setter
    def unrelated_feature_idx(self, value):
        """Setter for unrelated_feature_idx"""
        self._unrelated_feature_idx = value
    @property
    def unrelated_feature_idx_2(self):
        """Getter for unrelated_feature_idx_2"""
        return self._unrelated_feature_idx_2

    @unrelated_feature_idx_2.setter
    def unrelated_feature_idx_2(self, value):
        """Setter for unrelated_feature_idx_2"""
        self._unrelated_feature_idx_2 = value
    @property
    def positive_outcome(self):
        """Getter for positive outcome"""
        return self._positive_outcome
    @positive_outcome.setter
    def positive_outcome(self,value):
        """Setter for positive outcome"""
        self._positive_outcome = value
    @property
    def negative_outcome(self):
        """Getter for negative outcome"""
        return self._negative_outcome
    @negative_outcome.setter
    def negative_outcome(self,value):
        """Getter for negative outcome"""
        self._negative_outcome = value
    def predict_proba(self, X):
        # Using 0.5 to make it easier to detect decision boundary on perturbation
        A = np.where(X[:,self._unrelated_feature_idx] > .5, self._positive_outcome, self._negative_outcome)
        B = np.where(X[:,self._unrelated_feature_idx_2] < -.5, self._positive_outcome, self._negative_outcome)
        preds = np.logical_xor(A, B).astype(int)
        return one_hot_encode(preds)
