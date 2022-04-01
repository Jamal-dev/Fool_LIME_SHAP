"""
In this script both lime and shap are called, you can choose one or both.
 * Run the file and the COMPAS experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   both LIME and SHAP explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
 	    1: [('length_of_stay', 0.002592352559948153), ('unrelated_column_one', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to SHAP yielding only 0 shapley values 
"""

from importlib.resources import path
import sys
from pathlib import Path
import os
file_runing_dir = os.path.dirname(os.path.abspath(__file__))
path_main = Path(file_runing_dir)/ Path("..")
sys.path.append(str(path_main))

from classifiers.racist_innocuous_models import racist_model_f, innocuous_model_psi
from classifiers.adversarial_models import *

from utils.settings import params
import utils.settings as get_par
from utils.dataset_preprocessor import preprocess_german
from utils.utils import one_hot_encode
from utils.helper_routines import load_transfomed_data

from classifiers.lime_shap import lime_experiment, shap_experiment
from analysis.lime_shap_parameter_selection import lime_experiment_analysis, shap_experiment_analysis


import numpy as np
import matplotlib.pyplot as plt

from utils.utils import tab_printer
from utils.PDP import PDP
import argparse

def bool_v(var):
  if var ==0:
    return False
  else:
    return True

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    description = ["Main function will run on German Credit: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)",
                    "By default only LIME and SHAP will run but if you change option to --run_lime then only LIME will run. Also you can trun on or OFF PDP as well."]
    descp = ""
    for d in description:
        descp = descp+ d

    parser = argparse.ArgumentParser(description=descp)

    parser.add_argument("--run_PDP",
                        type=int,
                        nargs="?",
                        default=0,
                  help="non-zero value will run the PDP on the classifier as well")
    parser.add_argument("--run_lime",
                        type=int,
                        nargs="?",
                        default=0,
                  help="If non-zero it will only run LIME. Better for large datasets because SHAP will take more time")
    parser.add_argument("--run_shap",
                        type=int,
                        nargs="?",
                        default=0,
                  help="If non-zero it will only run SHAP.")
    args = parser.parse_args()
    run_PDP = bool_v(args.run_PDP)
    run_lime = bool_v(args.run_lime)
    run_shap = bool_v(args.run_shap)
      
    tab_printer(args)
    return run_PDP, run_lime, run_shap



# overriding the predict_proba function for the innocous
class innocuous_model_psi_child(innocuous_model_psi):
    def __init__(self,uc):
        super(innocuous_model_psi_child,self).__init__(uc)
        
     # Decision rule: classify according to loan rate indc
    def predict_proba(self, X): 
        # print('variables: ',self._positive_outcome,self._negative_outcome,self.unrelated_feature_idx)
        return one_hot_encode(np.array([self._negative_outcome if x[self._unrelated_feature_idx] > mean_lrpi else self._positive_outcome for x in X]))


def run_german():
    """
    The biased feature column is specified and the test,train split and feature names are computed.

    Parameters:
    ----------
    
    Returns:
    ----------
    xtrain: 
    ytrain: 
    xtest:
    ytest:
    feature_names:
    categorical_features:
    biased_feature_idx:
    unrelated_feature_idx:
    """

    # biased feature
    biased_feature_name = 'Gender'
    
    # load data and transformed it
    xtrain,ytrain,xtest,ytest,feature_names=load_transfomed_data(biased_feature_name=biased_feature_name
                                                                , preprocess_fun=preprocess_german
                                                                , unrelated_columns=1
                                                                , test_size=0.1
                                                                , params = params
                                                                , unrelated_column_from_feature="LoanRateAsPercentOfIncome")
    # Categorical features
    categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone',
                    'CheckingAccountBalance_geq_0','CheckingAccountBalance_geq_200',
                    'SavingsAccountBalance_geq_100','SavingsAccountBalance_geq_500',
                    'MissedPayments','NoCurrentLoan','CriticalAccountOrLoansElsewhere',
                    'OtherLoansAtBank','OtherLoansAtStore','HasCoapplicant','HasGuarantor',
                    'OwnsHouse','RentsHouse','Unemployed','YearsAtCurrentJob_lt_1',
                    'YearsAtCurrentJob_geq_4','JobClassIsSkilled']
    
    # Getting the index of cateogorical features
    categorical_features = [feature_names.index(c) for c in categorical]

    # calculating the mean of unrelated feature as a threshold for the innocuous model
    mean_lrpi = np.mean(xtrain[:,get_par.unrelated_feature_idx])

    # the index of biased feature
    biased_feature_idx = feature_names.index(biased_feature_name)
    
    # unrelated_column_one index
    unrelated_feature_idx = feature_names.index("LoanRateAsPercentOfIncome")

    # dataset_name
    dataset_name = "German_Credit"

    return(xtrain,ytrain,xtest,ytest,feature_names,categorical_features,mean_lrpi, biased_feature_idx, unrelated_feature_idx,dataset_name)


# Lime experiment
def lime_exp(analysis_mode=False):
    """
    Run through experiments for LIME on German Credit using the unrelated feature.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features

    Parameters:
    analysis_mode: bool
      if True LIME and SHAP will run in the analysis mode which is considering all the permutations
  
    Returns:
    ----------
    """

    if analysis_mode==False:
        l = lime_experiment(dataset_name='German credit', 
                        number_unrelated_column=1, 
                        categorical_features=categorical_features, 
                        feature_names=feature_names,
                        xtrain=xtrain,
                        ytrain=ytrain,
                        xtest=xtest,
                        racist_model=racist_model_f(biased_feature_idx),
                        innocuous_model=innocuous_model_psi_child(unrelated_feature_idx))

    else:
        l = lime_experiment_analysis(dataset_name='German credit', 
                        number_unrelated_column=1, 
                        categorical_features=categorical_features, 
                        feature_names= feature_names,
                        xtrain= xtrain,
                        ytrain= ytrain,
                        xtest= xtest,
                        racist_model=racist_model_f(biased_feature_idx),
                        innocuous_model=innocuous_model_psi_child(unrelated_feature_idx))
    return l
    

def shap_exp(analysis_mode=False):
    """
    Run through experiments for SHAP on German Credit using the unrelated feature.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features

    Parameters:
    analysis_mode: bool
      if True LIME and SHAP will run in the analysis mode which is considering all the permutations
  
    Returns:
    ----------
    """

    if analysis_mode==False:
        s = shap_experiment(dataset_name='German credit', 
                        number_unrelated_column=1,  
                        feature_names=feature_names,
                        xtrain=xtrain,
                        ytrain=ytrain,
                        xtest=xtest,
                        racist_model=racist_model_f(biased_feature_idx),
                        innocuous_model=innocuous_model_psi_child(unrelated_feature_idx))

    else:
        s = shap_experiment_analysis(dataset_name='German credit', 
                        number_unrelated_column=1,  
                        feature_names=feature_names,
                        xtrain=xtrain,
                        ytrain=ytrain,
                        xtest=xtest,
                        racist_model=racist_model_f(biased_feature_idx),
                        innocuous_model=innocuous_model_psi_child(unrelated_feature_idx))
    return s


def german_exp(analysis_mode=False):
    """
    Run through experiments for both LIME and SHAP on German Credit using the unrelated feature.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features

    Parameters:
    analysis_mode: bool
      if True LIME and SHAP will run in the analysis mode which is considering all the permutations
  
    Returns:
    ----------
    """

    global xtrain,ytrain,xtest,ytest,feature_names,categorical_features,mean_lrpi, biased_feature_idx, unrelated_feature_idx
    xtrain,ytrain,xtest,ytest,feature_names,categorical_features,mean_lrpi, biased_feature_idx, unrelated_feature_idx = run_german()
    lime_exp(analysis_mode=analysis_mode)
    shap_exp(analysis_mode=analysis_mode)
    plt.show()

def check_PDP(run_PDP,query_model,lime_shap:bool,model_num:int):
    """
    This will check if PDP analysis is needed to run or not
    

    Parameters:
    run_PDP: bool
      if True PDP will run
    query_model: classifier
      it must have predict function
    lime_shap: bool
      if True then image will be saved with lime name otherwise shap
    model_num: int
      This will be added into into the image so that you can save multiple images

    Returns:
    ----------
    """
    if run_PDP:
      feature_idx = list(range(len(feature_names)))
      if lime_shap:
        ana_name = "lime"
      else:
        ana_name = "shap"
      pdp_model = PDP(X=xtest,model=query_model.predict,
                feature_idx=feature_idx, 
                labels=feature_names, 
                width=8.0, 
                height = 8.0,
              xlabel_fontsize = 24.0, title_fontsize = 36.0,
              figname=f'pdp_{dataset_name}_{ana_name}_model_{model_num}.pdf',
              dataset_name=dataset_name)
      pdp_model.plot_pdp();

def main(analysis_mode):
    global run_PDP, run_lime, run_shap
    run_PDP, run_lime, run_shap = parameter_parser()
    
    global xtrain,ytrain,xtest,ytest,feature_names,categorical_features,mean_lrpi, biased_feature_idx, unrelated_feature_idx, dataset_name
    xtrain,ytrain,xtest,ytest,feature_names,categorical_features,mean_lrpi, biased_feature_idx, unrelated_feature_idx, dataset_name = run_german()
    if run_lime:
      lime_model = lime_exp(analysis_mode=analysis_mode)
      check_PDP(run_PDP,lime_model,lime_shap=True,model_num=1)
      plt.show()
      return
    if run_shap:
      shap_model = shap_exp(analysis_mode=analysis_mode)
      check_PDP(run_PDP,shap_model,lime_shap=False,model_num=1)
      plt.show()
      return

    lime_model = lime_exp(analysis_mode=analysis_mode)
    shap_model = shap_exp(analysis_mode=analysis_mode)
    check_PDP(run_PDP,lime_model,lime_shap=True,model_num=1)
    check_PDP(run_PDP,shap_model,lime_shap=False,model_num=1)
    plt.show()



if __name__ == "__main__":
    main(analysis_mode=False)
