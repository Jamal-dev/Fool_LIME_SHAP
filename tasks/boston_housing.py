"""
In this script both lime and shap are called, you can choose one or both.
 * Run the file and the Boston housing experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   both LIME and SHAP explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
 	    1: [('length_of_stay', 0.002592352559948153), ('unrelated_column', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to SHAP yielding only 0 shapley values 
"""

from cProfile import run
from importlib.resources import path
import sys
from pathlib import Path
import os
file_runing_dir = os.path.dirname(os.path.abspath(__file__))
path_main = Path(file_runing_dir)/ Path("..")
sys.path.append(str(path_main))

from classifiers.racist_innocuous_models import racist_model_f, innocuous_model_psi
from classifiers.adversarial_models import *
from classifiers.lime_shap import lime_experiment, shap_experiment
from analysis.lime_shap_parameter_selection import lime_experiment_analysis, shap_experiment_analysis

from utils.dataset_preprocessor import preprocess_boston_housing
from utils.helper_routines import load_transfomed_data
from utils.settings import params

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
  description = ["Main function will run on Boston Housing: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html",
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



def run_boston():
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

  # define biased feature
  biased_feature_name = 'B'

  # load data and transformed it
  xtrain,ytrain,xtest,ytest,feature_names=load_transfomed_data(biased_feature_name=biased_feature_name
                                                              , preprocess_fun=preprocess_boston_housing
                                                              , unrelated_columns=1
                                                              , test_size=0.1
                                                              , params = params)
  print(f"feature names: {feature_names}")

  # defining the categorical column
  categorical_features = [feature_names.index('unrelated_column_one')]

  # the index of biased feature
  biased_feature_idx = feature_names.index(biased_feature_name)

  # unrelated_column_one index
  unrelated_feature_idx = feature_names.index('unrelated_column_one')

  # dataset name
  dataset_name = "Boston_Housing"

  return(xtrain,ytrain,xtest,ytest,feature_names,categorical_features,biased_feature_idx, unrelated_feature_idx,dataset_name)


# Lime experiment
def lime_exp(analysis_mode=False):
    """
    Run through experiments for LIME on Boston Housing using the unrelated features.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features

    Parameters:
    analysis_mode: bool
      if True LIME and SHAP will run in the analysis mode which is considering all the permutations
  
    Returns:
    ----------
    """

    if analysis_mode==False:
      lime_model = lime_experiment(dataset_name='Boston Housing', 
                      number_unrelated_column=1, 
                      categorical_features= categorical_features,
                      feature_names= feature_names,
                      xtrain=xtrain,
                      ytrain=ytrain,
                      xtest=xtest,
                      racist_model=racist_model_f(biased_feature_idx),
                      innocuous_model=innocuous_model_psi(unrelated_feature_idx))
  
    else:
      lime_model = lime_experiment_analysis(dataset_name='Boston Housing', 
                      number_unrelated_column=1, 
                      categorical_features=categorical_features,
                      feature_names=feature_names,
                      xtrain=xtrain,
                      ytrain=ytrain,
                      xtest=xtest,
                      racist_model=racist_model_f(biased_feature_idx),
                      innocuous_model=innocuous_model_psi(unrelated_feature_idx))
    return lime_model


def shap_exp(analysis_mode=False):
    """
    Run through experiments for SHAP on boston housing using the unrelated feature.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features

    Parameters:
    analysis_mode: bool
      if True LIME and SHAP will run in the analysis mode which is considering all the permutations
  
    Returns:
    ----------
    """

    if analysis_mode==False:
      shap_model = shap_experiment(dataset_name='Boston Housing', 
                      number_unrelated_column=1,  
                      feature_names=feature_names,
                      xtrain= xtrain,
                      ytrain=ytrain,
                      xtest=xtest,
                      racist_model=racist_model_f(biased_feature_idx),
                      innocuous_model=innocuous_model_psi(unrelated_feature_idx))
      
    else:
      shap_model = shap_experiment_analysis(dataset_name='Boston Housing', 
                      number_unrelated_column=1,  
                      feature_names=feature_names,
                      xtrain=xtrain,
                      ytrain=ytrain,
                      xtest=xtest,
                      racist_model=racist_model_f(biased_feature_idx),
                      innocuous_model=innocuous_model_psi(unrelated_feature_idx))
    return shap_model


def boston_exp(analysis_mode=False):
    """
    Run through experiments for both LIME and SHAP on Boston Housing using the unrelated feature.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features

    Parameters:
    analysis_mode: bool
      if True LIME and SHAP will run in the analysis mode which is considering all the permutations
  
    Returns:
    ----------
    """

    global xtrain,ytrain,xtest,ytest,feature_names,categorical_features,biased_feature_idx, unrelated_feature_idx, dataset_name
    xtrain,ytrain,xtest,ytest,feature_names,categorical_features,biased_feature_idx, unrelated_feature_idx,dataset_name = run_boston()
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
      pdp_model.plot_pdp()

def main(analysis_mode):
    global run_PDP, run_lime, run_shap
    run_PDP, run_lime, run_shap = parameter_parser()
    
    global xtrain,ytrain,xtest,ytest,feature_names,categorical_features,biased_feature_idx,unrelated_feature_idx,  dataset_name
    xtrain,ytrain,xtest,ytest,feature_names,categorical_features,biased_feature_idx,unrelated_feature_idx, dataset_name = run_boston()
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
  main(analysis_mode=True)
