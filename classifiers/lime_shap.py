import sys
from pathlib import Path
import os
file_runing_dir = os.path.dirname(os.path.abspath(__file__))
path_main = Path(file_runing_dir)/ Path("..")
sys.path.append(str(path_main))

from classifiers.adversarial_models import *
from utils.utils import experiment_summary, plt_barplot
from classifiers.racist_innocuous_models import racist_model_f, innocuous_model_psi, innocuous_model_psi_two
import lime.lime_tabular
import shap
from numpy import empty
from operator import itemgetter

# Lime experiment
def lime_experiment(dataset_name='COMPAS', number_unrelated_column=1, categorical_features=[], feature_names=[],xtrain=empty((1,)),ytrain=empty((1,)),xtest=empty((1,)),racist_model=racist_model_f,innocuous_model=innocuous_model_psi):
    """
	Run through experiments for LIME on dataset_name using both one or two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
    Parameters
    ----------
    dataset_name            : string               , name of your dataset
    number_unrelated_column : int                  , specify if you have one unrelated column or two
    categorical_features    : list                 , list of of all your categorical features indexs
    feature_names           : list                 , list of names of all features
    xtrain                  : ndarray              , train dataset (number of instances, feature 1, feature 2, ... feature n)
    ytrain                  : ndarray              , train labeled dataset (number of instances,)
    xtest                   : ndarray              , test dataset (number of instances, feature 1, feature 2, ... feature n)
    racist_model            : racist_model_f       , classifier estimator for racist model, defined as https://scikit-learn.org/stable/glossary.html#term-_estimator_type
    innocuous_model         : innocuous_model_psi  , classifier estimator for innocuous model, defined as https://scikit-learn.org/stable/glossary.html#term-_estimator_type
    Returns 
    ----------
    adv_lime                : Adversarial_Lime_Model, adversial lime model
	"""

    print ('---------------------')
    print (f"Beginning LIME {dataset_name} Experiments....")
    print ("(These take some time to run because we have to generate explanations for every point in the test set) ") 
    print ('---------------------')

    print('Dataset size: ','Train instances=',xtrain.shape[0],'Number of features=',xtrain.shape[1])

    # Train the adversarial model for LIME with f and psi 
    adv_lime = Adversarial_Lime_Model(racist_model, innocuous_model).train(xtrain, ytrain, 
                                                                                categorical_features=categorical_features, 
                                                                                feature_names=feature_names, 
                                                                                perturbation_multiplier=30)
    adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, 
                                                            sample_around_instance=True, 
                                                            feature_names=adv_lime.get_column_names(), 
                                                            categorical_features=categorical_features, 
                                                            discretize_continuous=False)
                                                
    explanations = []
    for i in range(xtest.shape[0]):
        explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

    # Display Results
    print (f"LIME Ranks and Pct Occurances (1 corresponds to most important feature) for {number_unrelated_column} unrelated feature:")
    exp_summary = experiment_summary(explanations, feature_names)
    print (exp_summary)
    print('------------------------------------------------------------------------------------------------------------------')
    plt_barplot(exp_summary,f'LIME result for {dataset_name} dataset with {number_unrelated_column} unrelated columns')
    print ("Fidelity:", round(adv_lime.fidelity(xtest),2))
    return adv_lime


# Shap experiment
def shap_experiment(dataset_name='COMPAS', number_unrelated_column=1, feature_names=[],xtrain=empty((1,)),ytrain=empty((1,)),xtest=empty((1,)),racist_model=racist_model_f,innocuous_model=innocuous_model_psi):
    """
    Run through experiments for SHAP on dataset_name using both one or two unrelated features.
    * This may take some time given that we iterate through every point in the test set
    * We print out the rate at which features occur in the top three features
    Parameters
    ----------
    dataset_name            : string               , name of your dataset
    number_unrelated_column : int                  , specify if you have one unrelated column or two
    feature_names           : list                 , list of names of all features
    xtrain                  : ndarray              , train dataset (number of instances, feature 1, feature 2, ... feature n)
    ytrain                  : ndarray              , train labeled dataset (number of instances,)
    xtest                   : ndarray              , test dataset (number of instances, feature 1, feature 2, ... feature n)
    racist_model            : racist_model_f       , classifier estimator for racist model, defined as https://scikit-learn.org/stable/glossary.html#term-_estimator_type
    innocuous_model         : innocuous_model_psi  , classifier estimator for innocuous model, defined as https://scikit-learn.org/stable/glossary.html#term-_estimator_type
    Returns 
    ----------
    adv_shap                : Adversarial_Kernel_SHAP_Model, adversial SHAP model
    """

    print ('---------------------')
    print (f"Beginning SHAP {dataset_name} Experiments....")
    print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
    print ('---------------------')
    
    #Setup SHAP
    background_distribution = shap.kmeans(xtrain,10)
    adv_shap = Adversarial_Kernel_SHAP_Model(racist_model, innocuous_model).train(xtrain, ytrain, feature_names=feature_names)
    adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
    explanations = adv_kerenel_explainer.shap_values(xtest)

    # format for display
    formatted_explanations = []
    for exp in explanations:
        formatted_explanations.append([(feature_names[i], exp[i]) for i in range(len(exp))])

    print (f"SHAP Ranks and Pct Occurances {number_unrelated_column} unrelated features:")
    exp_summary = experiment_summary(formatted_explanations, feature_names)
    print (exp_summary)
    print('------------------------------------------------------------------------------------------------------------------')
    plt_barplot(exp_summary,f'SHAP result for {dataset_name} with {number_unrelated_column} unrelated column')
    print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

    return adv_shap
