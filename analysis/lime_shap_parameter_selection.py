"""
This Script calculates the possible permutations over LIME and SHAP parameters and iterates over them.

In each iteration, the model is trained with the current parameter combination then the model is applied to all the instance of test dataset.

Then the explanations are summarized for each permutation and their fidelity score is computed.

At the end the plots are generated for each explanation summary and the LIME and SHAP results are saved in a csv file.
"""

import sys
sys.path.insert(0, ".")

from numpy.random import permutation
from classifiers.adversarial_models import *
from utils.utils import experiment_summary, plt_barplot
from classifiers.racist_innocuous_models import racist_model_f, innocuous_model_psi, innocuous_model_psi_two
import lime.lime_tabular
import shap
from numpy import empty
import itertools
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Lime experiment
def lime_experiment_analysis(dataset_name='COMPAS', number_unrelated_column=1, categorical_features=[], feature_names=[],xtrain=empty((1,)),ytrain=empty((1,)),xtest=empty((1,)),racist_model=racist_model_f,innocuous_model=innocuous_model_psi):
    
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
	"""

    print ('---------------------')
    print (f"Beginning LIME {dataset_name} Experiments....")
    print('Dataset size: ','Train instances=',xtrain.shape[0],'Number of features=',xtrain.shape[1])
    print ("(These take some time to run because we have to generate explanations for every point in the test set) ") 
    print ('---------------------')
    
    
    # Making permutations for Lime Tabular Explainer
    print('creating LIME permutations ...')
    
    lime_params = {'kernel_width':[0.15,0.25,0.5,0.75,1.0], 
                    'discretizer':['quartile', 'decile'],
                    'discretize_continuous':[True,False],
                    'feature_selection':['auto'],
                    'sample_around_instance':[True,False]}

    keys, values = zip(*lime_params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # correcting false permutations
    for i in permutations:
        if i['discretize_continuous'] == False:
            i['discretizer'] = None

    print('Number of LIME permutations: ',len(permutations))

    
    # Train the adversarial model for LIME with f and psi 
    adv_lime = Adversarial_Lime_Model(racist_model, innocuous_model).train(xtrain, ytrain, 
                                                                                categorical_features=categorical_features, 
                                                                                feature_names=feature_names, 
                                                                                perturbation_multiplier=30)
    
    perm_values = []
    explanation_values = []
    fidelity_values = []
    
    # feeding different parameters to the explainer 
    for k in range(len(permutations)):
        print('Running LIME with parameters ({}/{}): {}'.format(k+1,len(permutations),permutations[k]))
        perm_values.append(permutations[k])
        
        # initializing explainer
        adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, 
                                                                feature_names=adv_lime.get_column_names(), 
                                                                categorical_features=categorical_features, 
                                                                kernel_width=permutations[k]['kernel_width']*xtrain.shape[1],
                                                                discretize_continuous=permutations[k]['discretize_continuous'],
                                                                discretizer=permutations[k]['discretizer'],
                                                                feature_selection=permutations[k]['feature_selection'],
                                                                sample_around_instance=permutations[k]['sample_around_instance'])

        # Generate explanation for every point in the test set
        explanations = []
        print('generating LIME explanations for test set ...')
        for i in tqdm(range(xtest.shape[0])):
            explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
        
        
        # Display Results
        print (f"LIME Ranks and Pct Occurances (1 corresponds to most important feature) for {number_unrelated_column} unrelated feature:")
        exp_summary = experiment_summary(explanations, feature_names)

        explanation_values.append(exp_summary)
        print (exp_summary)

        # Calculate fidelity
        fidelity_score = round(adv_lime.fidelity(xtest),2)
        fidelity_values.append(fidelity_score)
        print ("Fidelity:", fidelity_score)
        
        # Plot Results
        # plt_barplot(exp_summary,f'LIME result for {dataset_name} dataset with {number_unrelated_column} unrelated columns')
    
    # Plot Results
    features, score = [],[]
    for experiment in explanation_values:
        ft, sc = [],[]
        for _,v in experiment.items():
            for s in v:
                ft.append(s[0]); sc.append(s[1])
        features.append(ft)
        score.append(sc)

    y_pos = np.arange(len(features)) 

    px = 1/plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(7, 6, figsize=(2560*px, 1600*px))
    fig_title = f'LIME result for {dataset_name} dataset with {number_unrelated_column} unrelated columns and {len(perm_values)} parameters'
    fig.suptitle(fig_title, fontsize=20)


    for idx, (ft, sc, ax, param) in enumerate(zip(features, score, axes.flatten(), perm_values)):
        ax.barh(ft,sc)
        title = ''
        for key,value in param.items():
            title += str(key) + ':' + str(value) + ' '
        ax.set_title(str(title), fontsize=6)
        ax.invert_yaxis()  
        ax.set_xlabel('Score')
        plt.subplots_adjust(wspace=.7, hspace=.7)
    else:
        [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]

    fig.savefig("graphics/analysis/{}.png".format(fig_title))



    
    # Save Results
    with open('analysis/pickles/LIME_results.pkl', 'wb') as f:
        pickle.dump({'permutations':perm_values, 'explanations':explanation_values, 'fidelity':fidelity_values}, f)


    with open('analysis/pickles/LIME_results.pkl', 'rb') as f:
        lime_results = pickle.load(f)

    # Create a dataframe to compare the results
    df = pd.DataFrame(lime_results['permutations'])
    df['fidelity'] = lime_results['fidelity']
    df['lime_exp_first'] = [lime_results['explanations'][i].get(1)[0] for i in range(len(lime_results['explanations']))]
    df['lime_exp_second'] = [lime_results['explanations'][i].get(2) for i in range(len(lime_results['explanations']))]
    df_exp = df['lime_exp_second'].apply(pd.Series).add_prefix('lime_exp_second_')
    df = df.drop(['lime_exp_second'], axis=1)
    final_df = pd.concat([df, df_exp], axis=1)
    final_df.to_csv('analysis/analysis_results/LIME_results.csv')


# Shap experiment
def shap_experiment_analysis(dataset_name='COMPAS', number_unrelated_column=1, feature_names=[],xtrain=empty((1,)),ytrain=empty((1,)),xtest=empty((1,)),racist_model=racist_model_f,innocuous_model=innocuous_model_psi):
    
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
    """

    print ('---------------------')
    print (f"Beginning SHAP {dataset_name} Experiments....")
    print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
    print ('---------------------')
    
    
    # permutations of SHAP algorithms and number of clusters
    print('creating SHAP permutations ...')
    shap_parameters = {'algorithm':["auto", "permutation", "partition", "tree", "kernel", "sampling", "linear", "deep", "gradient"],
                        'masker':[10,15,20]}

    keys, values = zip(*shap_parameters.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print('Number of SHAP permutations: ',len(permutations))

    
    # initializing explainer
    adv_shap = Adversarial_Kernel_SHAP_Model(racist_model, innocuous_model).train(xtrain, ytrain, feature_names=feature_names)
    
    
    perm_values = []
    explanation_values = []
    fidelity_values = []

    # feeding different parameters to the explainer 
    for k in range(len(permutations)):
        print('Running SHAP with parameters ({}/{}): {}'.format(k+1,len(permutations),permutations[k]))
        perm_values.append(permutations[k])
        background_distribution = shap.kmeans(xtrain,permutations[k]['masker'])
        adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution, algorithm=permutations[k]['algorithm'])
        explanations = adv_kerenel_explainer.shap_values(xtest)

        # Generate explanation for every point in the test set
        print('generating SHAP explanations for test set ...')
        formatted_explanations = []
        for exp in explanations:
            formatted_explanations.append([(feature_names[i], exp[i]) for i in range(len(exp))])
        

        # Display Results
        print (f"SHAP Ranks and Pct Occurances {number_unrelated_column} unrelated features:")
        exp_summary = experiment_summary(formatted_explanations, feature_names)
        explanation_values.append(exp_summary)
        print (exp_summary)
        
        # calculate fidelity
        fidelity_score = round(adv_shap.fidelity(xtest),2)
        print ("Fidelity:",fidelity_score)
        fidelity_values.append(fidelity_score)

        # Plot Results
        # plt_barplot(exp_summary,f'SHAP result for {dataset_name} with {number_unrelated_column} unrelated column')

    # Plot Results
    features, score = [],[]
    for experiment in explanation_values:
        ft, sc = [],[]
        for _,v in experiment.items():
            for s in v:
                ft.append(s[0]); sc.append(s[1])
        features.append(ft)
        score.append(sc)

    y_pos = np.arange(len(features)) 

    px = 1/plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(7, 6, figsize=(2560*px, 1600*px))
    fig_title = f'SHAP result for {dataset_name} dataset with {number_unrelated_column} unrelated columns and {len(perm_values)} parameters'
    fig.suptitle(fig_title, fontsize=20)



    for idx, (ft, sc, ax, param) in enumerate(zip(features, score, axes.flatten(), perm_values)):
        ax.barh(ft,sc)
        title = ''
        for key,value in param.items():
            title += str(key) + ':' + str(value) + ' '
        ax.set_title(str(title), fontsize=6)
        ax.invert_yaxis()  
        ax.set_xlabel('Score')
        plt.subplots_adjust(wspace=.7, hspace=.7)
    else:
        [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]

    fig.savefig("graphics/analysis/{}.png".format(fig_title))

    # Save Results
    with open('analysis/pickles/SHAP_results.pkl', 'wb') as f:
        pickle.dump({'permutations':perm_values, 'explanations':explanation_values, 'fidelity':fidelity_values}, f)
        

    with open('analysis/pickles/SHAP_results.pkl', 'rb') as f:
        shap_results = pickle.load(f)

    # Create a dataframe to compare the results
    df = pd.DataFrame(shap_results['permutations'])
    df['fidelity'] = shap_results['fidelity']
    df['shap_exp_first'] = [shap_results['explanations'][i].get(1)[0] for i in range(len(shap_results['explanations']))]
    df['shap_exp_second'] = [shap_results['explanations'][i].get(2) for i in range(len(shap_results['explanations']))]
    df_exp = df['shap_exp_second'].apply(pd.Series).add_prefix('shap_exp_second_')
    df = df.drop(['shap_exp_second'], axis=1)
    final_df = pd.concat([df, df_exp], axis=1)
    final_df.to_csv('analysis/analysis_results/SHAP_results.csv')
