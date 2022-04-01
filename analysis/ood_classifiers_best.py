import sys
from pathlib import Path
import os
file_runing_dir = os.path.dirname(os.path.abspath(__file__))
path_main = Path(file_runing_dir)/ Path("..")
sys.path.append(str(path_main))

from classifiers.adversarial_models import *
from classifiers.racist_innocuous_models import racist_model_f, innocuous_model_psi, innocuous_model_psi_two


from utils.settings import params
from tasks.compas import run_compas
from utils.utils import experiment_summary, tab_printer

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import joblib
import pandas as pd 
import lime
import shap

import seaborn as sns
import csv
import argparse



def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    description = ["Run grid search for the OOD classifier.",
                    "\nGrid search will run on the 9 different classifiers:",
                    "\nLogisticRegression,",
                    "\nSVC,",
                    "\nKNeighborsClassifier,",
                    "\nGaussianNB,",
                    "\nMultinomialNB,",
                    "\nDecisionTreeClassifier,",
                    "\nRandomForestClassifier,",
                    "\nGradientBoostingClassifier,",
                    "\nMLPClassifier.",
                    "\n\n Parameter range for each classifer can be changed in parameters_classifiers() function.",
                    "\n If you want to find the classifer instead of COMPAS; you have to import first from tasks for instance run_cc(community crime) and change it in the line 343. And also, you have to see if the unrelated index is 1 or 2"]
    descp = ""
    for d in description:
        descp = descp+ d

    parser = argparse.ArgumentParser(description=descp)

    parser.add_argument("--lime_shap",
                        type=int,
                        nargs="?",
                        default=0,
	                help="0 will run the SHAP while non-zero will run the lime")
    args = parser.parse_args()
    lime_shap = args.lime_shap
    tab_printer(args)
    return lime_shap

# score function for shap
def shap_scorer_f1(estimator,xtrain,ytrain):
    
    adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(biased_feature_idx), innocuous_model_psi(unrelated_feature_idx_1)).\
                        train(xtrain, ytrain, 
                              estimator=estimator, 
                              feature_names=feature_names)
    y = adv_shap.ood_training_task_ability[0]
    yhat = adv_shap.ood_training_task_ability[1]
    # f1_score
    f1score = f1_score(y, yhat, average='micro') 
    
    # now for fct part
    adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
    explanations = adv_kerenel_explainer.shap_values(xtest[:50])

    formatted_explanations = []
    for exp in explanations:
        formatted_explanations.append([(feature_names[i], exp[i]) for i in range(len(exp))])
    summary = experiment_summary(formatted_explanations, feature_names)
    pct_occur = [0]
    # it can be in 1,2,3 if one is intersted in finding in the top 3 features
    for indc in [1]:
        found = False
        for tup in summary[indc]:
            if tup[0] == biased_feature_name:
                pct_occur.append(sum([pct_occur[-1], tup[1]]))
                found = True

        if not found:
            pct_occur.append(pct_occur[-1])

    pct_occur = pct_occur[1:]
    pct_first = pct_occur[0]
    # uncomment them if you change the line: for indc in [1,2,3]
#     pct_second = pct_occur[1]
#     pct_third = pct_occur[2]

    # classifier name
    for k,v in clfs.items():
        ch = str(estimator['classifier']).find(k)
        if ch==0:
            clf_name = k
#     clf_name = estimator.__class__.__name__
    # appending into the list
    fname = Path(file_runing_dir)/Path(f'log/{clf_name}.txt')
    print(f"{f1score},{pct_first}",file = open(fname,'a'))
    return f1score

# score function for LIME
def lime_scorer_f1(estimator,xtrain,ytrain):
    adv_lime = Adversarial_Lime_Model(racist_model_f(biased_feature_idx), innocuous_model_psi(unrelated_feature_idx_1)).\
                            train(xtrain, ytrain, 
                                  estimator=estimator, 
                                  feature_names=feature_names, 
                                  perturbation_multiplier=1)
    y = adv_lime.ood_training_task_ability[0]
    yhat = adv_lime.ood_training_task_ability[1]
    # f1 score
    f1score = f1_score(y, yhat, average='micro')
    
    formatted_explanations = []
    query_instances = xtest
    # For testing purposes
#     query_instances = np.array([[ 2.33857338, -0.91534459, -1.03332413, -0.68373012, \
#                         -0.34092398, -1.34669094, \
#                    1.34669094, -0.48141324,  0.48141324, -1.01049803,  1.00180213],\
#                  [ 0.12419497, -0.91534459, -1.03332413, -0.26805095, -0.31917296 , \
#                   -1.34669094,  1.34669094, -0.48141324,  0.48141324, -1.01049803, \
#                   -0.99820111]])
    for i in range(query_instances.shape[0]):
        # previously there was xtest instead of xtrain
        exp = adv_explainer.explain_instance(query_instances[i], adv_lime.predict_proba).as_list()
        formatted_explanations.append(exp)
#         print('formated_explanations:',formatted_explanations, file=open('analysis/log/success_err_adv_explainer.txt', 'a'))

        if i >= 50: 
            break
    
    summary = experiment_summary(formatted_explanations, feature_names)
    pct_occur = [0]
    # it can be in 1,2,3 if one is intersted in finding in the top 3 features
    for indc in [1]:
        found = False
        for tup in summary[indc]:
            if tup[0] == biased_feature_name:
                pct_occur.append(sum([pct_occur[-1], tup[1]]))
                found = True

        if not found:
            pct_occur.append(pct_occur[-1])

    pct_occur = pct_occur[1:]
    pct_first = pct_occur[0]
    # uncomment them if you change the line: for indc in [1,2,3]
#     pct_second = pct_occur[1]
#     pct_third = pct_occur[2]

    # classifier name
    for k,v in clfs.items():
        ch = str(estimator['classifier']).find(k)
        if ch==0:
            clf_name = k
#     clf_name = estimator.__class__.__name__
#     print(clf_name,f1score,pct_first)
    # appending into the list
    fname = Path(file_runing_dir)/Path(f'log/{clf_name}.txt')
    print(f"{f1score},{pct_first}",file = open(fname,'a'))
    
    return f1score 

# parameters for classifiers
def parameters_classifiers():
    clf1 = LogisticRegression(random_state=params.seed)
    clf2 = SVC(random_state=params.seed,probability=True)
    clf3 = KNeighborsClassifier()
    clf4 = GaussianNB()
    clf5 = MultinomialNB()
    clf6 = DecisionTreeClassifier(random_state=params.seed)
    clf7 = RandomForestClassifier(random_state=params.seed)
    clf8 = GradientBoostingClassifier(random_state=params.seed)
    clf9 = MLPClassifier(random_state=params.seed)

    # parameters for logistic regression
    params1 = {}
    params1['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    params1['classifier__penalty'] = [ 'l2']
    params1['classifier'] = [clf1]

    # parameters for SVC
    params2 = {}
    params2['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    params2['classifier__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    params2['classifier__degree'] = [3,4,5]
    params2['classifier'] = [clf2]

    # parameters for KNeighborsClassifier
    params3 = {}
    params3['classifier__n_neighbors'] = [3,4,5,6,7,8,9,25,30]
    params3['classifier__p'] = [1,2]
    params3['classifier'] = [clf3]

    # parameters for GaussianNB
    params4 = {}
    params4['classifier'] = [clf4]



    # parameters for DecisionTreeClassifier
    params6 = {}
    params6['classifier__criterion'] = ['gini','entropy']
    params6['classifier__max_depth'] = [5,10,15,25,30,None]
    params6['classifier__min_samples_split'] = [2,5,7,10,15]
    params6['classifier__max_features'] = [None,'auto','sqrt','log2']
    params6['classifier'] = [clf6]

    # parameters for RandomForestClassifier
    params7 = {}
    params7['classifier__n_estimators'] = [10, 20, 50, 100, 250, 300]
    params7['classifier__criterion'] = ['gini','entropy']
    params7['classifier__max_depth'] = [5,10,15,25,30,40,50,None]
    params7['classifier__min_samples_split'] = [2,5,7,10,15]
    params7['classifier__max_features'] = [None,'auto','sqrt','log2']
    params7['classifier'] = [clf7]

    # parameters for GradientBoostingClassifier
    params8 = {}
    params8['classifier__loss'] = ['deviance','exponential']
    params8['classifier__learning_rate'] = [10**-3,10**-2, 10**-1]
    params8['classifier__n_estimators'] = [10, 20, 30, 40, 50, 100]
    params8['classifier__max_depth'] = [5,10,15,25,30,40,50,None]
    params8['classifier__max_features'] = [None,'auto','sqrt','log2']
    params8['classifier'] = [clf8]

    # parameters for MLPClassifier
    params9 = {}
    params9['classifier__hidden_layer_sizes'] = [(100,),(5,5,),(10,10,),(5,5,5,),(10,10,10,)]
    params9['classifier__max_iter'] = [100, 150, 200, 250, 300]
    params9['classifier__activation'] = ['relu', 'tanh', 'logistic']
    params9['classifier__learning_rate'] = ['constant', 'adaptive']
    params9['classifier'] = [clf9]
    all_params = [params1, params2, params3, params4, params6, params7, params8, params9]
    
    return (clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9),(params1, params2, params3, params4, params6, params7, params8, params9), all_params

# for visualization
class visualize_results:
    def __init__(self,lime_shap=True):
        self._result = [[] for _ in range(9)]
        self._clfs = {"LogisticRegression":0,
                        "SVC":1,
                        "KNeighborsClassifier":2,
                        "GaussianNB":3,
                        "MultinomialNB":4,
                        "DecisionTreeClassifier":5,
                        "RandomForestClassifier":6,
                        "GradientBoostingClassifier":7,
                        "MLPClassifier":8}
        if lime_shap:
            self._ana_type = 'LIME'
        else:
            self._ana_type = 'SHAP'
    def extract_values(self,clf_name):
        fname = Path(file_runing_dir)/f'log/{clf_name}.txt'
        if os.path.exists(fname):
            file = open(fname,'r')
            self._file_exists = True
        else:
            self._file_exists = False
            return self._x, self._y
        data = csv.reader(file, delimiter=',')
        table = [row for row in data]
        table = np.asarray(table)
        x_var = table[:,0]
        x_var = [float(x) for x in x_var]

        y_var = table[:,1]
        y_var = [float(x) for x in y_var]
        
        self._x = np.asarray(x_var)
        self._y = np.asarray(y_var)
        return self._x, self._y
    def sc_plot(self,clf_name):
        fig,ax = plt.subplots()
#         plt.ylim(-.05,1.05)
#         plt.xlim(0,1)

        sns.scatterplot(x=self._x, y=self._y, ax=ax)
        ax.set_xlabel("F1 score on OOD task")
        ax.set_ylabel("% explanations with race as first feature")
        ax.set_title(f"{self._ana_type} Analysis on {clf_name}")
        plt.savefig(Path(file_runing_dir)/Path(f"analysis_results/pics/{self._ana_type}_{clf_name}_f1_first.png"))
    def plot_results(self):
        count = 0
        for clf_name,_ in self._clfs.items():
            self.extract_values(clf_name)
            print(clf_name,self._x.shape,self._y.shape)
            if self._file_exists:
                self.sc_plot(clf_name)
                count+=1
        print(f"{count} plots have been saved to analysis/analysis_results/pics/")
        plt.show()
            
def save_grdidSearchResults(gs,lime_shap=True):
    # defining initial path
    init_path = Path(file_runing_dir)/Path('pickles/')
    if lime_shap:
        var_name = 'LIME_' + 'grid_search_object.pkl'
    else:
        var_name = 'SHAP_' + 'grid_search_object.pkl'
    path_pickle = init_path / Path(var_name)
    # saving gs object
    joblib.dump(gs, str(path_pickle))
    # load gs object
    grid_results = joblib.load(str(path_pickle))
    # converting the result into the dataframe
    results_df = pd.DataFrame(grid_results.cv_results_)
    # slicing related columns
    main_df = results_df[['params','mean_test_f1_score','rank_test_f1_score']]
    # sorting
    main_df = main_df.sort_values(by=["rank_test_f1_score"])
    # defining all classifiers
    classifiers = ['LogisticRegression','SVC', 'KNeighborsClassifier', 'GaussianNB', 'MultinomialNB', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'MLPClassifier'] #

    # adding the name of classifier into a different column
    for index, row in main_df.iterrows():
        for clf in classifiers:
            if str(row['params']).find(clf) != -1:
                main_df.at[index,'classifier'] = clf

    main_df.set_index('classifier', inplace=True)
    
    # saving paths
    if lime_shap:
        filename = Path(file_runing_dir)/Path("analysis_results")/Path("lime_grid_search_results.csv")
    else:
        filename = Path(file_runing_dir)/Path("analysis_results/"+"shap_grid_search_results.csv")
    main_df.to_csv(str(filename))

def main_exp(lime_shap=True):
    global xtrain,ytrain,xtest,ytest,feature_names,categorical_features,biased_feature_idx,unrelated_feature_idx_1,unrelated_feature_idx_2, biased_feature_name, dataset_name
    xtrain,ytrain,xtest,ytest,feature_names,categorical_features,biased_feature_idx,unrelated_feature_idx_1,unrelated_feature_idx_2, biased_feature_name, dataset_name = run_compas()
    global adv_explainer, background_distribution
    if lime_shap:
        # lime case
        adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, 
                                                               feature_names=feature_names, 
                                                               discretize_continuous=False)
    else:
        # shap case
        background_distribution = shap.kmeans(xtrain,10)
    
    # parameters for classifiers
    classifiers,parameters,all_params = parameters_classifiers()
    clf1,*rest = classifiers
    params1,*rest = parameters
    # define score function
    if lime_shap:
        score = {'f1_score': lime_scorer_f1}
    else:
        score = {'f1_score': shap_scorer_f1}
    # pipline object
    pipeline = Pipeline([('classifier', clf1)])
    gs = GridSearchCV(pipeline, all_params, cv=5, n_jobs=-1, scoring=score, refit='f1_score', verbose=2).fit(xtrain, ytrain)
    print('*'*78)
    print('*'*78)
    print('*'*78)
    df = pd.DataFrame(data={"Best Estimator":gs.best_estimator_, "Parameters":gs.best_params_,"Score":gs.best_score_,"Refit time":gs.refit_time_})
    # saving paths
    if lime_shap:
        path_csv = Path(file_runing_dir)/Path("analysis_results")/Path("lime_grid_search_results_best.csv")
    else:
        path_csv = Path(file_runing_dir)/Path("analysis_results/"+"shap_grid_search_results_best.csv")
    
    df.to_csv(str(path_csv))
    # best classigiers
    print("Best estimator found: ",gs.best_estimator_)
    # best parameters
    print("Best parameters: ", gs.best_params_)
    # best score attained
    print("Best score: ", gs.best_score_)
    print("Seconds used for refiting: ",gs.refit_time_)
    print('*'*78)
    print('*'*78)
    print('*'*78)
    # save results
    save_grdidSearchResults(gs,lime_shap)
    # for the visulization
    v_r = visualize_results(lime_shap)
    v_r.plot_results()


if __name__ == "__main__":
    l_s = parameter_parser()
    if l_s ==0:
        lime_shap = False
    else:
        lime_shap = True


    # defining classifiers
    clfs = {"LogisticRegression":0,
        "SVC":1,
        "KNeighborsClassifier":2,
        "GaussianNB":3,
        "MultinomialNB":4,
        "DecisionTreeClassifier":5,
        "RandomForestClassifier":6,
        "GradientBoostingClassifier":7,
        "MLPClassifier":8}
    
    path_err_adv_explainer = Path(file_runing_dir) / Path("log/log_err_adv_explainer.txt") 
    if os.path.exists(str(path_err_adv_explainer)):
        os.remove(str(path_err_adv_explainer))
    for k,_ in clfs.items():
        path_log = Path(file_runing_dir) / Path(f"log/{k}.txt")
        if os.path.exists(str(path_log)):
            os.remove(str(path_log))
    # main file
    main_exp(lime_shap)
    print('Run has completed...')
