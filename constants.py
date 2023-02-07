import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

FF_DATA_PATH = "../data"
FF_DATA_PHENO = "../data/Phenotypic_V1_0b_preprocessed1.csv"

AVAILABLE_CLASSIFIERS_MAP = {
    'nn': ['nn', 'ANN', 'DNN'],
    'svm': ['svc', 'svm'],
    'lsvm': ['lsvm', 'linear svm', 'linsvm'],
    'xgb': ['xgb', 'extragradientboost'],
    'lr': ['logistic', 'logistic regression', 'lg', 'lr'],
    'gnb': ['naive_bayes', 'naive bayes', 'gaussian naive bayes', 'gnb'],
    'pagg': ['pagg', 'passive_aggressive', 'passive aggressive', 'passagg', 'pasag'],
    'ridge': ['ridge', 'rdg', 'rd'],
    'sgd': ['sgd', 'stochastic gradient descend'],
    'knn': ['knn', 'neighbors', 'k-nn'],
    'rf': ['rf', 'random forest'],
    'gradboost': ['gradboost','gradient trees', 'gradient boost trees', 'gbt'],
    'lgbm':['lgbm','gbm', 'lightgbm']
}

DATA_DIR = {
    'raw':'../data/raw',
    'raw_1file' : '../data/raw_allInOneJson',
    'feat_extract':'../data/feature_extraction',
    'data_divisor':'../data/data_divisor',
    'percentile':'../data/feature_extraction/percentile/raw.csv',
    'pheno': '../data/updated_pheno.csv',
    'phase2': '../notebooks/raw_data_perc_for_production_testing.csv'
}


MODELS_DIR = {
    'main': '../models',
    'config':'../models/exper_conf',
    'production': '../selected_models_for_production'
}

OUTPUT_DIR = "../output"
# OUTPUT_DIR_CLUSTER = "../output/bigdatacluster/randomizedOutput"
OUTPUT_DIR_CLUSTER = "../randomizedOutput"
BEST_RESULTS_JSON = "../notebooks/best_results_dict.json"


MAX_ITR = int(1e9)

# currying Look it up
DD_MIN_N_PER_CLASS = 100

CLC_DICT = {
    'lsvm': lambda: LinearSVC(max_iter=MAX_ITR),
    'pagg': lambda: PassiveAggressiveClassifier(max_iter=MAX_ITR),
    'lr': lambda: LogisticRegression(max_iter=MAX_ITR),
    'sgd': lambda: SGDClassifier(max_iter=MAX_ITR),
    'ridge': lambda: RidgeClassifier(max_iter=MAX_ITR),
    'knn': KNeighborsClassifier,
    'xgb': XGBClassifier,
    'gnb': GaussianNB,
    'rf': RandomForestClassifier,
    'svm': lambda: SVC(max_iter=MAX_ITR, probability=True),
    'nn': lambda: MLPClassifier(max_iter=MAX_ITR),
    'gradboost': GradientBoostingClassifier,
    'lgbm': lambda: LGBMClassifier(boosting_type='gbdt', objective='binary', n_jobs=-1)
}
PARAM_GRID={
    'lsvm': {
        'penalty':['l1','l2'],
        'loss':['hinge','squared_hinge'],
        'C':[0.1,1,5, 10]
    },
    'pagg': {
        'C':[0.1,1,5,10], 'n_iter_no_change':[1,5,10]
    },
    'sgd':{
        'loss':['modified_huber','squared_hinge','perceptron'],
        'penalty':['l1','l2','elasticnet'],
        'alpha': np.arange(0.1, 5, 0.1),
        'l1_ratio': np.arange(0,1,0.05),
    },
    'lr':{
        'penalty': ['l1','l2','elasticnet','none'],
        'C':[0.1,1,5, 10],
        # 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    'xgb': {
        'booster':['gbtree','gblinear','dart'],
        'learning_rate':[0.001, 0.01, 0.5, 1],
        'min_child_weight': [0.01, 0.5, 1, 10],
        'gamma': [0, 0.1, 1, 5, 50, 100],
        'reg_alpha':[0, 0.001, 0.5, 1, 10],
        'reg_lambda':[0, 0.001, 0.5, 1,  10],
        'colsample_bytree': [0.6, 0.8, 1.0],
    },
    'ridge':{
            'alpha':np.arange(0.1, 5, 0.1),
            'normalize':[True, False],
        },
    'knn': {
        'n_neighbors':np.arange(1,70, 2),
        'weights':['uniform','distance'],
        'leaf_size':np.arange(10,150, 10),
        'p':np.arange(1,7),
        'metric':['identifier','euclidean','manhattan','chebyshev','minkowski',
                  'wminkowski','seuclidean','mahalanobis']
        },
    'gnb': {

    },
    'rf': {
        'n_estimators':[50, 100, 200, 500, 1000],
        'criterion':['gini','entropy'],
        # 'max_features':['auto','sqrt'],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[0.0001,0.1,0.2,0.3,0.4,0.5],
        # 'bootstrap':[True,False]

    },
    'svm': {
        'C':[0.1,1,5, 10],
        'kernel':['poly','rbf','sigmoid'],
        'degree':[2,3,4,5,6],
        # 'gamma':['scale','auto'],
        'coef0':[0.0,0.01,0.5,5,50,100]
    },
    'nn':{
        'hidden_layer_sizes': [(200,), (150,), (100), (150,100,50,), (100,50,25,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [ 0.05, 0.1, 0.5],
        'learning_rate': ['constant','adaptive'],
        'beta_1':[ 0.01, 0.1, 0.3, 0.5, 0.9],
        'beta_2':[ 0.01, 0.1, 0.3, 0.5, 0.9],
    },
    'gradboost':{
        'loss':['deviance','exponential'],
        'learning_rate':np.arange(0.1, 5, 0.1),
        'n_estimators':np.arange(100,500,50),
        'subsample':np.arange(0.5,1, 0.1),
        'criterion':['mse','friedman_mse'],
        'min_samples_split':np.arange(2,10, 1),
        'min_samples_leaf':np.arange(1,10,1),
        'max_depth':np.arange(3,50,1),
        'warm_start':[True],
    },
    'lgbm':{
        'reg_alpha':np.arange(0,2,0.1),
        'reg_lambda':np.arange(0,2,0.1),
        'n_estimators':np.arange(100,2000,100),
        'learning_rate':np.arange(0.01,1,0.05),
        'max_depth':np.arange(1, 10, 2),
        'num_leaves':np.arange(3, 32, 5)
    }
}

# OUTPUT = {
#     'lsvm': '../data/models/lsvm',
#     'pagg': '../data/models/pagg',
#     'lr': '../data/models/lr',
#     'sgd': '../data/models/sgd',
#     'ridge':'../data/models/ridge',
#     'knn': '../data/models/knn',
#     'xgb': '../data/models/xgb',
#     'gnb': '../data/models/gnb',
#     'rf': '../data/models/rf',
#     'svm': '../data/models/svm',
#     'nn': '../data/models/nn',
#     'gradboost': '../data/models/gradboost'
# }

SRS_SCORES_MAP = {
    'TD': (0, 59),
    'mild':(60, 65),
    'moderate':(66, 75),
    'sever':(76, 1000)
}


TARGET = 'DX_GROUP'
ASD = 1
TD = 2