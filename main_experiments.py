import os.path
import warnings

import DataReaderFunc
import FeatureSelection
import Classifers
import constants
import dill
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from experiment_designer import exp_params
import utils
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.style as style
style.use('ggplot')
import numpy as np
import seaborn as sns
import matplotlib as mpl
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
mpl.rc('font', **font)
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, precision_score,
                            recall_score, roc_auc_score, plot_roc_curve)
from sklearn.base import clone


class Experiment:
    def __init__(self, **experiment_params):
        self._expr_params = experiment_params
        if 'output' in experiment_params:
            self.stampfldr_ = experiment_params['output']
            del experiment_params['output']
        else:
            self.stampfldr_ = None

        # Type of data to use
        self.data_repr = None
        # self.normalizer = StandardScaler()
        self.normalizer = False

        # Data divisor parameters
        self.DD_data= None
        self.DD_pheno= None
        self._DD_obj = None

        # Featureselection parameters
        self.FS_est = None
        self.FS_cv = None
        self.FS_scoring = None
        self.FS_n_jobs = None
        self.FS_verbose = None
        self.FS_step = None
        self.FS_min_features_to_select = None
        self._FS_obj = FeatureSelection.FeatureSelector()
        self.FS_selected_feats_ = None
        self.FS_grid_scores_ = None

        # Classifier Params
        self.ML_est = None
        self.ML_cv = None
        self.ML_scoring = None
        self.ML_n_jobs = None
        self.ML_verbose = None
        self.ML_hyper_search_type = None
        self.ML_agg_models = None # Only used when ML_set is a list of classifier keys
        self.ML_n_iter = None # Only used when hyper parameter select is randomized
        self._ML_obj = Classifers.CustomClassifier()
        self.ML_grid_ = None

        if len(experiment_params)> 1:
            self._parse_exp_params(experiment_params)
            self._expr_params = experiment_params
        else:
            raise FileNotFoundError("Please add an experiment json file")
            # self._parse_exp_params(exp_params)
            # self._expr_params = exp_params

    def _parse_exp_params(self, exp_params_dict):
        for key, item in exp_params_dict.items():
            if key in ['DD', 'FS', 'ML','FE']:
                data_dict = item
                self._validate_params(data_dict, key)
                for skey, sval in data_dict.items():
                    setattr(self, f'{key}_{skey}', sval)
            elif key == 'data_repr':
                self.data_repr = item
            elif key == 'normalizer':
                if isinstance(item, str):
                    if 'min' in item.lower():
                        self.normalizer = MinMaxScaler()
                    elif 'st' in item.lower():
                        self.normalizer = StandardScaler()
                    else:
                        raise ValueError("Normalizer can be eith 'minmax' or 'standard")
                elif ('fit' in dir(item)) and ('transform' in dir(item)) and ('fit_transform' in dir(item)):
                    self.normalizer = item
                else:
                    raise ValueError("normalizer can be either string (minmax, or standard) or a transform that "
                                     "implements 'fit','transform' and 'fit_transform' methods")
            else:
                raise KeyError(f'Experiment parameters should be one of the following ["DD","FS","ML"]')

    def _validate_params(self, dd:dict, prefix:str=None):
        for key, val in dd.items():
            if f'{prefix}_{key}' not in self.__dict__:
                raise KeyError(f'{prefix}_{key} is not a valid variable')

    def _check_and_fill_expr_params(self):
        for key, item in self.__dict__.items():
            parts = key.split('_')
            obj_type = parts[0]
            if obj_type in self._expr_params:
                self._expr_params[obj_type] = {'_'.join(parts[1:]):item}
            else:
                self._expr_params[obj_type]['_'.join(parts[1:])] = item

    def _plot_distr(self, df):
        f = plt.figure(num=1, figsize=(16, 10))
        ax = plt.gca()
        df[df['DX_GROUP']==1]['AGE_AT_SCAN'].rename('TD').plot(kind='bar', ax=ax, legend=True, color='blue', alpha=0.5)
        df[df['DX_GROUP']==0]['AGE_AT_SCAN'].rename('ASD').plot(kind='bar', ax=ax, legend=True, color='red', alpha=0.5)
        xmin1 = df[df['DX_GROUP'] == 1]['AGE_AT_SCAN'].min()
        xmin2 = df[df['DX_GROUP'] == 0]['AGE_AT_SCAN'].min()
        xmax1 = df[df['DX_GROUP'] == 1]['AGE_AT_SCAN'].max()
        xmax2 = df[df['DX_GROUP'] == 0]['AGE_AT_SCAN'].max()
        loc = 0
        step = int (len(ax.get_xticks())/5)
        updated_xticks = []
        while loc<len(ax.get_xticks()):
            updated_xticks.append(ax.get_xticks()[loc])
            loc += step

        custom_ticks = np.linspace(np.min([xmin1,xmin2]) , np.max([xmax1, xmax2]), len(updated_xticks))

        plt.xticks(updated_xticks, custom_ticks, visible=True, rotation=30)

        # ax.set_xticks(custom_ticks)
        # ax.set_xticklabels(custom_ticks)
        plt.xlabel('Age')
        # plt.xticks([])
        plt.ylabel('PDF')
        plt.savefig(f'{os.path.join(self.stampfldr_, "age_dist_group.png")}', bbox_inches='tight')
        plt.close(f)

    def _plot_score_grid(self, rfe_obj):
        f = plt.figure(num=1, figsize=(12, 8))
        ax = plt.gca()

        if isinstance(rfe_obj, dict):
            for key, item in rfe_obj.items():
                print(f"Grid scores dimension: {len(item.grid_scores_)}")
                grid = item.grid_scores_.mean(axis=0)
                plt.cla()
                plt.plot(np.arange(1, len(grid)+1), grid)
                plt.xlabel('# of features')
                plt.ylabel('Scores')
                plt.title(f'{self._expr_params ["FS"]["scoring"] if isinstance(self._expr_params ["FS"]["scoring"], str) else "Score"} vs.'
                          f'number of features')
                plt.savefig(f'{os.path.join(self.stampfldr_, f"FS_{key}.png")}', bbox_inches='tight')
        else:
            grid = rfe_obj.grid_scores_
            plt.plot(np.arange(1, len(grid)), grid)
            plt.xlabel('# of features')
            plt.ylabel('Scores')
            plt.title(f'{self._expr_params ["FS"]["scoring"] if isinstance(self._expr_params ["FS"]["scoring"]) else "Score"} vs.'
                      f'number of features')
            plt.savefig(f'{os.path.join(self.stampfldr_, "FS.png")}', bbox_inches='tight')

        plt.close(f)

    def _plot_feature_importance(self, df, rfe_obj):
        f = plt.figure(num=1, figsize=(18, 10))
        ax = plt.gca()
        df_local = df.drop('DX_GROUP', axis=1) if 'DX_GROUP' in df.columns else df

        if isinstance(rfe_obj, dict):
            for key, item in rfe_obj.items():
                if 'coef_' in item.estimator_.__dict__:
                    imp = item.estimator_.coef_.reshape(-1,).tolist()
                else:
                    imp = item.estimator_.feature_importances_.reshape(-1,).tolist()
                selected_feats = df_local.columns[item.support_]
                selected_feats_list = []
                for x, y in zip(selected_feats, imp):
                    selected_feats_list.append((x, y))
                selected_feats_list_sorted = sorted(selected_feats_list, key=lambda xy: abs(xy[1]), reverse=True)
                # sorted_feats = ['_'.join(xy[0].split('_')[0:2]) for xy in selected_feats_list_sorted]
                sorted_feats = [xy[0] for xy in selected_feats_list_sorted]
                sorted_imp = [xy[1] for xy in selected_feats_list_sorted]
                plt.cla()
                if len(sorted_feats)<=20:
                    sns.barplot(y=sorted_feats, x=sorted_imp)
                    plt.title(f'{len(sorted_feats)} are selected using {key}')
                else:
                    sns.barplot(y=sorted_feats[:20], x=sorted_imp[:20])
                    plt.title(f'{len(sorted_feats)} are selected using {key}. Only 20 are shown')
                fig_name = f"FS_importance_{item.estimator_.__str__().split('(')[0]}.png"
                plt.savefig(f'{os.path.join(self.stampfldr_,key+"_"+fig_name)}', bbox_inches='tight')
        else:
            if 'coef_' in rfe_obj.estimator_.__dict__:
                imp = rfe_obj.estimator_.coef_
            else:
                imp = rfe_obj.estimator_.feature_importances_
            selected_feats = df_local.columns[rfe_obj.support_]
            selected_feats_list = []
            for x, y in zip(selected_feats, imp[0]):
                selected_feats_list.append((x,y))
            selected_feats_list_sorted = sorted(selected_feats_list, key=lambda xy: abs(xy[1]), reverse=True)
            sorted_feats = ['_'.join(xy[0].split('_')[0:2]) for xy in selected_feats_list_sorted]
            sorted_imp = [xy[1] for xy in selected_feats_list_sorted]
            if len(sorted_feats) <= 20:
                sns.barplot(y=sorted_feats, x=sorted_imp)
                plt.title(f"{len(sorted_feats)} are selected using {rfe_obj.estimator_.__str__().split('(')[0]}")
            else:
                sns.barplot(y=sorted_feats[:20], x=sorted_imp[:20])
                plt.title(f"{len(sorted_feats)} are selected using {rfe_obj.estimator_.__str__().split('(')[0]}. Only 20 are shown")
            fig_name = f"FS_importance_{rfe_obj.estimator_.__str__().split('(')[0]}.png"
            plt.savefig(f'{os.path.join(self.stampfldr_, fig_name)}', bbox_inches='tight')

        plt.close(f)

    def _save_ML_scores(self, Xselected, ml_obj):
        results_dict = defaultdict(dict)
        if isinstance(Xselected, np.ndarray):
            if self._expr_params .get('FS') is None:
                new_Xselected = {'None':Xselected}
            else:
                est = self._expr_params .get('FS').get('est')
                if isinstance(est, str):
                    new_Xselected = {est: Xselected}
                else:
                    raise ValueError("There is a problem with understanding the experiment designer 'FS' estimator")
        elif isinstance(Xselected, dict):
            new_Xselected = Xselected
        else:
            raise ValueError("Xselected being passed to save model is not a numpy array or a dictionary")

        for krf, rfe_model in new_Xselected.items():
            for kml, ml_model in ml_obj[krf].items():
                bind = ml_obj[krf][kml].best_index_
                split_keys = [k for k in ml_obj[krf][kml].cv_results_.keys() if 'split' in k and 'test' in k]
                results_num = []
                for sk in split_keys:
                    results_num.append(ml_obj[krf][kml].cv_results_[sk][bind])
                results_dict[krf][kml] = f"{np.mean(results_num):.2f} +/- {np.std(results_num, ddof=1):.2f}"
        df = pd.DataFrame(results_dict).T
        df.to_csv(os.path.join(self.stampfldr_, 'ML_best_est_scores_rfRows_mlCols.csv'))

    def _save_selected_feats_json(self, selected_feats):
        with open(os.path.join(self.stampfldr_, 'selected_feats.json'), 'w') as f:
            if isinstance(selected_feats, dict):
                dict2save = {key: item.tolist() for key, item in selected_feats.items()}
                json.dump(dict2save, f)
            else:
                dict2save = {self._expr_params ['FS']['est']: selected_feats.tolist() if isinstance(selected_feats, np.ndarray)
                else selected_feats}
                json.dump(dict2save, f)

    def _create_pseudo_scores(self, Xselected,y, ml_obj):
        best_estimators_dict = defaultdict(dict)
        for krf, rfe_model in ml_obj.items():
            for kml, ml_model in ml_obj[krf].items():
                best_estimators_dict[krf][kml] = ml_obj[krf][kml].best_estimator_

        results_dict = defaultdict(dict)

        f = plt.figure(num=1, figsize=(12, 8))
        ax = plt.gca()
        for krf, rfe_model in best_estimators_dict.items():
            Xstrain, Xstest, ytrain, ytest = train_test_split(Xselected[krf], y, test_size=0.2, random_state=231,
                                                              shuffle=True, stratify=y)
            for kml, ml_model in best_estimators_dict[krf].items():
                clc = clone(best_estimators_dict[krf][kml])
                clc = clc.fit(Xstrain, ytrain)
                yhat = clc.predict(Xstest)

                uax = plot_roc_curve(best_estimators_dict[krf][kml], Xstest, ytest, ax=ax)
                uax.figure_.suptitle(f'RFE: {krf}, ML: {kml}')

                C = confusion_matrix(ytest, yhat)
                results_dict[krf][kml] = {}
                tn = C[0, 0]
                tp = C[1, 1]
                fn = C[1, 0]
                fp = C[0, 1]
                specificity = tn / (tn + fp)  # specificity, selectivity or true negative rate
                NPV = tn / (tn + fn)
                results_dict[krf][kml]['sens'] = recall_score(ytest, yhat)
                results_dict[krf][kml]['spec'] = specificity
                results_dict[krf][kml]['acc'] = accuracy_score(ytest, yhat)
                results_dict[krf][kml]['f1'] = f1_score(ytest, yhat)
                results_dict[krf][kml]['PPV'] = precision_score(ytest, yhat)
                results_dict[krf][kml]['NPV'] = NPV

            plt.savefig(f'{os.path.join(self.stampfldr_, f"ROC_{krf}_{kml}.png")}', bbox_inches='tight')
            plt.cla()

        first_level = list(best_estimators_dict.keys())
        second_level = list(best_estimators_dict[first_level[0]].keys())
        index = pd.MultiIndex.from_product([first_level, second_level],
                                           names=['RFE', 'Metrics'])
        df = pd.DataFrame(None, index=index, columns=results_dict[first_level[0]][second_level[0]].keys())
        for krf in results_dict:
            for kml in results_dict[krf]:
                for metric in results_dict[krf][kml]:
                    df.loc[(krf, kml), metric] = results_dict[krf][kml][metric]

        df.to_csv(os.path.join(self.stampfldr_, "pseudo_metrics.csv"))

    def run(self):
        if self.stampfldr_ is None:
            stamp = utils.get_time_stamp()
            self.stampfldr_ = os.path.join(constants.MODELS_DIR['main'], stamp)
        if not os.path.isdir(self.stampfldr_):
            os.mkdir(self.stampfldr_)

        if self._expr_params is None:
            self._check_and_fill_expr_params( )

        utils.save_experiment_params(self.stampfldr_, self._expr_params )

        if 'DD' in self._expr_params:
            self._DD_obj = DataReaderFunc.DataReaderFunc(data=self._expr_params['DD']['data'],
                                                         phenofile=self._expr_params['DD']['pheno'])
        else:
            self._DD_obj = None

        if 'FS' in self._expr_params:
            self._FS_obj.set_params(**self._expr_params['FS'], normalizer=self.normalizer)
        else:
            self._FS_obj = None

        if 'ML' in self._expr_params:
            self._ML_obj.set_params(**self._expr_params['ML'])
        else:
            self.ML_obj = None

        if self._DD_obj is not None:
            group_df, train_df, test_df = self._DD_obj.run()
        else:
            raise FileNotFoundError("You must set a DD attribute in your experiment json file with keys ['data','pheno']")

        train_df.to_csv(os.path.join(self.stampfldr_, 'train.csv'))
        test_df.to_csv(os.path.join(self.stampfldr_, 'test.csv'))

        # print(group_df)
        # self._plot_distr(group_df)
        group_df = train_df
        age = group_df.pop('AGE_AT_SCAN')
        sex = group_df.pop('SEX')
        _ = test_df.pop('AGE_AT_SCAN')
        _ = test_df.pop('SEX')

        if self._FS_obj is not None:
            # Normalizer to be refactored
            # normalize = True if self.normalizer is not None else False
            normalize = False
            Xselected, y, normalizer = self._FS_obj.run(group_df, normalizer=self.normalizer, normalize=normalize)
            # self._plot_feature_importance(group_df, self._FS_obj.rfe_)
            self.FS_selected_feats_ = self._FS_obj.selected_feats_
            self.FS_grid_scores_ = self._FS_obj.scores_
            Xselected_test = {}

            for key, columns in self.FS_selected_feats_.items():
                Xselected_test[key] = test_df[columns]

            if normalizer is not None:
                utils.save_model(os.path.join(self.stampfldr_, 'normalizer.p'), normalizer)

            utils.save_model(os.path.join(self.stampfldr_, "RFECV_obj"), self._FS_obj.rfe_)
            utils.save_model(os.path.join(self.stampfldr_, "GU_obj"), self._FS_obj.gu)

            self._save_selected_feats_json(self.FS_selected_feats_)
            # self._plot_score_grid(self._FS_obj.rfe_)
        else:
            sc = self.normalizer
            if 'DX_GROUP' in group_df.columns:
                X = sc.fit_transform(group_df.drop('DX_GROUP', axis=1))
            else:
                X = sc.fit_transform(group_df)
            Xselected = dict()
            Xselected["None"] = X
            y = group_df['DX_GROUP'].values
        ytest = test_df['DX_GROUP'].values
       ########################################################################################################
        # self.stampfldr_ = "D:\\PhD\\codes\\behavioralProject\\models\\20210516_012804"
        # with open(os.path.join(self.stampfldr_, 'FS_obj.p'), 'rb') as f:
        #     fs_obj = dill.load(f)
        # group_df = pd.read_csv(os.path.join(self.stampfldr_, 'group_df_afterFixation.csv'), index_col=0)
        # group_df.drop(['SEX','AGE_AT_SCAN '], axis=1, inplace=True)
        # self._FS_obj.rfe_ = fs_obj
        # y = group_df.pop('DX_GROUP').values
        # X = StandardScaler().fit_transform(group_df)
        # Xselected = {name: rfe.transform(X) for name, rfe in fs_obj.items()}
#################################################################################################################
        # Xselected_test = {}
        # for key, Xs
        utils.save_model(os.path.join(self.stampfldr_, 'Xselected.p'), (Xselected, y))
        if self._FS_obj is None:
            self.ML_grid_ = self._ML_obj.run(Xselected, y, est="None")
        else:
        # with open('..\\models\\20210524_211225\\FS_obj.p', 'rb') as f:
        #     fs_obj = dill.load(f)
        # groupdf = pd.read_csv('../models/20210524_211225/group_df_afterFixation.csv', index_col='subj_id')
        # groupdf.drop(['DX_GROUP','AGE_AT_SCAN ','SEX','SRS_COMMUNICATION_T','categories_COMMUNICATION'], axis=1,
        #              inplace=True)
        # groupdf.rename({'my_labels':'DX_GROUP'}, axis=1, inplace=True)
        # y = groupdf['DX_GROUP']
        # Xs = MinMaxScaler().fit_transform(groupdf.drop('DX_GROUP', axis=1))
        # self.stampfldr_ = "..\\models\\20210524_211225"
        # self.stampfldr_ = self.stampfldr_
        # Xselected = {}
        # for key, fs in fs_obj.items():
        #     Xselected[key] = Xs[:, np.where(fs.support_)[0]]
            self.ML_grid_ = self._ML_obj.run(Xselected, y, est=list(Xselected.keys()) if isinstance(Xselected, dict)
                                                                                    else self._expr_params ['FS']['est'])
        utils.save_model(os.path.join(self.stampfldr_, "ML_obj"), self._ML_obj.grid)

        self._save_ML_scores(Xselected, self.ML_grid_)
#         self.stampfldr_ = '..\\models\\20210527_220608'
#         df = pd.read_csv(os.path.join(self.stampfldr_, 'group_df_afterFixation.csv'), index_col=0)
#         with open(os.path.join(self.stampfldr_, 'FS_obj.p'), 'rb') as f:
#             fsobj = dill.load(f)
#         with open(os.path.join(self.stampfldr_, 'ML_obj.p'), 'rb') as f:
#             mlobj = dill.load(f)
#         df.drop(['DX_GROUP','AGE_AT_SCAN ','SEX','SRS_COMMUNICATION_T','categories_COMMUNICATION'], axis=1, inplace=True)
#         Xs = StandardScaler().fit_transform(df.drop('my_labels', axis=1))
#         y = df['my_labels']
#         Xselected = {}
#         for fkey, rfe in fsobj.items():
#             Xselected[fkey] = Xs[:, np.where(rfe.support_)[0]]
#         self.ML_grid_ = mlobj
#         self._create_pseudo_scores(Xselected, y, ml_obj=self.ML_grid_)
        self._create_pseudo_scores(Xselected_test, ytest, ml_obj=self.ML_grid_)

    def set_params(self, **params):
        item = params.get(next(iter(params)))
        if isinstance(item, dict):
            self._parse_exp_params(params)

        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a class member')

    def save_results(self, filename):
        # Needs to be modified
        if '.' in filename[-5:]:
            fname = filename.split('.')[1]
        else:
            fname = filename

        with open(os.path.join(self.stampfldr_, fname+'.p'), 'wb') as f:
            dill.dump(self, f)

import json
if __name__ == "__main__":
    with open("./fnc_exp.json", 'r') as f:
        exp_params = json.load(f)
    print(exp_params)
    experiment_1 = Experiment(**exp_params)
    experiment_1.run()
    experiment_1.save_results("singleFSML")

    # experiment_2 = Experiment(arguments2)
    # experiment_2.run()
    #
    # experiment_1.ML_est = "linear SVM"
    # params = {'FS_scoring':'balanced_accuracy',
    #                          'FS_n_jobs':-1,
    #                          'FS_verbose':3}
    # experiment_1.set_params(**params)
    #
    # print(experiment_1.__dict__)