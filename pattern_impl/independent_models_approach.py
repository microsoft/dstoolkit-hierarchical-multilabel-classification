# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
from base_class import Approach
import pandas as pd
import logging
from sklearn.metrics import accuracy_score
class IndependentModels(Approach):
    def __init__(self):
        pass

    def fit(self, df, X, X_test, y_test_labels, model, target_columns, additional_colms):
        """Fits independent models for each label

        Args:
            df (pd.DataFrame, optional): [description]. Defaults to None.
            X (matrix, optional): [description]. Defaults to None.
            target_columns (list, optional): [description]. Defaults to [].
            y_test_labels (np.array, optional): \
                [description]. Defaults to None.
            model (str, optional): [description]. Defaults to 'mnb'.
            additional_colms (list, optional): [description]. Defaults to [].

        Returns:
            [tuple]: Final accuracy of the independent model approach,
            a dict of model artifacts.
        """
        if model == 'mnb':
            _, num_columns, _ = self.get_dtype(X)
            test_df = X[num_columns]
            if (test_df.values < 0).any():
                logging.raiseExceptions(
                        "Negative values found in input data \
                            for Multinomial Naive Bayes")
        tgt_cols_list, res_independent_mdls = [], {}
        for idx in range(1, len(target_columns)+1):
            mdl_name = 'm'+str(idx)
            res_ind_mdl = self.multi_input_model(
                                            dt=df,
                                            X=X,
                                            tgt_colm=target_columns[idx-1],
                                            model=model,
                                            additional_colms=additional_colms)
            res_independent_mdls[mdl_name] = copy.deepcopy(res_ind_mdl)
            predicted_label = pd.Series(
                res_independent_mdls[mdl_name][2].inverse_transform(
                    res_independent_mdls[mdl_name][0].predict(X_test)))
            tgt_cols_list.append(predicted_label)
        comb_df = pd.concat(tgt_cols_list, axis=1)
        comb_df['combined_targets'] = comb_df.agg('--'.join, axis=1)
        y_pred_labels = comb_df['combined_targets']
        accuracy_independent_mdls = accuracy_score(y_test_labels, y_pred_labels)
        hl2 = self.h_loss(y_test_labels, y_pred_labels)
        print(f'\nAccuracy with Independent Model Approach using {model} ='
                f'{accuracy_independent_mdls:7,.4f}')
        print(f'Hamming loss in Independent Model Approach '
                f'using {model} = {hl2: 7,.4f}')
        return accuracy_independent_mdls, res_independent_mdls

    def predict(self, best_approach_dict, df):
        tgt_cols_list = []
        for idx in range(1, len(best_approach_dict['res'])+1):
            mdl_name = 'm'+str(idx)
            mdl = best_approach_dict['res'][mdl_name][0]
            le_tgt = best_approach_dict['res'][mdl_name][2]
            label = pd.Series(le_tgt.inverse_transform(
                mdl.predict(df)))
            tgt_cols_list.append(label)
        df_pred = pd.concat(tgt_cols_list, axis=1)
        return df_pred