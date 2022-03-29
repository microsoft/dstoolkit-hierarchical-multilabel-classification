# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
from base_class import Approach
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score

class ChainedModels(Approach):
    def __init__(self):
        pass

    def fit(self, df, X, X_test, y_test_labels, model, target_columns, additional_colms):
        """Fits a chain of models to the input features """
        if model == 'mnb':
            _, num_columns, _ = self.get_dtype(X)
            test_df = X[num_columns]
            if (test_df.values < 0).any():
                logging.raiseExceptions(
                        "Negative values found in input data \
                            for Multinomial Naive Bayes")
        # Approach1 - Chained models
        tgt_cols_list, res_chained_mdls = [], {}
        for idx in range(1, len(target_columns)+1):
            mdl_name = 'm'+str(idx)
            res_chain_mdl = self.multi_input_model(
                                        dt=df,
                                        X=X,
                                        tgt_colm=target_columns[idx-1],
                                        model=model,
                                        additional_colms=additional_colms)
            res_chained_mdls[mdl_name] = copy.deepcopy(res_chain_mdl)
            additional_colms.append(target_columns[idx-1])
            le_tc_tgt = res_chained_mdls[mdl_name][2]
            mdl = res_chained_mdls[mdl_name][0]

            if res_chained_mdls[mdl_name][3]:
                for tgt in res_chained_mdls[mdl_name][3]:
                    for i in range(len(tgt_cols_list)):
                        le_tc = \
                            res_chained_mdls[mdl_name][3][target_columns[i]]
                        X_test = X_test.reset_index(drop=True)
                        X_test.loc[:, target_columns[i]] = \
                            le_tc.transform(tgt_cols_list[i])
                predicted_label = pd.Series(le_tc_tgt.inverse_transform(
                    mdl.predict(X_test)))
                tgt_cols_list.append(predicted_label)
            else:
                predicted_label = pd.Series(le_tc_tgt.inverse_transform(
                    mdl.predict(X_test)))
                tgt_cols_list.append(predicted_label)
        comb_df = pd.concat(tgt_cols_list, axis=1)
        comb_df['combined_targets'] = comb_df.agg('--'.join, axis=1)
        y_pred_labels = comb_df['combined_targets']
        accuracy_chained_mdls = accuracy_score(y_test_labels, y_pred_labels)
        hl1 = self.h_loss(y_test_labels, y_pred_labels)

        print(f'\nAccuracy with Chained Model Approach using {model} ='
                f'{accuracy_chained_mdls:7,.4f}')
        print(f'Hamming loss in Chained Model Approach using {model} ='
                f'{hl1: 7,.4f}')
        return accuracy_chained_mdls, res_chained_mdls

    def predict(self, best_approach_dict, df):
        tgt_cols_list = []
        for idx in range(1, len(best_approach_dict['res'])+1):
            mdl_name = 'm'+str(idx)
            mdl = best_approach_dict['res'][mdl_name][0]
            le_tc_tgt = best_approach_dict['res'][mdl_name][2]
            if best_approach_dict['res'][mdl_name][3]:
                col_idx = 0
                le_enc = best_approach_dict['res'][mdl_name][3]
                for tgt, enc in le_enc.items():
                    df[tgt] = enc.transform(tgt_cols_list[col_idx])
                    col_idx += 1
                predicted_label = pd.Series(
                    le_tc_tgt.inverse_transform(mdl.predict(df)))
                tgt_cols_list.append(predicted_label)
            else:
                predicted_label = pd.Series(
                    le_tc_tgt.inverse_transform(
                        mdl.predict(df)))
                tgt_cols_list.append(predicted_label)
        df_pred = pd.concat(tgt_cols_list, axis=1)
        return df_pred