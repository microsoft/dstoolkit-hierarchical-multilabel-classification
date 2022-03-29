# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
from base_class import Approach
import pandas as pd
import logging
from sklearn.metrics import accuracy_score
class PowersetLabels(Approach):
    def __init__(self):
        pass

    def fit(self, df, X, X_test, y_test_labels, model, target_columns, additional_colms):
        """Fits a model for concatenation of labels

        Args:
            df (pd.DataFrame, optional): [description]. Defaults to None.
            X (matrix, optional): [description]. Defaults to None.
            y_test_labels (np.array, optional): \
                [description]. Defaults to None.
            ip_vector (matrix, optional): [description]. Defaults to None.
            ip_vectorizer (object, optional): [description]. Defaults to None.
            model (str, optional): [description]. Defaults to 'mnb'.
            additional_colms (list, optional): [description]. Defaults to [].

        Returns:
            [tuple]: Final accuracy of the independent model approach, \
                a dict of model artifacts.
        """
        if model == 'mnb':
            _, num_columns, _ = self.get_dtype(X)
            test_df = X[num_columns]
            if (test_df.values < 0).any():
                logging.raiseExceptions(
                        "Negative values found in input data \
                            for Multinomial Naive Bayes")
        res_powerset_mdl = {}
        mdl_name = 'm1'
        res_power_mdl = self.multi_input_model(
                                            dt=df,
                                            X=X,
                                            tgt_colm='combined_targets',
                                            model=model,
                                            additional_colms=additional_colms)
        res_powerset_mdl[mdl_name] = copy.deepcopy(res_power_mdl)
        y_pred_labels = res_powerset_mdl[mdl_name][2].inverse_transform(
            res_powerset_mdl[mdl_name][0].predict(X_test))
        accuracy_powerset = accuracy_score(y_test_labels, y_pred_labels)
        hamming_loss_powerset = self.h_loss(y_test_labels, y_pred_labels)
        print(f'\nAccuracy with Powerset Model using {model} = '
              f'{accuracy_powerset:7,.4f}')
        print(f'Hamming loss in Powerset Model using {model} = '
              f'{hamming_loss_powerset: 7,.4f}')
        return accuracy_powerset, res_powerset_mdl

    def predict(self, best_approach_dict, df):
        mdl = best_approach_dict['res']['m1'][0]
        le_tgt = best_approach_dict['res']['m1'][2]
        predicted_label = le_tgt.inverse_transform(
            mdl.predict(df))
        split_pred = []
        df_pred = pd.DataFrame()
        for label in predicted_label:
            split_pred.append(label.split('--'))
            df_pred = pd.DataFrame(split_pred)
        return df_pred