# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
import pandas as pd
import logging
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class Approach(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, df, X, X_test, y_test_labels, model, target_columns, additional_colms):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    def normalize_length(
        self,
        input_list,
        n,
        fill_na='none'
    ):
        res = []
        for element in input_list:
            if len(element) == n:
                res.append(list(element))
            elif len(element) > n:
                res.append(list(element)[:n])
            else:
                temp = []
                for i in range(n):
                    if i < len(element):
                        temp.append(element[i])
                    else:
                        temp.append(fill_na)
                res.append(temp)
        return res

    def h_loss(
        self,
        actuals,
        predictions,
        sep='--'
    ):

        y_test_labels = [row.split(sep) for row in actuals]
        y_test_flat = [element for row in y_test_labels for element in row]
        y_pred_labels = [row.split(sep) for row in predictions]
        total_labels = len(y_pred_labels[0])
        y_pred_flat = [element for row in self.normalize_length(
            y_pred_labels, n=total_labels) for element in row]
        _hamming_loss = 1 - accuracy_score(y_test_flat, y_pred_flat)
        return _hamming_loss

    def get_dtype(self, df: pd.DataFrame):
        cat_columns = list(df.select_dtypes(include='object').columns)
        num_columns = list(df.select_dtypes(include='number').columns)
        text_columns = []
        for col in cat_columns:
            if df[col].str.len().max() > 99:
                cat_columns.remove(col)
                text_columns.append(col)
        return cat_columns, num_columns, text_columns

    def multi_input_model(
        self,
        dt,
        X: pd.DataFrame,
        tgt_colm,
        model='gnb',
        additional_colms=[]
    ):
        if model == 'mnb':
            _, num_columns, _ = self.get_dtype(X)
            test_df = X[num_columns]
            if (test_df.values < 0).any():
                logging.raiseExceptions(
                        "Negative values found in input \
                            data for Multinomial Naive Bayes")
        le = LabelEncoder()
        tgt = le.fit_transform(dt[tgt_colm].tolist())
        le_dict = {}
        df_X = X.copy()
        if len(additional_colms):
            le_dict, tgt_dict = {}, {}
            for c in additional_colms:
                temp = LabelEncoder()
                tgt_dict[c] = temp.fit_transform(dt[c].tolist())
                le_dict[c] = temp
                df_X[c] = tgt_dict[c]
        X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                            tgt,
                                                            train_size=0.8,
                                                            random_state=7)
        logging.info(f'Shape of training data set = {X_train.shape};'
                     f'test_data_set = {X_test.shape}')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        logging.info(f'Accuracy of model = {acc:7,.4f}')
        return [model, acc, le, le_dict]