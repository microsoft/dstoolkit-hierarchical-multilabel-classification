# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List
import pandas as pd
import numpy as np
import time

from datetime import timedelta
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder, \
                                StandardScaler, \
                                PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,\
                             GradientBoostingClassifier,\
                             ExtraTreesClassifier,\
                             AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from inspect import isclass
from chained_models_approach import ChainedModels
from independent_models_approach import IndependentModels
from powerset_labels_approach import PowersetLabels
import logging
import copy


class HMLC():
    def __init__(
        self,
        time_limit: float = 30,
        ngram=(1, 1),
        stop_words='english',
        estimators_=['rfc', 'etc', 'gnb'],
        methods=['independent_models', 'chained_models', 'powerset_models'],
        additional_colms=[],
        validation_split: float = 0.2,
        max_features=5000,
        token_pattern=r'([a-zA-Z0-9/+-]{1,})',
        abbr_dict={},
        df_pred=pd.DataFrame()
    ):
        self.estimators_ = estimators_
        self.methods = methods
        self.additional_colms = additional_colms
        self.validation_split = validation_split
        self.token_pattern = token_pattern
        self.max_features = max_features
        self.stop_words = stop_words
        self.abbr_dict = abbr_dict
        self.ngram = ngram
        self.df_pred = df_pred
        self.time_limit = time_limit
        self.best_approach_dict = {
            "acc": 0
        }
        # logging.basicConfig(level=logging.DEBUG, encoding='utf-8')
        logging.basicConfig(level=logging.DEBUG)

    def expand_abbr(row, _dict):
        '''Replace full words'''
        try:
            res = ' '.join([_dict.get(t, t) for t in row.split()])
        except:
            print(f'Unable to expand abbreviations in {row}')
            res = row
        return res

    def get_dtype(self, df: DataFrame):
        cat_columns = list(df.select_dtypes(include='object').columns)
        num_columns = list(df.select_dtypes(include='number').columns)
        text_columns = []
        for col in cat_columns:
            if df[col].str.len().max() > 99:
                cat_columns.remove(col)
                text_columns.append(col)
        return cat_columns, num_columns, text_columns

    def prep_numeric_cols(
        self,
        X: DataFrame
    ):
        """Standardize and normalize numeric input columns
        """
        # Standardize the data
        scaler = StandardScaler()

        # Storing column names since column names are lost after scaling
        cols = X.columns
        df_num = X[cols].copy()
        # Fit and Transform the train data and test data
        df_num[cols] = scaler.fit_transform(df_num[cols])
        # Normailze the train and test data
        pt = PowerTransformer(method='yeo-johnson')
        df_num[cols] = pt.fit_transform(df_num[cols])
        return df_num, scaler, pt

    def prep_cat_cols(
        self,
        X: DataFrame,
        obj_cols: List
    ):
        """Encode categorical columns of string type

        Args:
            X (DataFrame): Input dataframe for the train data

        Returns:
            Tuple: Label encoded dataframe and \
            a dictionary of column with encoded labels
        """
        le_cat_dict = {}
        obj_df = pd.DataFrame()
        for col in obj_cols:
            val_count = X[col].value_counts().loc[lambda x: x > 1].count()
            if val_count > 25:
                X.drop([col], axis=1, inplace=True)
            else:
                le = LabelEncoder()
                obj_df[col] = le.fit_transform(X[col])
                le_cat_dict[col] = le
        return obj_df, le_cat_dict

    def tl(
        self,
        t0
    ):
        """Returns the difference between current date/time and t0 in seconds
        """
        return timedelta(seconds=round(time.time()-t0, 0))

    def vectorize_input(
        self,
        input_data,
        input_colm
    ):
        """Returns vector represenations of the input text.
        """
        ps = PorterStemmer()
        input_list = input_data[input_colm].tolist()
        stemmed_list = [' '.join([ps.stem(w) for w in ip.split()])
                        for ip in input_list]
        token_pattern = self.token_pattern
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_features=self.max_features,
            token_pattern=token_pattern)
        X = vectorizer.fit_transform(stemmed_list)
        logging.info(f'Number of features in vectorizer = '
                        f'{len(vectorizer.get_feature_names())}')
        return X, vectorizer

    def instantiate_model(
        self,
        model
    ):
        """Returns an instance of the required model.
        """
        if model == 'knn':
            return KNeighborsClassifier(n_neighbors=3)
        elif model == 'dtc':
            return DecisionTreeClassifier(max_leaf_nodes=8, random_state=13)
        elif model == 'gnb':
            return GaussianNB()
        elif model == 'mnb':
            return MultinomialNB()
        elif model == 'rfc':
            return RandomForestClassifier(n_estimators=100,
                                          max_depth=3,
                                          random_state=13)
        elif model == 'abc':
            return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
                                      max_depth=3),
                                      n_estimators=200,
                                      learning_rate=0.5,
                                      random_state=13)
        elif model == 'gbc':
            return GradientBoostingClassifier(n_estimators=100,
                                              learning_rate=0.5,
                                              max_depth=2,
                                              random_state=13)
        elif model == 'etc':
            return ExtraTreesClassifier(n_estimators=50,
                                        random_state=13)
        elif model == 'lrc':
            return LogisticRegression()
        else:
            logging.error(f"Invalid model. Select from \
            {['lrc', 'knn', 'dtc', 'gnb', 'mnb', 'rfc', 'abc', 'gbc', 'etc']}")
            return None

    def prep_input(
        self,
        X: pd.DataFrame
    ):
        """[summary]
        """
        # Store column names in list based on their datatype
        cat_columns, num_columns, text_columns = self.get_dtype(X)
        self.best_approach_dict["text_columns"] = text_columns

        # Preprocess categorical columns
        if cat_columns:
            df_cat, le_cat = self.prep_cat_cols(X[cat_columns].copy(),
                                                cat_columns)
            self.best_approach_dict["le_cat"] = le_cat

        # Preprocess text columns
        if text_columns:
            ip_vec = {}
            df_text = pd.DataFrame()
            if self.abbr_dict:
                for col in text_columns:
                    X[col] = \
                        X[col].apply(self.expand_abbr, _dict=self.abbr_dict)
            for col in text_columns:
                ip_vector, ip_vectorizer = self.vectorize_input(X, col)
                columns = ip_vectorizer.get_feature_names()
                df_text[columns] = pd.DataFrame(ip_vector.toarray())
                ip_vec[col] = ip_vectorizer
            self.best_approach_dict["vectorizer"] = ip_vec

        # Preprocess numeric columns
        if num_columns:
            df_num, scaler, pt = \
                            self.prep_numeric_cols(X[num_columns].copy())
            self.best_approach_dict["scaler"] = scaler
            self.best_approach_dict["pt"] = pt

        self.best_approach_dict["num_columns"] = \
            df_num.columns.to_list()

        return df_cat, df_text, df_num

    def validate_estimator(self):
        """
        Function to validate and instantiate estimator
        """
        if type(self.estimators_) == list:
            pass
        else:
            temp_est = []
            temp_est.append(self.estimators_)
            self.estimators_ = temp_est

        for index, estimator in enumerate(self.estimators_):
            if type(estimator) == str and len(estimator) == 3:
                self.estimators_[index] = self.instantiate_model(estimator)
            else:
                try:
                    check_estimator(estimator)
                except TypeError:
                    logging.raiseExceptions(
                        "Please pass valid SKLearn estimators")

    def get_approach_object(self, name):
        if name == 'chained_models_approach':
            return ChainedModels()
        elif name == 'independent_models_approach':
            return IndependentModels()
        elif name == 'powerset_labels_approach':
            return PowersetLabels()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):
        """Calculates and returns the best approach among the methods provided.

        Args:
            X (pd.DataFrame): Input dataframe
            y (pd.DataFrame): Dataframe of labels
        Returns:
            [dictionary]: Model artifacts for the best of the 3 approaches
        """
        # Preprocess input columns
        df_cat, df_text, df_num = self.prep_input(X)
        # Combine the preprocessed input dataframes and
        # target columns into a single dataframe
        df = pd.concat([df_cat, df_text, df_num, y], axis=1)
        # Drop null values in the data
        df = df.dropna()
        target_columns = y.columns.tolist()
        # Create a new column with the combined value of all target columns
        df['combined_targets'] = df[target_columns].apply(
                        lambda x: '--'.join(x.dropna().astype(str)), axis=1)
        le_combo = LabelEncoder()
        tgt = le_combo.fit_transform(df.combined_targets.tolist())
        X = df.drop(df[target_columns], axis=1)
        X = X.drop(['combined_targets'], axis=1).reset_index(drop=True)
        # Split the input dataframe into train and test datasets
        _, X_test, _, y_test = train_test_split(X,
                                                tgt,
                                                train_size=0.8,
                                                random_state=7)
        # Get the original labels
        y_test_labels = le_combo.inverse_transform(y_test)

        # Validate and instantiate estimator
        self.validate_estimator()

        time_elapsed = 0
        for model in self.estimators_:
            # Fit the chained models
            if(time_elapsed <= self.time_limit):
                start_time = time.time()
                if('chained_models' in (method.lower()
                                        for method in self.methods)):
                    add_cols = copy.deepcopy(self.additional_colms)
                    chained_obj: ChainedModels = self.get_approach_object('chained_models_approach')
                    accuracy_chained_mdls, res_chained_mdls = \
                        chained_obj.fit(df,
                                X,
                                X_test,
                                y_test_labels,
                                model,
                                target_columns,
                                add_cols)
                    if accuracy_chained_mdls > self.best_approach_dict["acc"]:
                        self.best_approach_dict["acc"] = accuracy_chained_mdls
                        self.best_approach_dict["res"] = \
                            copy.deepcopy(res_chained_mdls)
                        self.best_approach_dict["approach"] = "Chained Models"
                        self.best_approach_dict["model"] = model

                # Fit the independent models
                if('independent_models' in (method.lower()
                                            for method in self.methods)):
                    add_cols = copy.deepcopy(self.additional_colms)
                    independent_modelsobj: IndependentModels = self.get_approach_object('independent_models_approach')
                    accuracy_independent_mdls, res_independent_mdls = \
                        independent_modelsobj.fit(df,
                                                    X,
                                                    X_test,
                                                    y_test_labels,
                                                    model,
                                                    target_columns,
                                                    add_cols
                                                    )
                    if accuracy_independent_mdls > self.best_approach_dict["acc"]:
                        self.best_approach_dict["acc"] = accuracy_independent_mdls
                        self.best_approach_dict["res"] = \
                            copy.deepcopy(res_independent_mdls)
                        self.best_approach_dict["approach"] = "Independent Models"
                        self.best_approach_dict["model"] = model

                # Fit the powerset models
                if('powerset_models' in (method.lower()
                                        for method in self.methods)):
                    add_cols = copy.deepcopy(self.additional_colms)
                    powerset_approach: PowersetLabels = self.get_approach_object('powerset_labels_approach')
                    accuracy_powerset, res_powerset_mdl = \
                        powerset_approach.fit(df,
                                                X,
                                                X_test,
                                                y_test_labels,
                                                model,
                                                target_columns,
                                                add_cols
                                                )
                    if accuracy_powerset > self.best_approach_dict["acc"]:
                        self.best_approach_dict["acc"] = accuracy_powerset
                        # self.best_approach_dict["vectorizer"] = ip_vectorizer
                        self.best_approach_dict["res"] = \
                            copy.deepcopy(res_powerset_mdl)
                        self.best_approach_dict["approach"] = "Powerset Models"
                        self.best_approach_dict["model"] = model
                time_elapsed += (time.time() - start_time)
            else:
                logging.info(f"Execution terminated due to time limit at {time_elapsed}, {self.time_limit+time_elapsed}")
        return self

    def predict(self, X: pd.DataFrame):
        # Fetch the selected column names in training based on their datatype
        cat_columns = list(self.best_approach_dict["le_cat"].keys())
        num_columns = self.best_approach_dict["num_columns"]
        text_columns = self.best_approach_dict["text_columns"]
        X = X.dropna().reset_index()
        df = pd.DataFrame()
        # Preprocess categorical columns
        if cat_columns:
            le_cat_dict = self.best_approach_dict["le_cat"]
            for col in cat_columns:
                le = le_cat_dict[col]
                df[col] = le.transform(X[col])

        # Preprocess numeric columns
        if num_columns:
            X_num = X[num_columns].copy()
            sc = self.best_approach_dict["scaler"]
            X_num[num_columns] = \
                sc.transform(X_num[num_columns])
            pt = self.best_approach_dict["pt"]
            X_num[num_columns] = \
                pt.transform(X_num[num_columns])
            df[num_columns] = X_num.copy()
        # Preprocess text columns
        if text_columns:
            ip_vec = self.best_approach_dict["vectorizer"]
            for col in text_columns:
                vocab = ip_vec[col].vocabulary_
                ip_vectorizer = TfidfVectorizer(vocabulary=vocab)
                ps = PorterStemmer()
                input_list = X[col].squeeze().tolist()
                stemmed_list = [' '.join([ps.stem(w)
                                for w in ip.split()])
                                for ip in input_list]
                ip_vector = ip_vectorizer.fit_transform(stemmed_list)
                columns = ip_vec[col].get_feature_names()
                df[columns] = pd.DataFrame(ip_vector.toarray())
        df = df.dropna()
        if self.best_approach_dict['approach'] == "Powerset Models":
            powerset_approach: PowersetLabels = self.get_approach_object('powerset_labels_approach')
            self.df_pred = powerset_approach.predict(self.best_approach_dict, df)

        if self.best_approach_dict['approach'] == "Independent Models":
            independent_modelsobj: IndependentModels = self.get_approach_object('independent_models_approach')
            self.df_pred = independent_modelsobj.predict(self.best_approach_dict, df)

        if self.best_approach_dict['approach'] == "Chained Models":
            chained_obj: ChainedModels = self.get_approach_object('chained_models_approach')
            self.df_pred = chained_obj.predict(self.best_approach_dict, df)
        return self

    def check_is_fitted(
        self,
        estimator,
        msg=None
    ):
        if isclass(estimator):
            raise TypeError(
                "{} is a class, not an instance.".format(estimator))
        if msg is None:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
        if not hasattr(estimator, "fit"):
            raise TypeError("%s is not an estimator instance." % (estimator))
        fitted = [
            v for v in vars(estimator)
            if v.endswith("_") and not v.startswith("__")
        ]

        if not fitted:
            raise AttributeError(msg % {"name": type(estimator).__name__})

    def predict_proba(
        self,
        X
    ):
        self.check_is_fitted(self)
        # Check if fitted before predict methods implementation
        # Fetch the selected column names in training based on their datatype
        cat_columns = list(self.best_approach_dict["le_cat"].keys())
        num_columns = self.best_approach_dict["num_columns"]
        text_columns = self.best_approach_dict["text_columns"]
        X = X.dropna().reset_index()
        df = pd.DataFrame()
        # Preprocess categorical columns
        if cat_columns:
            le_cat_dict = self.best_approach_dict["le_cat"]
            for col in cat_columns:
                le = le_cat_dict[col]
                df[col] = le.transform(X[col])

        # Preprocess numeric columns
        if num_columns:
            X_num = X[num_columns].copy()
            sc = self.best_approach_dict["scaler"]
            X_num[num_columns] = \
                sc.transform(X_num[num_columns])
            pt = self.best_approach_dict["pt"]
            X_num[num_columns] = \
                pt.transform(X_num[num_columns])
            df[num_columns] = X_num.copy()
        # Preprocess text columns
        if text_columns:
            ip_vec = self.best_approach_dict["vectorizer"]
            for col in text_columns:
                vocab = ip_vec[col].vocabulary_
                ip_vectorizer = TfidfVectorizer(vocabulary=vocab)
                ps = PorterStemmer()
                input_list = X[col].squeeze().tolist()
                stemmed_list = [' '.join([ps.stem(w)
                                for w in ip.split()])
                                for ip in input_list]
                ip_vector = ip_vectorizer.fit_transform(stemmed_list)
                columns = ip_vec[col].get_feature_names()
                df[columns] = pd.DataFrame(ip_vector.toarray())
        df = df.dropna()
        if self.best_approach_dict['approach'] == "Powerset Models":
            mdl = self.best_approach_dict['res']['m1'][0]
            return mdl.predict_proba(df)

        if self.best_approach_dict['approach'] == "Independent Models":
            proba_dict = {}
            for idx in range(1, len(self.best_approach_dict['res'])+1):
                mdl_name = 'm'+str(idx)
                mdl = self.best_approach_dict['res'][mdl_name][0]
                temp_col = 'col'+str(idx)
                proba_dict[temp_col] = mdl.predict_proba(df)
            return proba_dict
        if self.best_approach_dict['approach'] == "Chained Models":
            tgt_cols_list = []
            proba_dict = {}
            for idx in range(1, len(self.best_approach_dict['res'])+1):
                mdl_name = 'm'+str(idx)
                mdl = self.best_approach_dict['res'][mdl_name][0]
                le_tc_tgt = self.best_approach_dict['res'][mdl_name][2]
                if self.best_approach_dict['res'][mdl_name][3]:
                    col_idx = 0
                    le_enc = self.best_approach_dict['res'][mdl_name][3]
                    for tgt, enc in le_enc.items():
                        df[tgt] = enc.transform(tgt_cols_list[col_idx])
                        col_idx += 1
                    predicted_label = pd.Series(
                        le_tc_tgt.inverse_transform(mdl.predict(df)))
                    temp_col = 'col'+str(idx)
                    proba_dict[temp_col] = mdl.predict_proba(df)
                    tgt_cols_list.append(predicted_label)
                else:
                    predicted_label = pd.Series(
                        le_tc_tgt.inverse_transform(
                            mdl.predict(df)))
                    temp_col = 'col'+str(idx)
                    proba_dict[temp_col] = mdl.predict_proba(df)
                    tgt_cols_list.append(predicted_label)
            return proba_dict
