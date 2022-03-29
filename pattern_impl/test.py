# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# from pathlib import Path
import pandas as pd
from hmlc import HMLC
import time
import os

data_path = ""
_data = os.path.join(data_path, "amazon_reviews_train.csv")
dt = pd.read_csv(_data)
print(f'Rows before dropping null rows: {len(dt):,}')
dt = dt.dropna()
print(f'Rows after dropping null rows: {len(dt):,}')
dt_train = dt[:5000]
dt_val = dt[5000:6001]
input_col_list = ['x_productId', 'x_Title', 'x_userId', 'x_Helpfulness',
                    'x_Score', 'x_Time', 'x_Text']
output_col_list = ['y_Cat1', 'y_Cat2', 'y_Cat3']
hmlc_obj = HMLC()
# Fetch the current time
t0 = time.time()
best_approach = hmlc_obj.fit(dt_train[input_col_list],
                                dt_train[output_col_list])
time_elapsed = best_approach.tl(t0)
predictions = best_approach.predict(dt_val[input_col_list])
pred_out_cols = [col+'_pred' for col in output_col_list]
predictions.df_pred.columns = pred_out_cols
y_pred = predictions.df_pred
proba = best_approach.predict_proba(dt_val[input_col_list])
print(proba)
# predictions.columns = output_col_list
print("*"*50)
print(f"The best estimator and approach is : \
{predictions.best_approach_dict['model']} and \
{predictions.best_approach_dict['approach']}\n\n")

print(f"Total time taken to train models : {time_elapsed}")
print("The predicted labels are:\n")
print(y_pred[:10])
dt_val = dt_val.reset_index(drop=True)
dataset_pred = pd.concat([dt_val, y_pred], axis=1)
output_path = os.path.join(data_path, 'output')
os.makedirs(output_path, exist_ok=True)
file_path = os.path.join(output_path, "predictions.csv")
dataset_pred.to_csv(file_path, index=False)
