import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score
from statistics import mean
from xgboost import XGBRegressor
import json
import shutil

"""
XGBoost training script:
This script trains a XGBoost model on the datasets.
Repeats each training x times from scratch for 
each dataset using seed = x. Predictions are made 
for each training, final average results and deviation 
are saved in json file inside a logs folder as well as the 
log file, images and a copy of the this script.
"""

json_file = 'resultsXGB_2.json' #json file name
fd = "\Logs\XGBoost\SSNormalization\Logs_2\\" #files_directory
images = True        #flag to create images

#lists to calculate average final results and deviation
avg_mape = []
avg_r2 = []
avg_mae = []

#features and labels list
features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]

#directories
cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\')

def time():
    #simple function to return current time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def trainxtimes(X, Y, filename,x):
    """
    This is the function that trains the model x times
    in a specific dataset and returns the results
    it also creates the images regarding loss curve
    and predictions vs actual values

    :param X: features
    :param Y: labels
    :param filename: name of dataset being used
    :param x: number of times to train the dataset
    :return: list with average result values and standard deviation
    """

    #lists to save results
    mapes = []
    r2s = []
    maes = []

    print("Normalization used is StandardScaler")
    scalerx = StandardScaler().fit(X)
    scalery = StandardScaler().fit(Y)

    for i in range(x):
        print("------------------------------------------------------------------")
        print("Using the dataset: " + filename)
        print("Using seed = " + str(i))

        print("Train/val/test division is 64/18/18")
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=i)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2,
                                                          random_state=i)

        # Normalize the data
        x_train_norm = scalerx.transform(x_train)
        y_train_norm = scalery.transform(y_train)
        x_val_norm = scalerx.transform(x_val)
        y_val_norm = scalery.transform(y_val)
        x_test_norm = scalerx.transform(x_test)

        xgboost = XGBRegressor(gamma=0,
                               max_depth=10,
                               subsample=0.8)

        print('XGBoost params:\n')
        print(xgboost.get_params())

        eval_set = [(x_train_norm, y_train_norm), (x_val_norm, y_val_norm)]
        print("started training at: " + str(time()))
        xgboost.fit(x_train_norm, y_train_norm.ravel(), eval_metric='rmse', eval_set=eval_set)
        print("ended training at: " + str(time()))

        predictions_norm = xgboost.predict(x_test_norm).reshape(-1, 1)
        predictions = scalery.inverse_transform(predictions_norm)

        """
        Create images, only creating for the first seed 
        because its unnecessary to create for all of them
        """
        if images == True and i == 0:
            if not os.path.exists(cur_dir + fd + '\Images\\'):
                os.mkdir(cur_dir + fd + '\Images\\')

            results = xgboost.evals_result()
            epochs = len(results['validation_0']['rmse'])
            x_axis = range(0, epochs)
            fig, ax = plt.subplots()
            ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
            ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
            ax.legend()
            plt.ylabel('RMSE')
            plt.xlabel('Epochs')
            plt.title('Dataset is ' + str(filename))
            plt.savefig(cur_dir + fd + '\Images\\' + r'Loss_' + filename + '.png')
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.scatter(y_test, predictions, c='crimson')
            plt.yscale('log')
            plt.xscale('log')
            p1 = max(max(predictions), max(y_test))
            p2 = min(min(predictions), min(y_test))
            plt.plot([p1, p2], [p1, p2], 'b-')
            plt.xlabel('True Values', fontsize=15)
            plt.ylabel('Predictions', fontsize=15)
            plt.axis('equal')
            plt.savefig(cur_dir + fd + '\Images\\' + r'PredictionsvsActual_' + filename + '.png')
            plt.close()

        #printing interesting data for the log file
        print("Feature importance:")
        print(list(zip(features, xgboost.feature_importances_)))
        print("Mean squared error is of " + str(mean_squared_error(y_test, predictions)))
        print("Mean absolute error:" + str(mean_absolute_error(y_pred=predictions, y_true=y_test)))
        print("MAPE:" + str(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)))
        print("R2 score:" + str(r2_score(y_pred=predictions, y_true=y_test)))
        print("------------------------------------------------------------------")

        mapes.append(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test))
        r2s.append(r2_score(y_pred=predictions, y_true=y_test))
        maes.append(mean_absolute_error(y_pred=predictions, y_true=y_test))

    return mapes, r2s, maes

#Create directory, redirect output to log file and save a copy of the script
if not os.path.exists(cur_dir + fd):
    os.makedirs(cur_dir + fd)
stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + fd + r'logXGBoost.txt', "w")
print("XGBoost Logs:\n")
print("Saving copy of script...")
shutil.copy(__file__, cur_dir + fd + r'XGBoost.py')

#training loop
df_all = []
for filename in datasets_list:
    df = pd.read_csv(datasets_dir + filename, index_col=None, header=0)

    # save data to later train on all the combined datasets
    df_all.append(df)

    X = df[features]
    Y = df[labels]

    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)

    mapes, r2s, maes = trainxtimes(X, Y, filename, 10)
    avg_mape.append((mean(mapes), np.std(mapes)))
    avg_r2.append((mean(r2s), np.std(r2s)))
    avg_mae.append((mean(maes), np.std(maes)))

#train for the all combined dataset and get results
df_all = pd.concat(df_all)
X_all = df_all[features]
Y_all = df_all[labels]
X_all = X_all.to_numpy()
Y_all = Y_all.to_numpy().reshape(-1, 1)
mapes, r2s, maes = trainxtimes(X_all, Y_all, "all_datasets", 10)
avg_mape.append((mean(mapes), np.std(mapes)))
avg_r2.append((mean(r2s), np.std(r2s)))
avg_mae.append((mean(maes), np.std(maes)))

#Save results in json file:
datasets_list.append("all_datasets")
XGBoost_data = {'model': "XGBoost", 'datasets': datasets_list, 'avg_mape': avg_mape, 'avg_r2': avg_r2, 'avg_mae': avg_mae}

if not os.path.exists(cur_dir + fd + json_file):
    open(cur_dir + fd + json_file, 'w').close()

with open(cur_dir + fd + json_file, 'r+') as f:
    try:
        json_data = json.load(f)
    except:
        json_data = []
    json_data.append(XGBoost_data)
    f.seek(0)
    f.truncate()
    json.dump(json_data, f, indent=4, separators=(',',': '))

sys.stdout.close()
sys.stdout = stdoutOrigin