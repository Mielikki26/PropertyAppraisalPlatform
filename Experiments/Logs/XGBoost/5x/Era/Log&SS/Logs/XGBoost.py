from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import os
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
import json
import shutil
from sklearn.preprocessing import OneHotEncoder
import random

"""
XGBoost training script:
This script trains a XGBoost model on Era the dataset.
Repeats each training x times from scratch for 
each dataset using seed = x. Predictions are made 
for each training, final average results and deviation 
are saved in json file inside a logs folder as well as the 
log file, images and a copy of the this script.
"""

random.seed(42)

json_file = 'resultsXGB.json' #json file name
fd = r"\Logs\XGBoost\5x\Era\Log&SS\Logs\\" #files_directory
images = True        #flag to create images

#lists to calculate average final results and deviation
avg_mape = []
avg_r2 = []
avg_mae = []

#features and labels list
continuos_features = ['Number of Bedrooms', 'WC', 'Latitude', 'Longitude',
                      'Number of Rooms', 'Typology', 'Floor area sq meter', 'Land area sq meter', 'Energy certification',
              'vista', 'renovado', 'renovação', 'parqueamento', 'garagem', 'obras', 'remodelado',
              'remodelação', 'com elevador', 'box', 'arrenda', 'arrendamento', 'inquilin',
              'financiamento', 'logradouro', 'R/C']
categorical_features = ['Agency', 'Property Type', 'Transaction Type', 'Status', 'Zone', 'Parish', 'County', 'Region']
labels = ["Price"]

#directories
cur_dir = os.getcwd()
datasets_dir = str(cur_dir) + r'\Era_datasets\\'

#datasets to use
datasets = "DataBase_Era.csv"

def time():
    #simple function to return current time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def trainxtimes(X_continuos, X_categorical, Y, filename,x):
    """
    This is the function that trains the model x times
    in a specific dataset and returns the results
    it also creates the images regarding the
    predictions vs actual values

    :param X_continuos: numpy array with the continuos features data
    :param X_categorical: numpy array with the categorical features data
    :param Y: numpy array with the labels data
    :param filename: name of dataset being used
    :param x: number of times to train the dataset
    :return: list with average result values and standard deviation
    """

    original_con = np.copy(X_continuos)
    original_cat = np.copy(X_categorical)
    original_y = np.copy(Y)

    # Normalize X
    indexes_lat = []
    indexes_lon = []
    for idx, i in enumerate(X_continuos):
        if i[3] < 0:
            indexes_lat.append(idx)
            X_continuos[idx][3] *= -1
        if i[4] < 0:
            indexes_lon.append(idx)
            X_continuos[idx][4] *= -1

    X_log = np.log1p(X_continuos)
    Y_log = np.log1p(Y)

    for index, i in enumerate(X_log):
        if index in indexes_lat:
            X_log[index][3] *= -1
        if index in indexes_lon:
            X_log[index][4] *= -1

    #lists to save results
    mapes = []
    r2s = []
    maes = []

    print("Normalization used is Log & StandardScaler")
    scalerx = StandardScaler().fit(X_log)
    scalery = StandardScaler().fit(Y_log)

    # Normalize the data
    X_norm = scalerx.transform(X_log)
    Y_norm = scalery.transform(Y_log)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_categorical)
    X_c = enc.transform(X_categorical).toarray()

    X = np.concatenate((X_norm, X_c), axis=1)
    X = np.float16(X)

    params = {
        'max_depth': 15,
        'tree_method': 'gpu_hist'
    }

    print("XGBoost LOGS WITH FOLLOWING PARAMS:\n" + str(params))

    for i in range(x):
        print("------------------------------------------------------------------")
        print("Using the dataset: " + filename)
        print("Using seed = " + str(i))

        print("Train/val/test division is 70/30")
        x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(X, Y_norm, train_size=0.7, test_size=0.3, random_state=i)

        model = XGBRegressor(seed=42,
                             max_depth=15,
                             tree_method='gpu_hist')

        print("started training at: " + str(time()))
        model.fit(x_train_norm, y_train_norm.ravel())
        print("ended training at: " + str(time()))

        predictions_norm = model.predict(x_test_norm).reshape(-1, 1)
        predictions = scalery.inverse_transform(predictions_norm)
        predictions[:, 0:25] = np.expm1(predictions[:, 0:25])

        y_train, y_test = train_test_split(original_y, train_size=0.7, test_size=0.3, random_state=i)

        if not os.path.exists(cur_dir + fd + '\Images\iter_' + str(i) + '\\'):
            os.makedirs(cur_dir + fd + '\Images\iter_' + str(i) + '\\')

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
        plt.savefig(cur_dir + fd + '\Images\iter_' + str(i) + '\\' + r'PredictionsvsActual_' + filename + '.png')
        plt.close()

        print("Mean squared error is of " + str(mean_squared_error(y_test, predictions)))
        print("Mean absolute error:" + str(mean_absolute_error(y_pred=predictions, y_true=y_test)))
        print("MAPE:" + str(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)))
        print("R2 score:" + str(r2_score(y_pred=predictions, y_true=y_test)))
        print(str(time()))
        print("------------------------------------------------------------------")

        mapes.append(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test))
        r2s.append(r2_score(y_pred=predictions, y_true=y_test))
        maes.append(mean_absolute_error(y_pred=predictions, y_true=y_test))

        if not os.path.exists(cur_dir + fd + '\Predictions\iter_' + str(i) + '\\'):
            os.makedirs(cur_dir + fd + '\Predictions\iter_' + str(i) + '\\')

        con_train, con_test, cat_train, cat_test = train_test_split(original_con, original_cat, train_size=0.7, test_size=0.3, random_state=i)

        full_array = np.column_stack([con_test, cat_test, y_test, predictions])
        results_df = pd.DataFrame(full_array, columns=['Number of Bedrooms', 'WC', 'Latitude', 'Longitude',
                      'Number of Rooms', 'Typology', 'Floor area sq meter', 'Land area sq meter', 'Energy certification',
              'vista', 'renovado', 'renovação', 'parqueamento', 'garagem', 'obras', 'remodelado',
              'remodelação', 'com elevador', 'box', 'arrenda', 'arrendamento', 'inquilin',
              'financiamento', 'logradouro', 'R/C', 'Agency', 'Property Type', 'Transaction Type', 'Status', 'Zone', 'Parish', 'County', 'Region',
                                                       "Price", "Prediction"])
        results_df.to_csv(cur_dir + fd + '\Predictions\iter_' + str(i) + '\\' + filename[:-4] + '_preds.csv',
                          index=False)

        model.save_model(cur_dir + fd + r'\tabnet_model_iter_' + str(i))

    return mapes, r2s, maes

#Create directory, redirect output to log file and save a copy of the script
if not os.path.exists(cur_dir + fd):
    os.makedirs(cur_dir + fd)
stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + fd + r'logXGB.txt', "w")
print("XGBoost Logs:\n")
print("Saving copy of script...")
shutil.copy(__file__, cur_dir + fd + r'XGBoost.py')

print("In this script only the Era dataset is used")

df = pd.read_csv(datasets_dir + datasets, index_col=None, header=0)
df = df.sample(frac=1)

X_continuos = df[continuos_features]
X_categorical = df[categorical_features]
Y = df[labels]

X_continuos = X_continuos.to_numpy()
X_categorical = X_categorical.to_numpy()
Y = Y.to_numpy().reshape(-1, 1)

mapes, r2s, maes = trainxtimes(X_continuos, X_categorical, Y, datasets, 5)
avg_mape.append((mean(mapes), np.std(mapes)))
avg_r2.append((mean(r2s), np.std(r2s)))
avg_mae.append((mean(maes), np.std(maes)))

#Save results in json file:
xgb_data = {'model': "XGBoost", 'datasets': datasets, 'avg_mape': avg_mape, 'avg_r2': avg_r2, 'avg_mae': avg_mae}

if not os.path.exists(cur_dir + fd + json_file):
    open(cur_dir + fd + json_file, 'w').close()

with open(cur_dir + fd + json_file, 'r+') as f:
    try:
        json_data = json.load(f)
    except:
        json_data = []
    json_data.append(xgb_data)
    f.seek(0)
    f.truncate()
    json.dump(json_data, f, indent=4, separators=(',',': '))

sys.stdout.close()
sys.stdout = stdoutOrigin