from sklearn.ensemble import RandomForestRegressor
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

"""
Random Forest training script:
This script trains a Random Forest model on the datasets.
Repeats each training x times from scratch for 
each dataset using seed = x. Predictions are made 
for each training, final average results and deviation 
are saved in json file inside a logs folder as well as the 
log file, images and a copy of the this script.
"""

json_file = 'resultsRF.json' #json file name
fd = r"\Logs\RF\5x\SSNormalization\Logs\\" #files_directory
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
datasets_dir = str(cur_dir) + r'\datasets\\'
datasets_list = os.listdir(str(cur_dir) + r'\datasets\\')

#datasets to use
datasets = ["all perth.csv" ,"ar properties.csv", "co properties.csv", "DC Properties.csv", "kc house data.csv",
            "Melbourne housing.csv", "pe properties.csv", "uy properties.csv", "Zameen Property.csv"]

def time():
    #simple function to return current time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def trainxtimes(X, Y, filename,x):
    """
    This is the function that trains the model x times
    in a specific dataset and returns the results
    it also creates the images regarding the
    predictions vs actual values

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

    params = {
        "n_estimators": [200],
        "max_features": ["auto"],
        "max_depth": [16]
    }
    print("RANDOM FOREST LOGS WITH FOLLOWING PARAMS:\n" + str(params))

    for i in range(x):
        print("------------------------------------------------------------------")
        print("Using the dataset: " + filename)
        print("Using seed = " + str(i))

        print("Train/val/test division is 70/30")
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=i)

        # Normalize the data
        x_train_norm = scalerx.transform(x_train)
        y_train_norm = scalery.transform(y_train)
        x_test_norm = scalerx.transform(x_test)

        model = RandomForestRegressor(n_estimators=200,
                                    max_features='auto',
                                    max_depth=16)

        print("started training at: " + str(time()))
        model.fit(x_train_norm, y_train_norm.ravel())
        print("ended training at: " + str(time()))

        predictions_norm = model.predict(x_test_norm).reshape(-1, 1)
        predictions = scalery.inverse_transform(predictions_norm)

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

        print("Feature importance:")
        print(list(zip(features, model.feature_importances_)))
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
        full_array = np.column_stack([x_test, y_test, predictions])
        results_df = pd.DataFrame(full_array,
                                  columns=["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year", "Price",
                                           "Prediction"])
        results_df.to_csv(cur_dir + fd + '\Predictions\iter_' + str(i) + '\\' + filename[:-4] + '_preds.csv',
                          index=False)

    return mapes, r2s, maes

#Create directory, redirect output to log file and save a copy of the script
if not os.path.exists(cur_dir + fd):
    os.makedirs(cur_dir + fd)
stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + fd + r'logRF.txt', "w")
print("RF Logs:\n")
print("Saving copy of script...")
shutil.copy(__file__, cur_dir + fd + r'RandomForest.py')

#training loop
df_all = []
for filename in datasets_list:
    if filename not in datasets:
        continue
    df = pd.read_csv(datasets_dir + filename, index_col=None, header=0)
    df = df.sample(frac=1)

    # save data to later train on all the combined datasets
    df_all.append(df)

    X = df[features]
    Y = df[labels]

    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)

    mapes, r2s, maes = trainxtimes(X, Y, filename, 5)
    avg_mape.append((mean(mapes), np.std(mapes)))
    avg_r2.append((mean(r2s), np.std(r2s)))
    avg_mae.append((mean(maes), np.std(maes)))

#train for the all combined dataset and get results
df_all = pd.concat(df_all)
df_all = df_all.sample(frac = 1)
X_all = df_all[features]
Y_all = df_all[labels]
X_all = X_all.to_numpy()
Y_all = Y_all.to_numpy().reshape(-1, 1)
mapes, r2s, maes = trainxtimes(X_all, Y_all, "all_datasets", 5)
avg_mape.append((mean(mapes), np.std(mapes)))
avg_r2.append((mean(r2s), np.std(r2s)))
avg_mae.append((mean(maes), np.std(maes)))

#Save results in json file:
datasets.append("all_datasets")
tabnet_data = {'model': "RF", 'datasets': datasets, 'avg_mape': avg_mape, 'avg_r2': avg_r2, 'avg_mae': avg_mae}

if not os.path.exists(cur_dir + fd + json_file):
    open(cur_dir + fd + json_file, 'w').close()

with open(cur_dir + fd + json_file, 'r+') as f:
    try:
        json_data = json.load(f)
    except:
        json_data = []
    json_data.append(tabnet_data)
    f.seek(0)
    f.truncate()
    json.dump(json_data, f, indent=4, separators=(',',': '))

sys.stdout.close()
sys.stdout = stdoutOrigin