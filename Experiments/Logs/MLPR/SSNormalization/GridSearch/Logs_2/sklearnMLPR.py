import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
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
MLPRegressor(scikit-learn) gridsearch script:
This script helps hyperparameter tuning by training a 
MLPRegressor model on the datasets using gridsearch.
This is one of the multiple gridsearch scripts used
in order to know how the performance varies depending
on the hyperparameters used.
Repeats each training x times from scratch for 
each dataset using seed = x. The x chosen was 1
because its not expected that the seed changes the
best hyperparameters much and using gridsearch is already 
costly enough.
Logs and a copy of the script are saved in a folder. 
The best parameters on each training can be seen in the log file.
"""

fd = r"\Logs\MLPR\SSNormalization\GridSearch\Logs_2\\" #files_directory

#features and labels list
features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]

#directories
cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\')

#grid
grid = {'hidden_layer_sizes': [(7,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001, 0.05, 0.01],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [0.01],
    'max_iter' : [20]}

def time():
    #simple function to return current time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def train(X, Y, filename):
    """
    This is the function that trains the model
    in a specific dataset using the gridsearch

    :param X: features
    :param Y: labels
    :param filename: name of dataset being used
    """

    print("------------------------------------------------------------------")

    print("Normalization used is StandardScaler")
    scalerx = StandardScaler().fit(X)
    scalery = StandardScaler().fit(Y)

    print("Using the dataset: " + filename)
    print("Using seed = 42")

    print("Train/test division is 70/30")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=42)

    # Normalize the data
    x_train_norm = scalerx.transform(x_train)
    y_train_norm = scalery.transform(y_train)

    #initiate gridsearch using 3fold-cross-validation
    mlpr = GridSearchCV(estimator=MLPRegressor(), param_grid=grid, cv=3, n_jobs=-1)
    search = mlpr.fit(x_train_norm, y_train_norm.ravel())


    print("Best hyperparameters:")
    print(search.best_params_)

    print("------------------------------------------------------------------")


#Create directory, redirect output to log file and save a copy of the script
if not os.path.exists(cur_dir + fd):
    os.makedirs(cur_dir + fd)
stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + fd + r'logMLPR.txt', "w")
print("MLPRegressor Logs:\n")
print("Saving copy of script...")
shutil.copy(__file__, cur_dir + fd + r'sklearnMLPR.py')

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

    train(X, Y, filename)

#train for the all combined dataset and get results
df_all = pd.concat(df_all)
X_all = df_all[features]
Y_all = df_all[labels]
X_all = X_all.to_numpy()
Y_all = Y_all.to_numpy().reshape(-1, 1)
train(X_all, Y_all, "all_datasets")

sys.stdout.close()
sys.stdout = stdoutOrigin