import pandas as pd
from pathlib import Path
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score
from statistics import mean
import json
import shutil
import random
from sklearn.preprocessing import OneHotEncoder

"""
TabNet training script:
This script trains a TabNet model on the Era dataset using TotalArea
Repeats each training x times from scratch for 
each dataset using seed = x. Predictions are made 
for each training, final average results and deviation 
are saved in json file inside a logs folder as well as the 
log file, images and a copy of the this script.
"""

random.seed(42)

json_file = 'resultsTabnet_Era_TotalArea.json' #json file name
fd = r"\Logs\TabNet\Era\TotalArea\Logs_Era_TotalArea\\" #files_directory
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
              'financiamento', 'logradouro', 'R/C', 'TotalArea']
categorical_features = ['Agency', 'Property Type', 'Transaction Type', 'Status', 'Zone', 'Parish', 'County', 'Region',]
labels = ["Price"]

#directories
cur_dir = os.getcwd()
datasets_dir = str(cur_dir) + r'\Era_dataset\\'
datasets_list = os.listdir(str(cur_dir) + r'\Era_dataset\\')

#datasets to use
datasets = "DataBase_Era_TotalArea.csv"

def time():
    #simple function to return current time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


def trainxtimes(X_continuos, X_categorical, Y, filename, x):
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

    # lists to save results
    mapes = []
    r2s = []
    maes = []

    print("Normalization used is StandardScaler")
    scalerx = StandardScaler().fit(X_continuos)
    scalery = StandardScaler().fit(Y)

    # Normalize the data
    X_norm = scalerx.transform(X_continuos)
    Y_norm = scalery.transform(Y)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_categorical)
    X_c = enc.transform(X_categorical).toarray()

    X = np.concatenate((X_norm, X_c), axis=1)

    for i in range(x):
        print("------------------------------------------------------------------")
        print("Using the dataset: " + filename)
        print("Using seed = " + str(i))

        print("Train/val/test division is 64/18/18")
        x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(X, Y_norm, train_size=0.8, test_size=0.2, random_state=i)
        x_train_norm, x_val_norm, y_train_norm, y_val_norm = train_test_split(x_train_norm, y_train_norm, train_size=0.8, test_size=0.2,
                                                          random_state=i)

        tabnet = TabNetRegressor(optimizer_params=dict(lr=0.05), n_d=12, n_a=12)

        print('TabNet params:\n')
        print(tabnet.get_params())

        eval_set = [(x_train_norm, y_train_norm), (x_val_norm, y_val_norm)]
        print("started training at: " + str(time()))
        tabnet.fit(
            X_train=x_train_norm, y_train=y_train_norm,
            eval_set=eval_set,
            eval_metric=['rmse'],
            max_epochs=150,
            patience=30,
        )
        print("ended training at: " + str(time()))

        predictions_norm = tabnet.predict(x_test_norm).reshape(-1, 1)
        predictions = scalery.inverse_transform(predictions_norm)
        y_test = scalery.inverse_transform(y_test_norm)

        """
        Create images, only creating for the first seed 
        because its unnecessary to create for all of them
        """
        if images == True and i == 0:
            if not os.path.exists(cur_dir + fd + '\Images\\'):
                os.mkdir(cur_dir + fd + '\Images\\')

            epochs = len(tabnet.history['loss'])
            x_axis = range(0, epochs)
            fig, ax = plt.subplots()
            ax.plot(x_axis, tabnet.history['val_0_rmse'], label='Train')
            ax.plot(x_axis, tabnet.history['val_1_rmse'], label='Validation')
            ax.legend()
            plt.ylabel('RMSE')
            plt.xlabel('Epoch')
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

        # printing interesting data for the log file
        print("Feature importance:")
        #print(list(zip(continuos_features + categorical_features, tabnet.feature_importances_)))
        print("Mean squared error is of " + str(mean_squared_error(y_test, predictions)))
        print("Mean absolute error:" + str(mean_absolute_error(y_pred=predictions, y_true=y_test)))
        print("MAPE:" + str(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)))
        print("R2 score:" + str(r2_score(y_pred=predictions, y_true=y_test)))
        print("------------------------------------------------------------------")

        mapes.append(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test))
        r2s.append(r2_score(y_pred=predictions, y_true=y_test))
        maes.append(mean_absolute_error(y_pred=predictions, y_true=y_test))

        if not os.path.exists(cur_dir + fd + '\Predictions\\'):
            os.mkdir(cur_dir + fd + '\Predictions\\')
        x_test_continuos = scalerx.inverse_transform(x_test_norm[:, 0:26])
        x_test_categorical = enc.inverse_transform(x_test_norm[:, 26:])
        full_array = np.column_stack([x_test_continuos, x_test_categorical, y_test, predictions])
        results_df = pd.DataFrame(full_array, columns=['Number of Bedrooms', 'WC', 'Latitude', 'Longitude',
                                                       'Number of Rooms', 'Typology', 'Floor area sq meter',
                                                       'Land area sq meter', 'Energy certification',
                                                       'vista', 'renovado', 'renovação', 'parqueamento', 'garagem',
                                                       'obras',
                                                       'remodelado',
                                                       'remodelação', 'com elevador', 'box', 'arrenda', 'arrendamento',
                                                       'inquilin',
                                                       'financiamento', 'logradouro', 'R/C', 'TotalArea', 'Agency', 'Property Type',
                                                       'Transaction Type', 'Status', 'Zone', 'Parish', 'County',
                                                       'Region',
                                                       'Price', 'Predictions'])
        results_df.to_csv(cur_dir + fd + '\Predictions\\' + filename[:-4] + '_preds.csv', index=False)

    return mapes, r2s, maes


#Create directory, redirect output to log file and save a copy of the script
if not os.path.exists(cur_dir + fd):
    os.makedirs(cur_dir + fd)
stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + fd + r'logTabNet.txt', "w")
print("TabNet Logs:\n")
print("Saving copy of script...")
shutil.copy(__file__, cur_dir + fd + r'TabNet.py')

print("In this script only the Era dataset is used")

df = pd.read_csv(datasets_dir + datasets, index_col=None, header=0)
df = df.sample(frac=1)

X_continuos = df[continuos_features]
X_categorical = df[categorical_features]
Y = df[labels]

X_continuos = X_continuos.to_numpy()
X_categorical = X_categorical.to_numpy()
Y = Y.to_numpy().reshape(-1, 1)

mapes, r2s, maes = trainxtimes(X_continuos, X_categorical, Y, datasets, 2)
avg_mape.append((mean(mapes), np.std(mapes)))
avg_r2.append((mean(r2s), np.std(r2s)))
avg_mae.append((mean(maes), np.std(maes)))

#Save results in json file:
tabnet_data = {'model': "TabNet", 'datasets': datasets, 'avg_mape': avg_mape, 'avg_r2': avg_r2, 'avg_mae': avg_mae}

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