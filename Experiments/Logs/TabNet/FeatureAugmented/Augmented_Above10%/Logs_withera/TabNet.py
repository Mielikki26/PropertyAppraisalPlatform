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
This script trains a TabNet model on the above 10% augmented full datasets.
Repeats each training x times from scratch for 
each dataset using seed = x. Predictions are made 
for each training, final average results and deviation 
are saved in json file inside a logs folder as well as the 
log file, images and a copy of the this script.
"""

random.seed(42)

json_file = 'resultsTabnet_geopy.json' #json file name
fd = r"\Logs\TabNet\Augmented_Above10%\Logs_withera\\" #files_directory
images = True        #flag to create images

#lists to calculate average final results and deviation
avg_mape = []
avg_r2 = []
avg_mae = []

#features and labels list
continuos_features = ["Area", "Baths", "Beds", "Latitude", "Longitude"]
categorical_features = ["country", "suburb", "municipality", "state", "postcode"]
labels = ["Price"]

#directories
cur_dir = os.getcwd()
datasets_dir = str(cur_dir) + r'\Above10%\withera\\'
datasets_list = os.listdir(str(cur_dir) + r'\Above10%\withera\\')

#datasets to use
datasets = ["all perth.csv", "ar properties.csv","co properties.csv", "DC Properties.csv",
            "kc house data.csv", "Melbourne housing.csv","pe properties.csv", "uy properties.csv", "DataBase_Era.csv"]

def time():
    #simple function to return current time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def trainxtimes(X_train_norm, X_test_norm, Y_train_norm, Y_test_norm, scalerx, scalery, enc, filename, x):

    # lists to save results
    mapes = []
    r2s = []
    maes = []

    for i in range(x):
        print("------------------------------------------------------------------")
        print("Using the dataset: " + filename)
        print("Using seed = " + str(i))

        print("Train/val/test division is 64/18/18")
        x_train_norm, x_val_norm, y_train_norm, y_val_norm = train_test_split(X_train_norm, Y_train_norm, train_size=0.8, test_size=0.2,
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

        predictions_norm = tabnet.predict(X_test_norm).reshape(-1, 1)
        predictions = scalery.inverse_transform(predictions_norm)
        y_test = scalery.inverse_transform(Y_test_norm)

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
        x_test_continuos = scalerx.inverse_transform(X_test_norm[:,0:5])
        x_test_categorical = enc.inverse_transform(X_test_norm[:,5:])
        full_array = np.column_stack([x_test_continuos,x_test_categorical,y_test,predictions])
        results_df = pd.DataFrame(full_array, columns=["Area", "Baths", "Beds", "Latitude",
                                                       "Longitude", "country",
                                                       "suburb", "municipality", "state", "postcode",
                                                       "Price", "Prediction"])
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

#training loop
df_all = []
for filename in datasets_list:
    if filename not in datasets:
        continue
    df = pd.read_csv(datasets_dir + filename, index_col=None, header=0)
    df = df.sample(frac=1)
    df['postcode'] = df['postcode'].astype(str)

    df_all.append(df)

df_all = pd.concat(df_all)
df_all = df_all.sample(frac = 1)

df_test = df_all[df_all['country'] == 'Portugal']
df_test = np.array_split(df_test, 2)[0]
#df_all = df_all[~df_all.isin(df_test)].dropna()
df_all = pd.concat([df_all, df_test]).drop_duplicates(keep=False)

X_train_continuos = df_all[continuos_features]
X_train_categorical = df_all[categorical_features]
Y_train = df_all[labels]

X_train_continuos = X_train_continuos.to_numpy()
X_train_categorical = X_train_categorical.to_numpy()
Y_train = Y_train.to_numpy().reshape(-1, 1)

# Normalize the data
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train_categorical)
scalerx = StandardScaler().fit(X_train_continuos)
scalery = StandardScaler().fit(Y_train)

X_train_cont_norm = scalerx.transform(X_train_continuos)
Y_train_norm = scalery.transform(Y_train)
X_train_cat_norm = enc.transform(X_train_categorical).toarray()

X_train_norm = np.concatenate((X_train_cont_norm, X_train_cat_norm), axis=1)
X_train_norm = np.int32(X_train_norm)



X_test_continuos = df_test[continuos_features]
X_test_categorical = df_test[categorical_features]
Y_test = df_test[labels]

X_test_continuos = X_test_continuos.to_numpy()
X_test_categorical = X_test_categorical.to_numpy()
Y_test = Y_test.to_numpy().reshape(-1, 1)

# Normalize the data
X_test_cont_norm = scalerx.transform(X_test_continuos)
Y_test_norm = scalery.transform(Y_test)
X_test_cat_norm = enc.transform(X_test_categorical).toarray()

#concat into a single array
X_test_norm = np.concatenate((X_test_cont_norm, X_test_cat_norm), axis=1)
X_test_norm = np.int32(X_test_norm)

mapes, r2s, maes = trainxtimes(X_train_norm, X_test_norm, Y_train_norm, Y_test_norm, scalerx, scalery, enc, "Era_dataset", 1)
avg_mape.append((mean(mapes), np.std(mapes)))
avg_r2.append((mean(r2s), np.std(r2s)))
avg_mae.append((mean(maes), np.std(maes)))

#Save results in json file:
tabnet_data = {'model': "TabNet", 'datasets': "Era_dataset", 'avg_mape': avg_mape, 'avg_r2': avg_r2, 'avg_mae': avg_mae}

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