import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score
from statistics import mean
import json

fd = "\MLPR\Logs1\\" #files_directory
m_mapes = []
m_r2s = []
m_maes = []
features = ["Area", "Baths", "Beds", "Latitude", "Longitude", "Month", "Year"]
labels = ["Price"]
all = labels+features
cur_dir = os.getcwd()
parent_dir = Path(cur_dir).parent.absolute()
datasets_dir = str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\'
datasets_list = os.listdir(str(parent_dir) + r'\Research\Datasets\CreatedDatasets\Datasets_in_use\\')

if not os.path.exists(cur_dir + fd):
    os.mkdir(cur_dir + fd)

stdoutOrigin=sys.stdout
sys.stdout = open(cur_dir + fd + r'logMLPR.txt', "w")
print("MLPR LOGS:\n")

def time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    return current_time


def normalize(df):
    print("Normalization used is StandardScaler")
    # https://stackoverflow.com/questions/60998512/how-to-scale-all-columns-except-last-column
    scalar = StandardScaler()
    # uncomment for price normalization too and comment the one after it
    #standardized_features = pd.DataFrame(scalar.fit_transform(df[all].copy()), columns=all)
    standardized_features = pd.DataFrame(scalar.fit_transform(df[features].copy()), columns=features)
    old_shape = df.shape
    df.drop(features, axis=1, inplace=True)
    df = pd.concat([df, standardized_features], axis=1)
    assert old_shape == df.shape, "something went wrong!"

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean())

    return df

print("start: "+ str(time()))

j = 0
for filename in datasets_list:
    mapes = []
    r2s = []
    maes = []
    print("------------------------------------------------------------------")
    print("Dataset: " + filename)
    df = pd.read_csv(datasets_dir + filename, index_col=None, header=0)

    df = normalize(df)
    df = df.reset_index()

    print("Dataset total size after normalization: " + str(df.shape))

    X = df[features]
    Y = df[labels]

    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=1)
    print("Train/test division is 70/30")

    paramsNN = {
        "activation": ["relu"],
        "solver": ["adam"],
        "learning_rate": ["invscaling", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [2500000],
        "hidden_layer_sizes": [(7, 7)],
        "early_stopping": [True]
    }

    print("MLPR SKLEARN LOGS WITH FOLLOWING PARAMS:\n" + str(paramsNN))

    for i in range(10):
        j += 1
        print("j = " + str(j))

        gsNN = GridSearchCV(estimator=MLPRegressor(), param_grid=paramsNN, cv=5, scoring="neg_mean_squared_error")

        print("Starting search for best params: " + str(time()))
        gsNN.fit(x_train, y_train.ravel())
        print("Best params were the following:" + str(time()))
        print(gsNN.best_params_)
        print("Applying prediction with the params...")

        model = MLPRegressor(**gsNN.best_params_)
        model.fit(x_train, y_train.ravel())

        predictions = model.predict(x_test)

        #try:
        #    loss_values = model.estimator.loss_curve_
        #    plt.plot(loss_values)
        #    plt.savefig(cur_dir + r'\MLPR\\' + r'Loss_NP.png')
        #except:
        #    pass

        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!True vs Preds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #for idx, i in enumerate(y_test):
        #    print(str(y_test[idx]) + " --- " + str(predictions[idx]))
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        #plt.figure(figsize=(10, 10))
        #plt.scatter(y_test, predictions, c='crimson')
        #plt.yscale('log')
        #plt.xscale('log')
        #p1 = max(max(predictions), max(y_test))
        #p2 = min(min(predictions), min(y_test))
        #plt.plot([p1, p2], [p1, p2], 'b-')
        #plt.xlabel('True Values', fontsize=15)
        #plt.ylabel('Predictions', fontsize=15)
        #plt.axis('equal')
        #plt.savefig(cur_dir + r'\MLPR\\' + r'Predictions vs Actual_NP.png')

        print("Mean squared error is of " + str(mean_squared_error(y_test, predictions)))
        print("Mean absolute error:" + str(mean_absolute_error(y_pred=predictions, y_true=y_test)))
        print("MAPE:" + str(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)))
        print("R2 score:" + str(r2_score(y_pred=predictions, y_true=y_test)))
        print(str(time()))
        print("------------------------------------------------------------------")

        mapes.append(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test))
        r2s.append(r2_score(y_pred=predictions, y_true=y_test))
        maes.append(mean_absolute_error(y_pred=predictions, y_true=y_test))

    m_mapes.append((mean(mapes), np.std(mapes)))
    m_r2s.append((mean(r2s), np.std(r2s)))
    m_maes.append((mean(maes), np.std(maes)))

tabnet_data = {'model': "MLPR", 'datasets': datasets_list, 'm_mapes': m_mapes, 'm_r2s': m_r2s, 'm_maes': m_maes}

if not os.path.exists("Results.json"):
    open("Results.json", 'w').close()

with open('Results.json', 'r+') as f:
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